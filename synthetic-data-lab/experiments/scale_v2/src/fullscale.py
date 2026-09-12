"""Full source datasets, new frozen test splits, and explicit transfer experiments.

No cap, unused partition, or real-row subsample exists here. Stochastic minibatch
ordering and generative draws remain necessary parts of the algorithms. Every
permitted real training row must receive gradient exposure. No test outcomes are
read until all candidates in this invocation have been frozen.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, math, os, platform, shutil, time, traceback, zipfile
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from urllib.request import Request, urlopen
import numpy as np
import scipy, sklearn, torch
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.io import arff
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from torch.nn import functional as F
from synthlab_next.common import array_hash, state_hash, file_hash, write_json, seal, verify_seal, optimize, validate_xy
from synthlab_next.students import StudentNet, StudentConfig, TrainedStudent
from synthlab_next.portfolio import select_portfolio
from coverage_v2 import CoverageCycle, load_coverage_generators

SOURCES={
 'wine_red':('https://archive.ics.uci.edu/static/public/186/wine+quality.zip',1599,11,'Cortez et al. 2009; UCI 10.24432/C56S3T'),
 'wine_white':('https://archive.ics.uci.edu/static/public/186/wine+quality.zip',4898,11,'Cortez et al. 2009; UCI 10.24432/C56S3T'),
 'dry_bean':('https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip',13611,16,'Koklu and Ozkan 2020; UCI 10.24432/C50S4B'),
 'sensorless':('https://archive.ics.uci.edu/static/public/325/dataset+for+sensorless+drive+diagnosis.zip',58509,48,'Bator 2013; UCI 10.24432/C5VP5F'),
}
CORE=['smote','copula','ctgan','tvae','ddpm']
ABLATIONS=['ctgan_unpacked','ctgan_one_mode','ddpm_standardized']
PORTFOLIOS=['mixture_uniform','mixture_utility','mixture_support','mixture_utility_support']
SYNTHESIS_SEEDS=[101,202,303]
STUDENT_SEEDS=[7,11]


def download_source(dataset,cache):
    url,n,d,credit=SOURCES[dataset];cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    dest=cache/(str(url.split('/')[5])+'.zip')
    if not dest.exists():
        request=Request(url,headers={'User-Agent':'synthetic-data-lab reproducible research/2'})
        with urlopen(request,timeout=120) as response:
            if response.geturl().split('/')[2]!='archive.ics.uci.edu':raise ValueError('Unexpected source host')
            raw=response.read(100_000_001)
        if len(raw)>100_000_000:raise ValueError('Download exceeds declared bound')
        dest.write_bytes(raw)
    with zipfile.ZipFile(dest) as archive:
        if dataset.startswith('wine_'):
            suffix='winequality-'+dataset.split('_')[1]+'.csv'
            path=next(p for p in archive.namelist() if p.endswith(suffix))
            rows=list(csv.reader(io.StringIO(archive.read(path).decode()),delimiter=';'))
            names=rows[0][:-1];matrix=np.array(rows[1:],dtype=float)
            x=matrix[:,:-1];original_y=matrix[:,-1].astype(int)
            # Explicit binary task with identical semantics in both source domains.
            y=(original_y>=6).astype(np.int64);target_names=['quality_below_6','quality_at_least_6']
        elif dataset=='dry_bean':
            path=next(p for p in archive.namelist() if p.endswith('.arff'))
            records,meta=arff.loadarff(io.StringIO(archive.read(path).decode()))
            names=list(records.dtype.names[:-1]);x=np.column_stack([records[k] for k in names]).astype(float)
            labels=records[records.dtype.names[-1]];target_names=sorted(set(v.decode() if isinstance(v,bytes) else str(v) for v in labels))
            y=np.array([target_names.index(v.decode() if isinstance(v,bytes) else str(v)) for v in labels],dtype=np.int64);original_y=y.copy()
        else:
            path=next(p for p in archive.namelist() if p.endswith('Sensorless_drive_diagnosis.txt'))
            matrix=np.loadtxt(io.BytesIO(archive.read(path)));x=matrix[:,:-1];original_y=matrix[:,-1].astype(int)
            y=original_y-1;names=[f'current_feature_{i+1}' for i in range(x.shape[1])];target_names=[str(i+1) for i in range(11)]
    x,y=validate_xy(x,y)
    if x.shape!=(n,d):raise ValueError(f'Full source shape mismatch: {dataset} {x.shape}, expected {(n,d)}')
    return x,y,{'dataset':dataset,'source_url':url,'archive_sha256':file_hash(dest),'source_member':path,'source_rows':n,'features':names,'target_names':target_names,'credit':credit,'license':'CC BY 4.0','source_x_sha256':array_hash(x),'source_y_sha256':array_hash(y),'original_targets_sha256':array_hash(original_y),'target_definition':'quality >= 6' if dataset.startswith('wine') else 'original category','real_rows_discarded':0}


def group_partition(x):
    """Content groups prevent identical inputs, including conflicting labels, leaking."""
    x=np.array(x,dtype='<f8',copy=True);x[x==0]=0 # canonicalize signed zero
    if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('Invalid partition features')
    buckets=np.array([int.from_bytes(hashlib.sha256(b'scale_v2_fixed_split_20260911'+row.tobytes()).digest()[:8],'big')%100 for row in x])
    return {'train':np.flatnonzero(buckets<60),'dev':np.flatnonzero((buckets>=60)&(buckets<80)),'test':np.flatnonzero(buckets>=80)}


def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p)
    if y.ndim!=1 or not np.issubdtype(y.dtype,np.integer) or len(y)==0:raise ValueError('Integer labels required')
    if p.ndim!=2 or len(p)!=len(y) or p.shape[1]<2 or not np.isfinite(p).all() or (p<0).any() or (p>1).any() or not np.allclose(p.sum(1),1,rtol=0,atol=1e-5):raise ValueError('Invalid probabilities')
    if (y<0).any() or (y>=p.shape[1]).any():raise ValueError('Unknown class')
    pred=p.argmax(1);k=p.shape[1];cm=np.zeros((k,k),dtype=np.int64);np.add.at(cm,(y,pred),1)
    counts=cm.sum(1);recall=np.divide(cm.diagonal(),counts,out=np.zeros(k),where=counts>0)
    f1=np.divide(2*cm.diagonal(),cm.sum(0)+counts,out=np.zeros(k),where=(cm.sum(0)+counts)>0)
    q=np.clip(p.astype(float),1e-12,1);q/=q.sum(1,keepdims=True);loss=-np.log(q[np.arange(len(y)),y])
    return {'n':int(len(y)),'correct':int((pred==y).sum()),'accuracy':float((pred==y).mean()),'balanced_accuracy':float(recall[counts>0].mean()),'macro_f1':float(f1.mean()),'log_loss':float(loss.mean()),'per_class_recall':recall.tolist(),'per_class_loss':[float(loss[y==c].mean()) if (y==c).any() else None for c in range(k)],'confusion_matrix':cm.tolist(),'roc_auc':float(roc_auc_score(y,p[:,1])) if k==2 and len(np.unique(y))==2 else None}


def compare_selection(rows):
    return sorted(rows,key=lambda r:(-r['dev']['balanced_accuracy'],r['dev']['log_loss'],r['method']))[0]['method']


def full_student(x,y,eval_x,*,seed,steps,batch=128,real=None,path=None,return_model=False):
    x,y=validate_xy(x,y);rx=ry=None;k=len(np.unique(y));start=time.monotonic()
    if real is None:mean=x.mean(0);var=x.var(0)
    else:
        rx,ry=validate_xy(*real);mean=(x.mean(0)+rx.mean(0))/2;var=((x-mean).var(0)+(rx-mean).var(0))/2+(x.mean(0)-rx.mean(0))**2/4
    scale=np.maximum(np.sqrt(var),1e-8);cycle=CoverageCycle(len(x),seed+10007)
    a=torch.tensor((x-mean)/scale,dtype=torch.float32);b=torch.tensor(y)
    if real is not None:
        rcycle=CoverageCycle(len(rx),seed+20011);ra=torch.tensor((rx-mean)/scale,dtype=torch.float32);rb=torch.tensor(ry)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);net=StudentNet(x.shape[1],k,64);initial=state_hash(net)
        optimizer=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.001);losses=[]
        for _ in range(steps):
            ids=cycle.take(batch if real is None else batch//2);xb,yb=a[ids],b[ids]
            if real is not None:
                ri=rcycle.take(batch//2);xb=torch.cat([xb,ra[ri]]);yb=torch.cat([yb,rb[ri]])
            losses.append(optimize(F.cross_entropy(net(xb),yb),optimizer,net.parameters()))
        net.eval()
    if cycle.counts.min()<1 or (real is not None and rcycle.counts.min()<1):raise RuntimeError('Student did not see every permitted row')
    audit={'initial_sha256':initial,'final_sha256':state_hash(net),'n_classes':k,'n_features':x.shape[1],'losses':losses,'completed_steps':steps,'training_records':len(y),'training_x_sha256':array_hash(x),'training_y_sha256':array_hash(y),'unique_training_rows_seen':int((cycle.counts>0).sum()),'minimum_row_exposures':int(cycle.counts.min()),'maximum_row_exposures':int(cycle.counts.max()),'fit_seconds':time.monotonic()-start,'real_rows':0 if real is None else len(ry),'real_unique_rows_seen':0 if real is None else int((rcycle.counts>0).sum()),'real_x_sha256':None if real is None else array_hash(rx),'real_y_sha256':None if real is None else array_hash(ry),'real_batch_fraction':None if real is None else .5,'examples_seen':steps*batch,'batch_policy':'Full shuffled row cycles'}
    model=TrainedStudent(net,mean,scale,StudentConfig(steps=steps,batch_size=batch,seed=seed),audit)
    if path:model.save(path)
    return (model if return_model else model.predict_proba(eval_x)),audit


def _geometry_predictions(x,names):
    cols={s.lower():i for i,s in enumerate(names)}
    def get(name):return x[:,cols[name.lower()]]
    area,p,major,minor,convex=get('Area'),get('Perimeter'),get('MajorAxisLength'),get('MinorAxisLength'),get('ConvexArea')
    valid=(area>0)&(p>0)&(major>=minor)&(minor>0)&(convex>=area)
    with np.errstate(invalid='ignore',divide='ignore'):
        aspect=major/minor;eq=np.sqrt(4*area/np.pi)
        values={'aspectration':aspect,'aspectratio':aspect,'eccentricity':np.sqrt(1-(minor/major)**2),'equivdiameter':eq,'solidity':area/convex,'roundness':4*np.pi*area/p**2,'compactness':eq/major}
    return {name:v for name,v in values.items() if name in cols},cols,valid


def project_geometry(x,names):
    """Recompose six declared derived measurements, not unknown category labels."""
    out=np.asarray(x,dtype=float).copy();values,cols,valid=_geometry_predictions(out,names)
    if not valid.all():raise ValueError('Invalid geometric primitives; no silent repair of them')
    for name,v in values.items():out[:,cols[name]]=v
    return out


def geometry_errors(x,names):
    x=np.asarray(x);values,cols,valid=_geometry_predictions(x,names)
    errors={};bad=~valid
    for name,v in values.items():
        error=np.abs(x[:,cols[name]]-v)>1e-5*np.maximum(np.abs(v),1e-8)+1e-7
        error|=~np.isfinite(v);errors[name]=float(error.mean());bad|=error
    return {'rows_checked':len(x),'equation_violation_rates':errors,'primitive_invalid_rate':float((~valid).mean()),'any_equation_violation_rate':float(bad.mean()),'guarantee':'Only six named equations and primitive inequalities are checked; class correctness is not established'}


def labels_for(y,n,seed):
    p=np.bincount(y)/len(y);count=np.floor(n*p).astype(int)
    for c in np.argsort(-(n*p-count),kind='stable')[:n-count.sum()]:count[c]+=1
    out=np.repeat(np.arange(len(p)),count);np.random.default_rng(seed).shuffle(out);return out


def draw(model,labels,seed):
    arrays=[];attempts=[]
    for i in range(0,len(labels),1024):
        x,y=model.sample_labels(labels[i:i+1024],seed=seed+i*31,max_batches=80);arrays.append(x);attempts.append(dict(model.sample_audit))
    return np.concatenate(arrays),labels.copy(),{'requested':len(labels),'generated':sum(a.get('generated',0) for a in attempts),'chunks':len(attempts),'chunk_audits':attempts}


def take_prefix_classes(x,y,wanted):
    # A matched synthetic output size is a declared condition, not real-data subsampling.
    out=np.empty((len(wanted),x.shape[1]))
    for c in np.unique(wanted):
        slots=np.flatnonzero(wanted==c);ids=np.flatnonzero(y==c)
        if len(ids)<len(slots):raise ValueError('Insufficient synthesized class yield')
        out[slots]=x[ids[:len(slots)]]
    return out,wanted.copy()


def diagnostics(real,synthetic,names):
    scale=np.maximum(real.std(0),1e-8)
    ks=[float(ks_2samp(real[:,i],synthetic[:,i],method='asymp').statistic) for i in range(real.shape[1])]
    wd=[float(wasserstein_distance(real[:,i],synthetic[:,i])/scale[i]) for i in range(real.shape[1])]
    corr=lambda x:np.nan_to_num(np.corrcoef(x,rowvar=False),nan=0.)
    a,b=corr(real),corr(synthetic);mask=~np.eye(real.shape[1],dtype=bool)
    keys=lambda x:[np.asarray(r,dtype='<f8').tobytes() for r in x]
    reference=set(keys(real));skeys=keys(synthetic)
    return {'all_real_rows_checked':len(real),'all_generated_rows_checked':len(synthetic),'mean_marginal_ks':float(np.mean(ks)),'normalized_wasserstein_mean':float(np.mean(wd)),'correlation_mae':float(np.abs(a-b)[mask].mean()),'exact_reference_copy_fraction':sum(k in reference for k in skeys)/len(skeys),'synthetic_duplicate_fraction':1-len(set(skeys))/len(skeys),'negative_fraction':float((synthetic<0).any(1).mean()),'geometry':geometry_errors(synthetic,names) if 'Area' in names else None}


def data_for(name,root,cache):
    x,y,meta=download_source(name,cache);parts=group_partition(x);base=root/'data'/name;base.mkdir(parents=True)
    for kind,idx in parts.items():
        if len(np.unique(y[idx]))!=len(np.unique(y)):raise ValueError('Partition lost a class; protocol must be revised before testing')
        np.savez_compressed(base/(kind+'.npz'),x=x[idx],y=y[idx],ids=idx)
    meta['counts']={k:len(v) for k,v in parts.items()};meta['unique_feature_rows']=len({r.tobytes() for r in x})
    meta['split_policy']='Fixed feature-content hash buckets: train 60%, dev 20%, test 20%; no real-row cap; all exact input duplicates stay together'
    write_json(base/'source.json',meta)
    # No test arrays handed to fitting code.
    return {'x':x[parts['train']],'y':y[parts['train']],'dx':x[parts['dev']],'dy':y[parts['dev']],'meta':meta}


def train_case(dataset,output,cache,smoke=False):
    root=Path(output).resolve()
    if root.exists():raise ValueError('Output already exists; immutable evidence will not be overwritten')
    root.mkdir(parents=True);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    started=time.monotonic();mod=load_coverage_generators(root/'code_snapshot/coverage_generators.py')
    for f in Path(__file__).parent.glob('*.py'):shutil.copyfile(f,root/'code_snapshot'/f.name)
    env={'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'platform':platform.platform(),'processor':platform.processor(),'cpuinfo':Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').exists() else '', 'torch_build':torch.__config__.show(),'torch_parallel':torch.__config__.parallel_info(),'threads':torch.get_num_threads(),'source_commit':os.environ.get('GITHUB_SHA'),'cuda_used':False,'pretrained_models_used':False,'smoke_only':smoke}
    write_json(root/'environment.json',env)
    part=data_for(dataset,root,cache);n=len(part['y']);names=part['meta']['features'];rx,ry,dx,dy=part['x'],part['y'],part['dx'],part['dy']
    target=None
    if dataset.startswith('wine'):
        other='wine_white' if dataset=='wine_red' else 'wine_red';target=data_for(other,root,cache)
        if names!=target['meta']['features']:raise ValueError('Transfer schema mismatch')
    gsteps=max(1000,math.ceil(5*n/200));ssteps=max(500,math.ceil(9*n/128));psteps=max(250,math.ceil(2*n/128))
    all_methods=CORE+ABLATIONS
    protocol={'version':2,'dataset':dataset,'real_train_rows':n,'real_unused_rows':0,'synthesis_seeds':SYNTHESIS_SEEDS,'student_seeds':STUDENT_SEEDS,'core_methods':CORE,'ablations':ABLATIONS,'portfolios':PORTFOLIOS,'generator_updates':gsteps,'student_updates':ssteps,'probe_updates':psteps,'generator_batch':200,'student_batch':128,'tree_iterations':100,'test_is_closed_until_all_candidates_frozen':True,'all_real_training_rows_must_be_seen':True,'size_curves':['1x all methods','3x ddpm and mixture_utility_support'],'constraints':'No test tuning; full source tables; duplicate-safe split; iteration counts are not equal FLOPs; numerical adaptations not paper reproductions','smoke_only':smoke}
    if smoke:protocol['synthesis_seeds']=[101];protocol['student_seeds']=[7];gsteps=40;ssteps=max(10,math.ceil(9*n/128));psteps=max(10,math.ceil(2*n/128))
    write_json(root/'protocol.json',protocol);rows=[];datasets=[];failures=[];generators=[];replay=[]
    def announce(**v):
        print(json.dumps(v),flush=True)
        with (root/'events.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
    def train_table(method,regime,tx,ty,gen_seed,size=1,real=None,domain=None):
        domain=domain or dataset;td=part if domain==dataset else target
        folder=root/'candidates'/f'{gen_seed}_{method}_{regime}_{size}_{domain}';folder.mkdir(parents=True)
        np.savez_compressed(folder/'training.npz',x=tx,y=ty)
        if real is not None:np.savez_compressed(folder/'real.npz',x=real[0],y=real[1])
        steps=max(ssteps, math.ceil(len(tx)/(64 if real is not None else 128)), math.ceil(len(real[1])/64) if real is not None else 0)
        for seed in protocol['student_seeds']:
            model,audit=full_student(tx,ty,td['dx'],seed=seed,steps=steps,real=real,path=folder/str(seed),return_model=True)
            p=model.predict_proba(td['dx']);np.savez_compressed(folder/str(seed)/'dev.npz',probabilities=p,labels=td['dy'])
            rows.append({'method':method,'regime':regime,'size_multiplier':size,'generation_seed':gen_seed,'student_seed':seed,'student':'mlp','train_domain':dataset,'test_domain':domain,'path':str((folder/str(seed)).relative_to(root)),'data_path':str((folder/'training.npz').relative_to(root)),'real_path':str((folder/'real.npz').relative_to(root)) if real is not None else None,'dev':metrics(td['dy'],p),'steps':steps,'real_rows':0 if real is None else len(real[1]),'training_rows':len(ty)})
        # A stronger second learner; full fitting table and no internal early-stop holdout.
        xx=tx if real is None else np.r_[tx,real[0]];yy=ty if real is None else np.r_[ty,real[1]]
        weights=None if real is None else np.r_[np.full(len(tx),.5/len(tx)),np.full(len(real[1]),.5/len(real[1]))]*len(xx)
        tree=HistGradientBoostingClassifier(max_iter=100,learning_rate=.1,max_leaf_nodes=15,l2_regularization=1.,early_stopping=False,random_state=7)
        tree.fit(xx,yy,sample_weight=weights)
        import pickle
        (folder/'tree').mkdir();(folder/'tree/model.pkl').write_bytes(pickle.dumps(tree,protocol=5))
        tp=tree.predict_proba(td['dx']);np.savez_compressed(folder/'tree/dev.npz',probabilities=tp,labels=td['dy'])
        rows.append({'method':method,'regime':regime,'size_multiplier':size,'generation_seed':gen_seed,'student_seed':7,'student':'hist_gradient_boosting','train_domain':dataset,'test_domain':domain,'path':str((folder/'tree').relative_to(root)),'data_path':str((folder/'training.npz').relative_to(root)),'real_path':str((folder/'real.npz').relative_to(root)) if real is not None else None,'dev':metrics(td['dy'],tp),'steps':100,'real_rows':0 if real is None else len(real[1]),'training_rows':len(ty)})
        announce(stage='students',method=method,regime=regime,domain=domain,seed=gen_seed,size=size)
    # Real controls are not spuriously replicated under generation seeds.
    train_table('real_only','real_only',rx,ry,0)
    if target is not None:
        train_table('target_real_only','real_only',target['x'],target['y'],0,domain=other)
        train_table('source_real_transfer','source_only',rx,ry,0,domain=other)
    for gs in protocol['synthesis_seeds']:
        pool_labels=labels_for(ry,4*n,gs+31);wanted=labels_for(ry,n,gs+37);large=labels_for(ry,3*n,gs+37)
        pools={};tables={};models={};cfg=mod.GeneratorConfig(steps=gsteps,batch_size=200,width=128,latent=64,diffusion_steps=100)
        for method in all_methods:
            start=time.monotonic();folder=root/'generators'/str(gs)/method;folder.mkdir(parents=True)
            announce(stage='generator_start',method=method,seed=gs,real_rows=n)
            try:
                if method=='ctgan_unpacked':model=mod.CTGAN(replace(cfg,pac=1),gs)
                elif method=='ctgan_one_mode':model=mod.CTGAN(replace(cfg,max_modes=1),gs)
                elif method=='ddpm_standardized':model=mod.TabDDPM(replace(cfg,quantile_transform=False),gs)
                elif method in ['copula','smote']:model={'copula':mod.Copula,'smote':mod.Interpolation}[method](gs)
                else:model={'ctgan':mod.CTGAN,'tvae':mod.TVAE,'ddpm':mod.TabDDPM}[method](cfg,gs)
                model.fit(rx,ry);model.save(folder);models[method]=model
                px,py,draw_audit=draw(model,pool_labels,gs+503);pools[method]=(px,py)
                np.savez_compressed(folder/'pool.npz',x=px,y=py)
                tables[method]=take_prefix_classes(px,py,wanted)
                write_json(folder/'audit.json',{'fit':model.audit,'draw':draw_audit,'pool_x_sha256':array_hash(px),'pool_y_sha256':array_hash(py),'elapsed_seconds':time.monotonic()-start})
                generators.append(str(folder.relative_to(root)))
                # Repeated fit in the SAME environment: all rows, same full methodology.
                if gs==protocol['synthesis_seeds'][0] and method=='ddpm':
                    again=mod.TabDDPM(cfg,gs).fit(rx,ry);same=state_hash(again.network)==state_hash(model.network)
                    ax,ay,_=draw(again,wanted,gs+503);bx,by,_=draw(model,wanted,gs+503)
                    replay.append({'method':method,'seed':gs,'same_final_weights':same,'same_generated_draw':bool(np.array_equal(ax,bx)),'first':state_hash(model.network),'second':state_hash(again.network),'scope':'Whole fit and same draw in same runtime, not hardware portability'})
                    again.save(folder/'replay');del again
                announce(stage='generator_complete',method=method,seed=gs,seconds=time.monotonic()-start)
            except Exception as exc:
                record={'stage':'generator','method':method,'seed':gs,'error_type':type(exc).__name__,'message':str(exc)};failures.append(record);write_json(folder/'FAILED.json',record);announce(**record)
        source_order=['copula','ctgan','tvae','ddpm'];losses=[]
        if all(m in tables for m in source_order):
            for m in source_order:
                individual=[]
                for seed in protocol['student_seeds']:
                    pf=root/'probes'/str(gs)/m/str(seed)
                    p,a=full_student(*tables[m],dx,seed=seed,steps=psteps,path=pf)
                    score=metrics(dy,p);individual.append(score['per_class_loss']);np.savez_compressed(pf/'dev.npz',probabilities=p,labels=dy);write_json(pf/'dev.json',score)
                losses.append(np.mean(individual,axis=0))
            for m in PORTFOLIOS:
                try:
                    px,py,a=select_portfolio({s:pools[s] for s in source_order},rx,ry,np.array(losses),np.bincount(dy),wanted,seed=gs+811,weighted=m in ['mixture_utility','mixture_utility_support'],gated=m in ['mixture_support','mixture_utility_support'])
                    tables[m]=(px,py);write_json(root/'portfolios'/str(gs)/(m+'.json'),a)
                except Exception as exc:failures.append({'stage':'portfolio','method':m,'seed':gs,'error_type':type(exc).__name__,'message':str(exc)})
        # Separate primitive rejection from equation reconstruction, using identical rows.
        if dataset=='dry_bean':
            for m in ['copula','ddpm','mixture_utility_support']:
                try:
                    if m=='mixture_utility_support':
                        valid_pools={}
                        for name in source_order:
                            px,py=pools[name];_,_,valid=_geometry_predictions(px,names)
                            valid_pools[name]=(px[valid],py[valid])
                        px,py,pa=select_portfolio(valid_pools,rx,ry,np.array(losses),np.bincount(dy),wanted,seed=gs+811,weighted=True,gated=True)
                        write_json(root/'portfolios'/str(gs)/(m+'_primitive_screen.json'),pa)
                    else:
                        px,py=pools[m];_,_,valid=_geometry_predictions(px,names)
                        px,py=take_prefix_classes(px[valid],py[valid],wanted)
                    tables[m+'_primitive_screen']=(px.copy(),py.copy())
                    tables[m+'_geometry']=(project_geometry(px,names),py.copy())
                except Exception as exc:
                    failures.append({'stage':'projection','method':m,'seed':gs,'error_type':type(exc).__name__,'message':str(exc)})
        for m,(tx,ty) in tables.items():
            data_dir=root/'tables'/str(gs)/m;data_dir.mkdir(parents=True)
            np.savez_compressed(data_dir/'synthetic.npz',x=tx,y=ty)
            write_json(data_dir/'quality.json',diagnostics(rx,tx,names));datasets.append(str(data_dir.relative_to(root)))
            for regime in ['synthetic_only','augment']:train_table(m,regime,tx,ty,gs,real=(rx,ry) if regime=='augment' else None)
        for m in ['ddpm','mixture_utility_support']:
            try:
                if m=='ddpm':tx,ty=take_prefix_classes(*pools['ddpm'],large)
                else:
                    tx,ty,a=select_portfolio({s:pools[s] for s in source_order},rx,ry,np.array(losses),np.bincount(dy),large,seed=gs+811,weighted=True,gated=True)
                    write_json(root/'portfolios'/str(gs)/(m+'_3x.json'),a)
                for regime in ['synthetic_only','augment']:train_table(m,regime,tx,ty,gs,size=3,real=(rx,ry) if regime=='augment' else None)
            except Exception as exc:failures.append({'stage':'3x','method':m,'seed':gs,'error_type':type(exc).__name__,'message':str(exc)})
        if target is not None:
            # Direct source-trained students, source synthetic students, and real target anchors.
            for m in ['copula','ddpm','mixture_utility_support']:
                if m in tables:
                    for regime in ['source_only','target_anchored']:
                        train_table(m,regime,*tables[m],gs,domain=other,real=(target['x'],target['y']) if regime=='target_anchored' else None)
            # No invented new field: repeat three generations and test the adjacent target.
            if 'copula' in tables:
                for anchored in [False,True]:
                    current_x,current_y=tables['copula'];base_label=ry
                    for hop in [2,3]:
                        fx,fy=(np.r_[current_x,target['x']],np.r_[current_y,target['y']]) if anchored else (current_x,current_y)
                        chain=mod.Copula(gs+hop).fit(fx,fy)
                        current_x,current_y,_=draw(chain,wanted,gs+hop*701)
                        chain.save(root/'chains'/str(gs)/f'{anchored}_{hop}')
                        train_table(f'copula_hop_{hop}', 'target_anchor_each_hop' if anchored else 'synthetic_replacement',current_x,current_y,gs,domain=other)
        write_json(root/'progress.json',{'finished_seed':gs,'candidate_students':len(rows),'failures':failures,'elapsed_seconds':time.monotonic()-started})
    # Select once on development. Test results cannot affect this file or candidates.
    groups=defaultdict(list)
    for row in rows:groups[(row['generation_seed'],row['test_domain'],row['student'])].append(row)
    winners=[]
    for key,group in groups.items():
        by=defaultdict(list)
        for r in group:by[(r['method'],r['regime'],r['size_multiplier'])].append(r)
        ranked=[]
        for combo,rs in by.items():ranked.append({'method':'|'.join(map(str,combo)),'dev':{'balanced_accuracy':float(np.mean([r['dev']['balanced_accuracy'] for r in rs])),'log_loss':float(np.mean([r['dev']['log_loss'] for r in rs]))}})
        winners.append({'group':list(key),'winner':compare_selection(ranked),'candidates':ranked})
    write_json(root/'CANDIDATES.json',{'rows':rows,'generator_paths':generators,'quality_paths':datasets,'failures':failures,'winners':winners,'test_evaluated':False,'replay':replay,'elapsed_seconds':time.monotonic()-started})
    paths=[str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()]
    frozen=seal(root,paths)
    print(json.dumps({'stage':'FROZEN','candidate_manifest':frozen['manifest_sha256'],'students':len(rows),'failures':len(failures)}),flush=True)


def evaluate_case(output):
    root=Path(output);frozen=verify_seal(root);index=json.loads((root/'CANDIDATES.json').read_text());out=root/'evaluation'
    if out.exists():raise ValueError('Evaluation is immutable')
    out.mkdir();scores=[];torch.set_num_threads(1)
    from synthlab_next.students import load_student
    import pickle
    tests={};test_labels={}
    for name in SOURCES:
        f=root/'data'/name/'test.npz'
        if f.exists():
            with np.load(f,allow_pickle=False) as data:tests[name]={k:data[k].copy() for k in data.files}
    for row in index['rows']:
        folder=root/row['path'];test=tests[row['test_domain']]
        # The pickle is only our own newly trained tree in a verified sealed directory.
        model=load_student(folder) if row['student']=='mlp' else pickle.loads((folder/'model.pkl').read_bytes())
        p=model.predict_proba(test['x']);score=metrics(test['y'],p);tag=hashlib.sha256(row['path'].encode()).hexdigest()[:24]
        np.savez_compressed(out/(tag+'.npz'),probabilities=p,labels=test['y'],ids=test['ids'])
        scores.append({**row,'test':score,'prediction_path':'evaluation/'+tag+'.npz'})
    write_json(out/'scores.json',{'candidate_manifest':frozen['manifest_sha256'],'rows':scores,'test_used_for_tuning':False})
    holdout_quality=[]
    dataset=json.loads((root/'protocol.json').read_text())['dataset']
    names=read_names=json.loads((root/'data'/dataset/'source.json').read_text())['features']
    for rel in index['quality_paths']:
        with np.load(root/rel/'synthetic.npz',allow_pickle=False) as table:
            quality=diagnostics(tests[dataset]['x'],table['x'],names)
        holdout_quality.append({'path':rel,'reference':'complete frozen real test partition','metrics':quality})
    write_json(out/'quality_holdout.json',holdout_quality)
    seal(out,[p.name for p in out.iterdir() if p.is_file()])
    groups=defaultdict(list)
    for row in scores:groups[(row['train_domain'],row['test_domain'],row['method'],row['regime'],row['size_multiplier'],row['student'])].append(row)
    means=[]
    for key,rs in groups.items():
        byseed=defaultdict(list)
        for r in rs:byseed[r['generation_seed']].append(r['test']['balanced_accuracy'])
        means.append(dict(zip(['train_domain','test_domain','method','regime','size_multiplier','student'],key),conditions=len(rs),mean_balanced_accuracy=float(np.mean([r['test']['balanced_accuracy'] for r in rs])),mean_accuracy=float(np.mean([r['test']['accuracy'] for r in rs])),mean_log_loss=float(np.mean([r['test']['log_loss'] for r in rs])),generation_seed_means={str(k):float(np.mean(v)) for k,v in byseed.items()},test_rows=rs[0]['test']['n'],total_correct_across_repeats=sum(r['test']['correct'] for r in rs),test_replicates=len(rs)))
    write_json(root/'SUMMARY.json',{'status':'completed_with_declared_failures' if index['failures'] else 'completed','source':json.loads((root/'data'/json.loads((root/'protocol.json').read_text())['dataset']/'source.json').read_text()),'candidate_manifest':frozen['manifest_sha256'],'evaluation_manifest':json.loads((out/'FROZEN.json').read_text())['manifest_sha256'],'methods':means,'failures':index['failures'],'replay':index['replay'],'scored_conditions':len(scores),'holdout_quality':holdout_quality,'independence':'One fixed content-group split; three synthesis seeds; two neural student seeds; not independent population samples'})
    print(json.dumps({'stage':'evaluated','conditions':len(scores)}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['train','evaluate']);parser.add_argument('--dataset',choices=list(SOURCES));parser.add_argument('--output',required=True);parser.add_argument('--cache',default='source_cache')
    args=parser.parse_args()
    if args.command=='train':train_case(args.dataset,args.output,args.cache)
    else:evaluate_case(args.output)

if __name__=='__main__':main()
