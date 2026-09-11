"""Real fitted generators, not replay providers.

CTGAN and TVAE preserve the mechanisms in Xu et al. (NeurIPS 2019), sections
4.2 to 4.5. TabDDPM uses the numerical class conditional branch of Kotelnikov
et al. (ICML 2023). Architecture sizes, training budgets and datasets are pilot
configurations, not replications of the published experiments. Numerical
features and one categorical target are the explicit supported scope.
"""
from __future__ import annotations
from dataclasses import asdict,dataclass
import math
from pathlib import Path
import time

import numpy as np
from scipy.special import ndtr,ndtri
from scipy.stats import rankdata
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer,StandardScaler
import torch
from torch import nn
from torch.nn import functional as F

from .common import validate_xy,validate_labels,state_hash,array_hash,write_json,optimize
from .transforms import ModeTransformer


@dataclass(frozen=True)
class GeneratorConfig:
    steps: int=1000
    batch_size: int=100
    width: int=128
    latent: int=64
    max_modes: int=5
    pac: int=10
    diffusion_steps: int=100
    class_conditioning: bool=True
    quantile_transform: bool=True
    def __post_init__(self):
        for name in ('steps','batch_size','width','latent','max_modes','pac','diffusion_steps'):
            if type(getattr(self,name)) is not int or getattr(self,name)<1:raise ValueError(f'Invalid {name}')
        if type(self.class_conditioning) is not bool or type(self.quantile_transform) is not bool:raise ValueError('Ablation flags must be booleans')
        if self.batch_size%self.pac:raise ValueError('Batch size must be divisible by pack size')
        if self.width<4 or self.diffusion_steps<2:raise ValueError('Model width and diffusion schedule are too small')


class BaseGenerator:
    method='base'
    def __init__(self,config: GeneratorConfig|None=None,seed: int=17):
        self.config=config or GeneratorConfig();self.seed=seed;self.audit={};self.sample_audit={}

    def _start(self,x,y):
        self.x,self.y=validate_xy(x,y);self.n_features=self.x.shape[1];self.n_classes=len(np.unique(self.y))
        self.class_prob=np.bincount(self.y,minlength=self.n_classes)/len(self.y)
        self.audit={'method':self.method,'seed':self.seed,'config':asdict(self.config),
            'training_x_sha256':array_hash(self.x),'training_y_sha256':array_hash(self.y),
            'training_records':len(self.y),'real_model_training':True,'pretrained_model':False,
            'paper_reproduction':False,'feature_scope':'continuous features and one categorical target',
            'losses':[],'completed_steps':0}
        self.rng=np.random.default_rng(self.seed)

    def _finish(self,started,initial,network,updates=None):
        self.network=network
        self.audit.update(initial_sha256=initial,final_sha256=state_hash(network),
            completed_steps=self.config.steps,optimizer_updates=updates or self.config.steps,
            parameters=sum(p.numel() for p in network.parameters()),fit_seconds=time.monotonic()-started)
        if self.audit['initial_sha256']==self.audit['final_sha256']:raise RuntimeError('Training did not change model weights')
        return self

    def _check_n(self,n):
        if type(n) is not int or n<1:raise ValueError('Positive integer sample count required')
        if not self.audit:raise ValueError('Fit must precede sampling')

    def _draw_for(self,labels,seed):
        return self.sample_native(len(labels),seed=seed)

    def sample_labels(self,labels,seed=17,max_batches=40):
        """Bounded rejection sampling. Never attach an ungenerated class label."""
        labels=validate_labels(labels,self.n_classes)
        if type(max_batches) is not int or max_batches<1:raise ValueError('Positive batch cap required')
        result=np.empty((len(labels),self.n_features));attempted=0;matched=0;calls=0
        for c in range(self.n_classes):
            slots=np.flatnonzero(labels==c);position=0
            for batch in range(max_batches):
                if position==len(slots):break
                size=max(64,2*(len(slots)-position));calls+=1
                x,y=self._draw_for(np.full(size,c,dtype=np.int64),seed+100003*c+batch)
                attempted+=len(x);valid=x[y==c];matched+=len(valid)
                take=min(len(valid),len(slots)-position)
                if take:result[slots[position:position+take]]=valid[:take];position+=take
            if position!=len(slots):
                self.sample_audit={'status':'failed_class_yield','requested':len(labels),'generated':attempted,
                                   'class':c,'accepted_for_class':position,'required_for_class':len(slots)}
                raise RuntimeError(f'{self.method}: insufficient genuine class {c} samples within budget')
        if not np.isfinite(result).all():raise FloatingPointError('Nonfinite generated table')
        self.sample_audit={'status':'completed','requested':len(labels),'generated':attempted,'sampling_batches':calls,
            'condition_matches_including_surplus':matched,'retained_fraction':len(labels)/attempted if attempted else 0,
            'label_policy':'Generated category must match requested category; unmatched rows discarded'}
        return result,labels.copy()

    def save(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        torch.save(self.network.state_dict(),path/'weights.pt')
        metadata={'method':self.method,'config':asdict(self.config),'seed':self.seed,'audit':self.audit,
            'class_prob':self.class_prob.tolist(),'n_features':self.n_features,'n_classes':self.n_classes}
        if hasattr(self,'transformer'):metadata['transformer']=self.transformer.to_dict()
        if hasattr(self,'qt'):
            if self.config.quantile_transform:
                np.savez_compressed(path/'quantiles.npz',quantiles=self.qt.quantiles_,references=self.qt.references_)
            else:
                np.savez_compressed(path/'quantiles.npz',mean=self.qt.mean_,scale=self.qt.scale_,variance=self.qt.var_)
        write_json(path/'model.json',metadata)


def activate(raw,spans,tau=.2):
    parts=[]
    for start,width,kind in spans:
        z=raw[:,start:start+width]
        parts.append(torch.tanh(z) if kind=='tanh' else F.gumbel_softmax(z,tau=tau,hard=False,dim=-1))
    return torch.cat(parts,dim=1)


class ResidualConcat(nn.Module):
    def __init__(self,input_dim,width):
        super().__init__();self.layer=nn.Sequential(nn.Linear(input_dim,width),nn.BatchNorm1d(width),nn.ReLU())
    def forward(self,x):return torch.cat([x,self.layer(x)],dim=1)


class ConditionalGenerator(nn.Module):
    def __init__(self,input_dim,output_dim,width):
        super().__init__();self.layers=nn.Sequential(ResidualConcat(input_dim,width),
            ResidualConcat(input_dim+width,width),nn.Linear(input_dim+2*width,output_dim))
    def forward(self,x):return self.layers(x)


class PackedCritic(nn.Module):
    def __init__(self,input_dim,width=128,pac=10):
        super().__init__();self.input_dim=input_dim;self.pac=pac
        self.layers=nn.Sequential(nn.Linear(input_dim*pac,width),nn.LeakyReLU(.2),nn.Dropout(.5),
            nn.Linear(width,width),nn.LeakyReLU(.2),nn.Dropout(.5),nn.Linear(width,1))
    def forward(self,x):
        if x.ndim!=2 or x.shape[1]!=self.input_dim or len(x)%self.pac:raise ValueError('Invalid packed critic shape')
        return self.layers(x.reshape(-1,self.pac*self.input_dim))

    def penalty(self,real,fake):
        n,d=real.shape
        alpha=torch.rand(n//self.pac,1,1).expand(-1,self.pac,d).reshape(n,d)
        interpolated=(alpha*real+(1-alpha)*fake).requires_grad_(True)
        scores=self(interpolated)
        gradient=torch.autograd.grad(scores,interpolated,torch.ones_like(scores),create_graph=True)[0]
        return 10*((gradient.reshape(-1,self.pac*d).norm(2,dim=1)-1)**2).mean()


class CTGAN(BaseGenerator):
    method='ctgan'
    def fit(self,x,y):
        self._start(x,y);started=time.monotonic();cfg=self.config
        self.transformer=ModeTransformer(cfg.max_modes,self.seed).fit(self.x,self.y)
        encoded=torch.from_numpy(self.transformer.transform(self.x,self.y,self.seed))
        self.audit['preprocessing_clipped_values']=self.transformer.last_clipped
        self.audit['mixture_convergence_warnings']=self.transformer.fit_warnings
        self.audit['mechanisms']=['variational mixture normalization','log frequency condition sampling',
            'conditional cross entropy','residual concatenation generator','packed critic','Wasserstein gradient penalty','Gumbel softmax']
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.g=ConditionalGenerator(cfg.latent+self.n_classes,self.transformer.output_dim,cfg.width)
            self.d=PackedCritic(self.transformer.output_dim+self.n_classes,cfg.width,cfg.pac)
            network=nn.ModuleDict({'generator':self.g,'critic':self.d});initial=state_hash(network)
            og=torch.optim.Adam(self.g.parameters(),lr=2e-4,betas=(.5,.9),weight_decay=1e-6)
            od=torch.optim.Adam(self.d.parameters(),lr=2e-4,betas=(.5,.9),weight_decay=1e-6)
            logp=np.log(np.bincount(self.y)+1);logp=logp/logp.sum()
            class_rows=[np.flatnonzero(self.y==c) for c in range(self.n_classes)]
            for step in range(cfg.steps):
                labels=self.rng.choice(self.n_classes,cfg.batch_size,p=logp)
                indices=np.array([self.rng.choice(class_rows[c]) for c in labels])
                condition=F.one_hot(torch.tensor(labels),self.n_classes).float()
                noise=torch.randn(cfg.batch_size,cfg.latent)
                raw=self.g(torch.cat([noise,condition],dim=1));fake=activate(raw,self.transformer.spans)
                realc=torch.cat([encoded[indices],condition],dim=1);fakec=torch.cat([fake.detach(),condition],dim=1)
                dloss=self.d(fakec).mean()-self.d(realc).mean()+self.d.penalty(realc,fakec)
                optimize(dloss,od,self.d.parameters())
                # Fresh noise and an explicit conditional likelihood penalty for generator optimization.
                raw=self.g(torch.cat([torch.randn(cfg.batch_size,cfg.latent),condition],dim=1))
                fake=activate(raw,self.transformer.spans)
                gloss=-self.d(torch.cat([fake,condition],dim=1)).mean()+F.cross_entropy(raw[:,self.transformer.class_start:],torch.tensor(labels))
                self.audit['losses'].append(optimize(gloss,og,self.g.parameters()))
            self.g.eval();self.d.eval()
        return self._finish(started,initial,network,updates=2*cfg.steps)

    def _draw_for(self,labels,seed):
        labels=validate_labels(labels,self.n_classes)
        with torch.random.fork_rng(devices=[]),torch.no_grad():
            torch.manual_seed(seed);self.g.eval()
            cond=F.one_hot(torch.tensor(labels),self.n_classes).float()
            raw=self.g(torch.cat([torch.randn(len(labels),self.config.latent),cond],dim=1))
            z=activate(raw,self.transformer.spans).numpy()
        return self.transformer.inverse(z)

    def sample_native(self,n,seed=17):
        self._check_n(n);labels=np.random.default_rng(seed).choice(self.n_classes,n,p=self.class_prob)
        return self._draw_for(labels,seed)


class VariationalNet(nn.Module):
    def __init__(self,dim,width,latent,n_numeric):
        super().__init__();self.encoder=nn.Sequential(nn.Linear(dim,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU())
        self.mu=nn.Linear(width,latent);self.logvar=nn.Linear(width,latent)
        self.decoder=nn.Sequential(nn.Linear(latent,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,dim))
        self.sigma=nn.Parameter(torch.full((n_numeric,),.1))
    def forward(self,x):
        h=self.encoder(x);mu=self.mu(h);logvar=self.logvar(h).clamp(-12,12)
        z=mu+torch.exp(.5*logvar)*torch.randn_like(mu)
        return self.decoder(z),mu,logvar


class TVAE(BaseGenerator):
    method='tvae'
    def fit(self,x,y):
        self._start(x,y);started=time.monotonic();cfg=self.config
        self.transformer=ModeTransformer(cfg.max_modes,self.seed).fit(self.x,self.y)
        data=torch.from_numpy(self.transformer.transform(self.x,self.y,self.seed))
        self.audit['preprocessing_clipped_values']=self.transformer.last_clipped
        self.audit['mixture_convergence_warnings']=self.transformer.fit_warnings
        self.audit['mechanisms']=['variational mixture normalization','Gaussian latent posterior','reparameterization',
            'learned numerical reconstruction variance','categorical likelihood','KL regularization']
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.net=VariationalNet(self.transformer.output_dim,cfg.width,cfg.latent,self.n_features)
            initial=state_hash(self.net);optimizer=torch.optim.Adam(self.net.parameters(),lr=1e-3,weight_decay=1e-5)
            for step in range(cfg.steps):
                batch=data[torch.randint(len(data),(cfg.batch_size,))]
                raw,mu,logvar=self.net(batch);reconstruction=torch.zeros(len(batch));numeric=0
                for start,width,kind in self.transformer.spans:
                    if kind=='tanh':
                        sigma=self.net.sigma[numeric]
                        reconstruction+=((batch[:,start]-torch.tanh(raw[:,start]))**2/(2*sigma**2)+torch.log(sigma));numeric+=1
                    else:reconstruction+=F.cross_entropy(raw[:,start:start+width],batch[:,start:start+width].argmax(1),reduction='none')
                kl=-.5*torch.sum(1+logvar-mu.square()-logvar.exp(),dim=1)
                loss=(reconstruction+kl).mean()
                self.audit['losses'].append(optimize(loss,optimizer,self.net.parameters()))
                with torch.no_grad():self.net.sigma.clamp_(.01,1.0)
            self.net.eval()
        return self._finish(started,initial,self.net)

    def sample_native(self,n,seed=17):
        self._check_n(n)
        with torch.random.fork_rng(devices=[]),torch.no_grad():
            torch.manual_seed(seed);self.net.eval();raw=self.net.decoder(torch.randn(n,self.config.latent))
            pieces=[];numeric=0
            for start,width,kind in self.transformer.spans:
                if kind=='tanh':
                    # The original generative likelihood samples a numerical residual with learned variance.
                    part=torch.tanh(raw[:,start:start+1])+self.net.sigma[numeric]*torch.randn(n,1);numeric+=1
                    pieces.append(part)
                else:
                    p=torch.softmax(raw[:,start:start+width],dim=1)
                    sampled=torch.multinomial(p,1).squeeze(1);pieces.append(F.one_hot(sampled,width).float())
            z=torch.cat(pieces,dim=1).numpy()
        return self.transformer.inverse(z)


def diffusion_schedule(steps: int) -> dict[str,torch.Tensor]:
    if type(steps) is not int or steps<2:raise ValueError('At least two diffusion steps required')
    grid=torch.linspace(0,1,steps+1,dtype=torch.float64)
    abar=torch.cos((grid+.008)/1.008*math.pi/2).square();abar=abar/abar[0]
    beta=(1-abar[1:]/abar[:-1]).clamp(1e-6,.999).float();alpha=1-beta
    return {'beta':beta,'alpha':alpha,'abar':torch.cumprod(alpha,dim=0)}


def posterior_from_x0(x0,xt,t,schedule):
    beta=schedule['beta'][t];alpha=schedule['alpha'][t];abar=schedule['abar'][t]
    previous=schedule['abar'][t-1] if t>0 else torch.tensor(1.0)
    if t==0:return x0,torch.tensor(0.0)
    mean=(beta*previous.sqrt()*x0+(1-previous)*alpha.sqrt()*xt)/(1-abar)
    variance=beta*(1-previous)/(1-abar)
    return mean,variance


class Denoiser(nn.Module):
    def __init__(self,features,classes,width,class_conditioning=True):
        super().__init__();self.width=width;self.class_conditioning=class_conditioning
        self.x=nn.Linear(features,width);self.y=nn.Embedding(classes,width)
        self.time=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width))
        self.body=nn.Sequential(nn.Linear(width,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,features))
    def forward(self,x,t,y):
        half=self.width//2
        f=torch.exp(-math.log(10000)*torch.arange(half,dtype=torch.float32)/max(half-1,1))
        args=t.float()[:,None]*f[None,:]
        emb=torch.cat([torch.sin(args),torch.cos(args)],dim=1)
        if emb.shape[1]<self.width:emb=F.pad(emb,(0,1))
        return self.body(self.x(x)+(self.y(y) if self.class_conditioning else 0)+self.time(emb))


class TabDDPM(BaseGenerator):
    method='ddpm'
    def fit(self,x,y):
        self._start(x,y);cfg=self.config;started=time.monotonic()
        self.qt=(QuantileTransformer(n_quantiles=min(1000,len(self.x)),output_distribution='normal',subsample=None,random_state=self.seed)
                 if cfg.quantile_transform else StandardScaler())
        data=torch.tensor(self.qt.fit_transform(self.x),dtype=torch.float32);labels=torch.tensor(self.y)
        self.schedule=diffusion_schedule(cfg.diffusion_steps)
        self.audit['mechanisms']=['training only Gaussian quantile transform','class and sinusoidal time conditioning',
            'random timestep noise prediction','full ancestral reverse diffusion']
        self.audit['class_conditioning']=cfg.class_conditioning
        self.audit['quantile_transform']=cfg.quantile_transform
        self.audit['deviations']=['Numerical feature branch only; no multinomial feature diffusion','No EMA or hyperparameter search',
            'Cosine schedule with configured step budget','Predicted clean quantile coordinates clipped to the transform range']
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed);self.net=Denoiser(self.n_features,self.n_classes,cfg.width,cfg.class_conditioning)
            initial=state_hash(self.net);optimizer=torch.optim.AdamW(self.net.parameters(),lr=1e-3,weight_decay=1e-5)
            for step in range(cfg.steps):
                indices=torch.randint(len(data),(cfg.batch_size,));t=torch.randint(cfg.diffusion_steps,(cfg.batch_size,))
                noise=torch.randn(cfg.batch_size,self.n_features);a=self.schedule['abar'][t,None]
                noisy=a.sqrt()*data[indices]+(1-a).sqrt()*noise
                loss=F.mse_loss(self.net(noisy,t,labels[indices]),noise)
                self.audit['losses'].append(optimize(loss,optimizer,self.net.parameters()))
            self.net.eval()
        return self._finish(started,initial,self.net)

    def _draw_for(self,labels,seed):
        labels=validate_labels(labels,self.n_classes);n=len(labels)
        with torch.random.fork_rng(devices=[]),torch.no_grad():
            torch.manual_seed(seed);self.net.eval();z=torch.randn(n,self.n_features);y=torch.tensor(labels)
            for t in reversed(range(self.config.diffusion_steps)):
                epsilon=self.net(z,torch.full((n,),t,dtype=torch.long),y);a=self.schedule['abar'][t]
                x0=((z-(1-a).sqrt()*epsilon)/a.sqrt()).clamp(-5.2,5.2)
                mean,var=posterior_from_x0(x0,z,t,self.schedule)
                z=mean+var.sqrt()*torch.randn_like(z) if t else mean
        return self.qt.inverse_transform(z.numpy()).astype(float),labels.copy()

    def sample_native(self,n,seed=17):
        self._check_n(n);y=np.random.default_rng(seed).choice(self.n_classes,n,p=self.class_prob)
        return self._draw_for(y,seed)

    def sample_labels(self,labels,seed=17,max_batches=40):
        labels=validate_labels(labels,self.n_classes);x,y=self._draw_for(labels,seed)
        self.sample_audit={'status':'completed','requested':len(labels),'generated':len(labels),
            'sampling_batches':1,'reverse_network_calls':self.config.diffusion_steps,'retained_fraction':1.,
            'label_policy':'Class conditional diffusion; categorical label is an input, not a predicted feature'}
        return x,y


class Copula:
    """Class conditional empirical Gaussian copula with shrinkage covariance."""
    method='copula'
    def __init__(self,seed=17):self.seed=seed;self.audit={};self.sample_audit={}
    def fit(self,x,y):
        started=time.monotonic();self.x,self.y=validate_xy(x,y);self.n_features=self.x.shape[1];self.n_classes=len(np.unique(self.y))
        self.models=[]
        for c in range(self.n_classes):
            group=self.x[self.y==c]
            if len(group)<2:raise ValueError('At least two real rows per class required')
            probabilities=np.column_stack([(rankdata(group[:,j],method='average')-.5)/len(group) for j in range(self.n_features)])
            z=ndtri(probabilities);fit=LedoitWolf().fit(z)
            self.models.append((np.sort(group,axis=0),z.mean(axis=0),fit.covariance_))
        self.audit={'method':self.method,'real_model_training':True,'neural_model':False,'seed':self.seed,
            'training_x_sha256':array_hash(self.x),'training_y_sha256':array_hash(self.y),'fit_seconds':time.monotonic()-started,
            'mechanisms':['class conditional empirical marginals','Gaussian copula','Ledoit Wolf covariance shrinkage']}
        return self
    def sample_labels(self,labels,seed=17,**kwargs):
        labels=validate_labels(labels,self.n_classes);rng=np.random.default_rng(seed);result=np.empty((len(labels),self.n_features))
        for c in range(self.n_classes):
            idx=np.flatnonzero(labels==c)
            if not len(idx):continue
            sorted_x,mu,cov=self.models[c];z=rng.multivariate_normal(mu,cov,len(idx),check_valid='raise');u=ndtr(z)
            p=(np.arange(len(sorted_x))+.5)/len(sorted_x)
            for j in range(self.n_features):result[idx,j]=np.interp(u[:,j],p,sorted_x[:,j])
        self.sample_audit={'status':'completed','generated':len(labels),'requested':len(labels),'retained_fraction':1.}
        return result,labels.copy()
    def save(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        arrays={}
        for c,(values,mu,cov) in enumerate(self.models):
            arrays.update({f'values_{c}':values,f'mean_{c}':mu,f'covariance_{c}':cov})
        np.savez_compressed(path/'model_arrays.npz',**arrays)
        write_json(path/'model.json',{'method':self.method,'seed':self.seed,'audit':self.audit,
                                    'n_classes':self.n_classes,'n_features':self.n_features})


class Interpolation:
    """Within class nearest neighbor interpolation; SMOTE style baseline, all classes."""
    method='smote'
    def __init__(self,seed=17):self.seed=seed;self.audit={};self.sample_audit={}
    def fit(self,x,y):
        started=time.monotonic();self.x,self.y=validate_xy(x,y);self.n_features=self.x.shape[1];self.n_classes=len(np.unique(self.y))
        self.groups=[];self.neighbors=[];scale=np.maximum(self.x.std(0),1e-8)
        for c in range(self.n_classes):
            group=self.x[self.y==c]
            if len(group)<2:raise ValueError('At least two real rows per class required')
            indices=NearestNeighbors(n_neighbors=min(6,len(group))).fit(group/scale).kneighbors(group/scale,return_distance=False)
            # Never rely on self being first when rows duplicate.
            choices=[row[row!=i][:5] for i,row in enumerate(indices)]
            self.groups.append(group);self.neighbors.append(choices)
        self.audit={'method':self.method,'seed':self.seed,'real_model_training':True,'neural_model':False,
            'training_x_sha256':array_hash(self.x),'training_y_sha256':array_hash(self.y),'fit_seconds':time.monotonic()-started,
            'adaptation':'SMOTE interpolation applied to all classes under matched class counts, not only minority oversampling'}
        return self
    def sample_labels(self,labels,seed=17,**kwargs):
        labels=validate_labels(labels,self.n_classes);rng=np.random.default_rng(seed);x=np.empty((len(labels),self.n_features))
        for i,c in enumerate(labels):
            a=rng.integers(len(self.groups[c]));b=rng.choice(self.neighbors[c][a]);weight=rng.random()
            x[i]=self.groups[c][a]+weight*(self.groups[c][b]-self.groups[c][a])
        self.sample_audit={'status':'completed','generated':len(labels),'requested':len(labels),'retained_fraction':1.}
        return x,labels.copy()
    def save(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path/'model_arrays.npz',x=self.x,y=self.y)
        write_json(path/'model.json',{'method':self.method,'seed':self.seed,'audit':self.audit})


NEURAL_GENERATORS={'ctgan':CTGAN,'tvae':TVAE,'ddpm':TabDDPM}
