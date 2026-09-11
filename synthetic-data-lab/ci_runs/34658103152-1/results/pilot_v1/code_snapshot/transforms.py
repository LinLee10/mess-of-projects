"""Original implementation of CTGAN mode specific normalization, paper section 4.2.

Scope: numerical feature columns and one categorical class column. Clipping is
recorded rather than silently claimed to be exactly invertible for all outliers.
"""
from __future__ import annotations
import warnings
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from sklearn.exceptions import ConvergenceWarning
from .common import validate_xy,validate_labels


class ModeTransformer:
    def __init__(self,max_modes: int=5,seed: int=17):
        if type(max_modes) is not int or max_modes<1:raise ValueError('Positive mode limit required')
        self.max_modes,self.seed=max_modes,seed

    def fit(self,x,y):
        x,y=validate_xy(x,y);self.n_features=x.shape[1];self.n_classes=len(np.unique(y))
        self.models=[];self.spans=[];offset=0;self.fit_warnings=[]
        for j in range(self.n_features):
            column=x[:,j]
            if np.ptp(column)<1e-12:
                self.models.append({'mean':np.array([column[0]]),'std':np.array([1.0]),'weight':np.array([1.0]),'constant':True})
            else:
                # Standardize only to improve mixture optimization, then preserve the fitted units.
                mu,scale=float(column.mean()),float(column.std())
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always',ConvergenceWarning)
                    m=BayesianGaussianMixture(n_components=min(self.max_modes,len(x)),
                        weight_concentration_prior_type='dirichlet_process',weight_concentration_prior=.001,
                        max_iter=500,n_init=1,reg_covar=1e-5,random_state=self.seed+j).fit(((column-mu)/scale)[:,None])
                self.fit_warnings.extend(str(w.message) for w in caught)
                active=m.weights_>.005
                if not active.any():active[np.argmax(m.weights_)]=True
                weights=m.weights_[active];weights=weights/weights.sum()
                self.models.append({'mean':m.means_[active,0]*scale+mu,
                    'std':np.sqrt(m.covariances_[active,0,0])*scale,'weight':weights,'constant':False})
            k=len(self.models[-1]['mean'])
            self.spans.extend([(offset,1,'tanh'),(offset+1,k,'softmax')]);offset+=1+k
        self.class_start=offset;self.spans.append((offset,self.n_classes,'softmax'))
        self.output_dim=offset+self.n_classes;self.last_clipped=0
        return self

    def transform(self,x,y,seed=17):
        x=np.asarray(x,dtype=float);y=validate_labels(y,self.n_classes)
        if x.shape!=(len(y),self.n_features) or not np.isfinite(x).all():raise ValueError('Invalid transform input')
        rng=np.random.default_rng(seed);result=np.zeros((len(x),self.output_dim),dtype=np.float32);offset=0
        clipped=0
        for j,m in enumerate(self.models):
            means,std,weights=m['mean'],m['std'],m['weight'];k=len(means)
            logp=-.5*((x[:,j,None]-means)/std)**2-np.log(std)+np.log(weights)
            p=np.exp(logp-logp.max(axis=1,keepdims=True));p/=p.sum(axis=1,keepdims=True)
            mode=(rng.random(len(x))[:,None]>np.cumsum(p,axis=1)).sum(axis=1);mode=np.minimum(mode,k-1)
            alpha=(x[:,j]-means[mode])/(4*std[mode]);clipped+=int(np.sum(np.abs(alpha)>.99))
            result[:,offset]=np.clip(alpha,-.99,.99)
            result[np.arange(len(x)),offset+1+mode]=1;offset+=1+k
        result[np.arange(len(x)),self.class_start+y]=1
        self.last_clipped=clipped
        return result

    def inverse(self,z):
        z=np.asarray(z,dtype=float)
        if z.ndim!=2 or z.shape[1]!=self.output_dim or not np.isfinite(z).all():raise ValueError('Invalid encoded table')
        x=np.empty((len(z),self.n_features));offset=0
        for j,m in enumerate(self.models):
            k=len(m['mean']);mode=z[:,offset+1:offset+1+k].argmax(axis=1)
            x[:,j]=z[:,offset]*4*m['std'][mode]+m['mean'][mode]
            if m['constant']:x[:,j]=m['mean'][0]
            offset+=1+k
        y=z[:,self.class_start:].argmax(axis=1).astype(np.int64)
        return x,y

    def to_dict(self):
        return {'max_modes':self.max_modes,'seed':self.seed,'n_features':self.n_features,'n_classes':self.n_classes,
            'models':[{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in m.items()} for m in self.models],
            'spans':self.spans,'class_start':self.class_start,'output_dim':self.output_dim,
            'last_clipped':self.last_clipped,'fit_warnings':self.fit_warnings}

    @classmethod
    def from_dict(cls,value):
        obj=cls(value['max_modes'],value['seed'])
        for key in ['n_features','n_classes','class_start','output_dim','last_clipped','fit_warnings']:
            setattr(obj,key,value[key])
        obj.spans=[tuple(s) for s in value['spans']]
        obj.models=[{k:np.array(v,dtype=float) if k in {'mean','std','weight'} else v for k,v in m.items()} for m in value['models']]
        return obj
