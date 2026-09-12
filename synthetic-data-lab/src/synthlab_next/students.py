"""Actual fixed budget neural students, transparent prediction and checkpoint export."""
from __future__ import annotations
from dataclasses import dataclass,asdict
import json
from pathlib import Path
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,log_loss,confusion_matrix
import torch
from torch import nn
from torch.nn import functional as F

from .common import validate_xy,state_hash,array_hash,write_json,optimize


@dataclass(frozen=True)
class StudentConfig:
    steps: int=250
    batch_size: int=64
    width: int=64
    seed: int=7
    learning_rate: float=.003
    weight_decay: float=.001
    def __post_init__(self):
        for key in ('steps','batch_size','width'):
            if type(getattr(self,key)) is not int or getattr(self,key)<1:raise ValueError('Positive integer student sizes required')
        if self.batch_size%2:raise ValueError('Even batch size required for exact half real augmentation')
        if not np.isfinite(self.learning_rate) or self.learning_rate<=0:raise ValueError('Invalid learning rate')
        if not np.isfinite(self.weight_decay) or self.weight_decay<0:raise ValueError('Invalid regularization')
        if type(self.seed) is not int or self.seed<0:raise ValueError('Nonnegative integer seed required')


class StudentNet(nn.Module):
    def __init__(self,features,classes,width):
        super().__init__();self.layers=nn.Sequential(nn.Linear(features,width),nn.ReLU(),
            nn.Linear(width,max(width//2,2)),nn.ReLU(),nn.Linear(max(width//2,2),classes))
    def forward(self,x):return self.layers(x)


class TrainedStudent:
    def __init__(self,network,mean,scale,config,audit):
        self.network=network;self.mean=np.asarray(mean);self.scale=np.asarray(scale);self.config=config;self.audit=audit
    def predict_proba(self,x):
        x=np.asarray(x,dtype=float)
        if x.ndim!=2 or x.shape[1]!=len(self.mean) or not np.isfinite(x).all():raise ValueError('Invalid inference features')
        with torch.no_grad():
            self.network.eval();z=torch.tensor((x-self.mean)/self.scale,dtype=torch.float32)
            return torch.softmax(self.network(z),dim=1).numpy().astype(float)
    def save(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        torch.save(self.network.state_dict(),path/'weights.pt')
        write_json(path/'model.json',{'type':'tabular_mlp','config':asdict(self.config),'mean':self.mean.tolist(),
            'scale':self.scale.tolist(),'classes':self.audit['n_classes'],'audit':self.audit,
            'warning':'Experimental benchmark model, not production or clinical certification'})


def train_student(x,y,config: StudentConfig,*,real=None,synthetic=False):
    x,y=validate_xy(x,y);started=time.monotonic();classes=len(np.unique(y))
    if real is not None:
        rx,ry=validate_xy(*real)
        if rx.shape[1]!=x.shape[1] or len(np.unique(ry))!=classes:raise ValueError('Mismatched real and synthetic schema')
        mean=(x.mean(0)+rx.mean(0))/2
        variance=(np.mean((x-mean)**2,axis=0)+np.mean((rx-mean)**2,axis=0))/2
    else:rx=ry=None;mean=x.mean(0);variance=x.var(0)
    scale=np.maximum(np.sqrt(variance),1e-8)
    data=torch.tensor((x-mean)/scale,dtype=torch.float32);targets=torch.tensor(y)
    if rx is not None:
        rdata=torch.tensor((rx-mean)/scale,dtype=torch.float32);rtargets=torch.tensor(ry)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed);net=StudentNet(x.shape[1],classes,config.width);initial=state_hash(net)
        optimizer=torch.optim.AdamW(net.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
        losses=[];net.train()
        for _ in range(config.steps):
            n=config.batch_size if rx is None else config.batch_size//2
            idx=torch.randint(len(data),(n,));xb,yb=data[idx],targets[idx]
            if rx is not None:
                ridx=torch.randint(len(rdata),(config.batch_size-n,));xb=torch.cat([xb,rdata[ridx]]);yb=torch.cat([yb,rtargets[ridx]])
            losses.append(optimize(F.cross_entropy(net(xb),yb),optimizer,net.parameters()))
        net.eval()
    audit={'config':asdict(config),'n_classes':classes,'n_features':x.shape[1],'initial_sha256':initial,'final_sha256':state_hash(net),
        'losses':losses,'completed_steps':config.steps,'examples_seen':config.steps*config.batch_size,
        'training_records':len(y),'training_x_sha256':array_hash(x),'training_y_sha256':array_hash(y),
        'real_model_training':True,'pretrained_model':False,'source':'real_and_synthetic' if rx is not None else ('synthetic_only' if synthetic else 'real_only'),
        'real_pool_records':0 if rx is None else len(ry),'real_pool_sha256':None if rx is None else array_hash(rx),
        'real_batch_fraction':.5 if rx is not None else (0. if synthetic else 1.),
        'fit_seconds':time.monotonic()-started,'parameters':sum(p.numel() for p in net.parameters()),
        'normalization':'Training distribution moments, with equal real and synthetic mixture weight for augmentation'}
    if initial==audit['final_sha256']:raise RuntimeError('Student parameters did not update')
    return TrainedStudent(net,mean,scale,config,audit)


def load_student(path):
    path=Path(path);meta=json.loads((path/'model.json').read_text());config=StudentConfig(**meta['config'])
    net=StudentNet(len(meta['mean']),meta['classes'],config.width)
    net.load_state_dict(torch.load(path/'weights.pt',map_location='cpu',weights_only=True))
    if state_hash(net)!=meta['audit']['final_sha256']:raise ValueError('Checkpoint state hash mismatch')
    net.eval()
    return TrainedStudent(net,meta['mean'],meta['scale'],config,meta['audit'])


def classification_metrics(y,probabilities):
    # Validate before conversion: truncating 1.9 to 1 can falsely certify a label.
    y=np.asarray(y);p=np.asarray(probabilities)
    if (y.ndim!=1 or not np.issubdtype(y.dtype,np.number) or np.iscomplexobj(y)
            or not np.isfinite(y).all() or not np.equal(y,np.floor(y)).all()):
        raise ValueError('Evaluation labels must be a one dimensional vector of integer class values')
    if (p.ndim!=2 or p.shape[0]!=len(y) or len(y)==0 or p.shape[1]<2
            or not np.issubdtype(p.dtype,np.number) or np.iscomplexobj(p)
            or not np.isfinite(p).all() or np.any(p<0)):
        raise ValueError('Invalid classification outcomes')
    if np.any(y<0) or np.any(y>=p.shape[1]):
        raise ValueError('Evaluation labels are outside the probability class schema')
    y=y.astype(np.int64);p=p.astype(float)
    if not np.allclose(p.sum(1),1,atol=1e-5):raise ValueError('Probabilities must sum to one')
    p=np.clip(p,1e-12,1);p=p/p.sum(1,keepdims=True);pred=p.argmax(1)
    losses=-np.log(p[np.arange(len(y)),y]);classes=p.shape[1]
    return {'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
        'macro_f1':float(f1_score(y,pred,labels=np.arange(classes),average='macro',zero_division=0)),
        'log_loss':float(log_loss(y,p,labels=np.arange(classes))),
        'per_class_loss':[float(losses[y==c].mean()) if np.any(y==c) else None for c in range(classes)],
        'per_class_recall':[float(np.mean(pred[y==c]==c)) if np.any(y==c) else None for c in range(classes)],
        'confusion_matrix':confusion_matrix(y,pred,labels=np.arange(classes)).tolist(),'n':len(y)}


def fit_logistic(x,y,*,real=None):
    x,y=validate_xy(x,y)
    if real is not None:
        rx,ry=validate_xy(*real);weights=np.r_[np.full(len(x),50/len(x)),np.full(len(rx),50/len(rx))]
        x=np.r_[x,rx];y=np.r_[y,ry]
    else:weights=np.full(len(x),100/len(x))
    mean=np.average(x,axis=0,weights=weights);scale=np.maximum(np.sqrt(np.average((x-mean)**2,axis=0,weights=weights)),1e-8)
    classifier=LogisticRegression(C=1.0,max_iter=2000,solver='lbfgs').fit((x-mean)/scale,y,sample_weight=weights)
    return {'coef':classifier.coef_.tolist(),'intercept':classifier.intercept_.tolist(),
        'classes':classifier.classes_.tolist(),'mean':mean.tolist(),'scale':scale.tolist(),
        'normalization':'Total sample weight fixed at 100 across conditions; augmentation uses 50 real plus 50 synthetic',
        'iterations':classifier.n_iter_.tolist()}


def logistic_predict(model,x):
    x=np.asarray(x,dtype=float);logits=(x-np.asarray(model['mean']))/np.asarray(model['scale'])@np.asarray(model['coef']).T+np.asarray(model['intercept'])
    if logits.shape[1]==1:
        logits=np.c_[np.zeros(len(x)),logits[:,0]]
    logits=logits-logits.max(1,keepdims=True);p=np.exp(logits);return p/p.sum(1,keepdims=True)
