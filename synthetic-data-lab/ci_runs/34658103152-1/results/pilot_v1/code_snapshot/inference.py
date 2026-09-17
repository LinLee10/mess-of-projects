"""Reload locally trained generator checkpoints without executable pickle objects."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import QuantileTransformer,StandardScaler
import torch
from torch import nn
from .common import state_hash
from .transforms import ModeTransformer
from .generators import (GeneratorConfig,CTGAN,TVAE,TabDDPM,Copula,Interpolation,
    ConditionalGenerator,PackedCritic,VariationalNet,Denoiser,diffusion_schedule)


def load_generator(path):
    path=Path(path);m=json.loads((path/'model.json').read_text());method=m['method']
    if method=='portfolio':
        return PortfolioGenerator(path,m)
    if method=='copula':
        model=Copula(m['seed']);model.n_features=m['n_features'];model.n_classes=m['n_classes'];model.models=[]
        with np.load(path/'model_arrays.npz',allow_pickle=False) as a:
            for c in range(model.n_classes):model.models.append((a[f'values_{c}'].copy(),a[f'mean_{c}'].copy(),a[f'covariance_{c}'].copy()))
    elif method=='smote':
        with np.load(path/'model_arrays.npz',allow_pickle=False) as a:model=Interpolation(m['seed']).fit(a['x'],a['y'])
    elif method in {'ctgan','tvae','ddpm'}:
        cfg=GeneratorConfig(**m['config']);model={'ctgan':CTGAN,'tvae':TVAE,'ddpm':TabDDPM}[method](cfg,m['seed'])
        model.n_features=m['n_features'];model.n_classes=m['n_classes'];model.class_prob=np.asarray(m['class_prob'])
        if method in {'ctgan','tvae'}:
            model.transformer=ModeTransformer.from_dict(m['transformer']);dim=model.transformer.output_dim
            if method=='ctgan':
                model.g=ConditionalGenerator(cfg.latent+model.n_classes,dim,cfg.width)
                model.d=PackedCritic(dim+model.n_classes,cfg.width,cfg.pac)
                model.network=nn.ModuleDict({'generator':model.g,'critic':model.d})
            else:model.net=VariationalNet(dim,cfg.width,cfg.latent,model.n_features);model.network=model.net
        else:
            model.net=Denoiser(model.n_features,model.n_classes,cfg.width,cfg.class_conditioning);model.network=model.net
            model.schedule=diffusion_schedule(cfg.diffusion_steps)
            with np.load(path/'quantiles.npz',allow_pickle=False) as a:
                if cfg.quantile_transform:
                    model.qt=QuantileTransformer(output_distribution='normal',subsample=None)
                    model.qt.quantiles_=a['quantiles'].copy();model.qt.references_=a['references'].copy()
                    model.qt.n_quantiles_=len(model.qt.references_);model.qt.n_features_in_=model.n_features
                else:
                    model.qt=StandardScaler();model.qt.mean_=a['mean'].copy();model.qt.scale_=a['scale'].copy()
                    model.qt.var_=a['variance'].copy();model.qt.n_features_in_=model.n_features
        model.network.load_state_dict(torch.load(path/'weights.pt',map_location='cpu',weights_only=True))
        if state_hash(model.network)!=m['audit']['final_sha256']:raise ValueError('Generator checkpoint hash mismatch')
        model.network.eval()
    else:raise ValueError('This artifact is not a supported fitted generator')
    model.audit=m['audit'];model.sample_audit={}
    return model


def sample_csv(checkpoint,output,count,seed=17):
    if type(count) is not int or not 1<=count<=50000:raise ValueError('Sample count must be an integer from 1 to 50000')
    target=Path(output)
    if target.exists():raise ValueError('Output exists; overwrite refused')
    model=load_generator(checkpoint);prob=getattr(model,'class_prob',np.full(model.n_classes,1/model.n_classes))
    labels=np.random.default_rng(seed).choice(model.n_classes,count,p=prob)
    x,y=model.sample_labels(labels,seed=seed);target.parent.mkdir(parents=True,exist_ok=True)
    schema=Path(checkpoint)/'schema.json'
    features=(json.loads(schema.read_text())['feature_names'] if schema.exists() else [f'feature_{i}' for i in range(model.n_features)])
    import csv
    with target.open('w',newline='') as f:
        w=csv.writer(f);w.writerow([*features,'class_id']);w.writerows([*row,int(label)] for row,label in zip(x,y))
    return {'rows':len(y),'generator':model.method,'output':str(target),'sampling':model.sample_audit}


class PortfolioGenerator:
    """Sample fresh component outputs using a previously frozen curation rule."""
    method='portfolio'
    def __init__(self,path,metadata):
        self.path=Path(path).resolve();self.metadata=metadata
        self.n_features=metadata['n_features'];self.n_classes=metadata['n_classes']
        self.class_prob=np.asarray(metadata['class_prob']);self.audit=metadata['audit'];self.sample_audit={}
        self.components={}
        for name,relative in metadata['components'].items():
            p=(self.path/relative).resolve()
            if not p.is_relative_to(self.path):raise ValueError('Component path escapes portfolio')
            self.components[name]=load_generator(p)
        with np.load(self.path/'reference.npz',allow_pickle=False) as a:
            self.real_x=a['x'].copy();self.real_y=a['y'].copy()
        from .common import array_hash
        if array_hash(self.real_x)!=metadata['reference_x_sha256'] or array_hash(self.real_y)!=metadata['reference_y_sha256']:
            raise ValueError('Portfolio reference hash mismatch')

    def sample_labels(self,labels,seed=17,**kwargs):
        from .common import validate_labels
        from .portfolio import select_portfolio
        labels=validate_labels(labels,self.n_classes)
        # Each source receives four times each requested class count. Selection is
        # bounded and fails rather than changing generators when yield is low.
        requested=np.tile(labels,4)
        # select_portfolio validates complete class schemas, including rare classes.
        for c in range(self.n_classes):
            if not np.any(requested==c):requested=np.r_[requested,np.full(4,c)]
        pools={};component_audits={}
        for i,(name,model) in enumerate(self.components.items()):
            pools[name]=model.sample_labels(requested,seed=seed+100003*i)
            component_audits[name]=model.sample_audit
        x,y,curation=select_portfolio(pools,self.real_x,self.real_y,
            np.asarray(self.metadata['development_losses']),np.asarray(self.metadata['development_counts']),
            labels,seed=seed+811,weighted=self.metadata['weighted'],gated=self.metadata['gated'])
        self.sample_audit={'requested':len(labels),'generated':sum(a['generated'] for a in component_audits.values()),
            'component_sampling':component_audits,'curation':curation,
            'model_updates_during_inference':False}
        return x,y


def predict_csv(checkpoint,input_path,output):
    import csv
    from .students import load_student
    target=Path(output)
    if target.exists():raise ValueError('Prediction output exists; overwrite refused')
    model=load_student(checkpoint)
    schema_path=Path(checkpoint)/'schema.json'
    if not schema_path.exists():raise ValueError('Named input schema is required for CSV inference')
    schema=json.loads(schema_path.read_text());names=schema['feature_names']
    if (not isinstance(names,list) or not names or any(not isinstance(n,str) or not n for n in names)
            or len(set(names))!=len(names) or 'class_id' in names or len(names)!=len(model.mean)):
        raise ValueError('Named feature schema is invalid or does not match the model')
    with Path(input_path).open(newline='') as f:
        reader=csv.DictReader(f);headers=reader.fieldnames
        if (headers is None or len(headers)!=len(set(headers))
                or any(name not in headers for name in names)
                or not set(headers).issubset(set(names)|{'class_id'})):
            raise ValueError('Input columns must be unique named features plus an optional class_id')
        rows=list(reader)
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise ValueError('Every CSV row must have exactly the declared number of columns')
    if not rows:raise ValueError('Input has no observations')
    x=np.array([[float(row[name]) for name in names] for row in rows]);p=model.predict_proba(x)
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['prediction',*[f'p_class_{i}' for i in range(p.shape[1])]])
        w.writerows([int(a.argmax()),*a] for a in p)
    return {'rows':len(p),'output':str(target),'warning':'Experimental classification, not physical validation'}
