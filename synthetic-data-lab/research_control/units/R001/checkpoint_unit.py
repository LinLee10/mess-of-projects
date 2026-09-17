"""R001: CPU checkpoint protocol. Tiny fixture is a software test, not research data."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w') as f:
        json.dump(value, f, sort_keys=True, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def save(root, state):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    name = f"step_{state['step']:04d}.pt"
    path = root / name; temp = root / (name + '.tmp')
    with temp.open('wb') as f:
        torch.save(state, f); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)
    atomic_json(root/'LATEST.json', {'file': name, 'sha256': digest(path)})


def load(root):
    root = Path(root).resolve(); ref = json.loads((root/'LATEST.json').read_text())
    path = root/ref['file']
    if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError('Unsafe checkpoint reference')
    if digest(path) != ref['sha256']: raise ValueError('Checkpoint checksum mismatch')
    return torch.load(path, map_location='cpu', weights_only=True)


def train(root, resume=False, interrupt=0, omit='', learning_rate=.003):
    root = Path(root)
    if not resume and root.exists(): raise ValueError('Fresh output must not exist')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.manual_seed(17); random.seed(17); rng = np.random.default_rng(17)
    data = torch.arange(47*5, dtype=torch.float32).reshape(47,5).sin()
    mean, scale = data.mean(0), data.std(0).clamp(min=1e-8)
    x = (data-mean)/scale
    model = torch.nn.Sequential(torch.nn.Linear(5,13), torch.nn.ReLU(), torch.nn.Dropout(.25), torch.nn.Linear(13,5))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=.9)
    config = {'epochs':5, 'batch_size':9, 'lr':learning_rate, 'seed':17}
    input_hash = hashlib.sha256(data.numpy().tobytes()).hexdigest()
    source_hash = digest(__file__)
    epoch, cursor, order, step, losses, visited = 0,0,torch.randperm(len(x)),0,[],[]
    if resume:
        s = load(root)
        if s['config'] != config or s['input_sha256'] != input_hash or s['source_sha256'] != source_hash:
            raise ValueError('Checkpoint configuration, input or source mismatch')
        if not torch.equal(s['mean'],mean) or not torch.equal(s['scale'],scale):raise ValueError('Preprocessing mismatch')
        model.load_state_dict(s['weights'])
        if omit != 'optimizer': optimizer.load_state_dict(s['optimizer'])
        scheduler.load_state_dict(s['scheduler'])
        epoch,cursor,order,step,losses,visited = (s[k] for k in ['epoch','cursor','order','step','losses','visited'])
        if omit != 'rng':
            torch.set_rng_state(s['torch_rng']); rng.bit_generator.state=json.loads(s['numpy_rng_json']);random.setstate(s['python_rng'])
        if omit == 'order': order=order.flip(0)
    while epoch < config['epochs']:
        idx=order[cursor:cursor+config['batch_size']]
        jitter=torch.from_numpy(rng.normal(0,.02,size=(len(idx),5)).astype('float32'))
        noisy=x[idx]+jitter+(random.random()-.5)*.005
        optimizer.zero_grad(set_to_none=True)
        loss=torch.nn.functional.mse_loss(model(noisy),x[idx]);loss.backward();optimizer.step();scheduler.step()
        losses.append(float(loss.detach()));visited.extend(idx.tolist());step+=1;cursor+=len(idx)
        if cursor==len(x):
            epoch+=1;cursor=0
            if epoch<config['epochs']:order=torch.randperm(len(x))
        s={'weights':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
           'torch_rng':torch.get_rng_state(),'numpy_rng_json':json.dumps(rng.bit_generator.state),
           'python_rng':random.getstate(),'epoch':epoch,'cursor':cursor,'order':order,'step':step,
           'losses':losses,'visited':visited,'mean':mean,'scale':scale,'config':config,
           'input_sha256':input_hash,'source_sha256':source_hash,'complete':epoch==config['epochs']}
        save(root,s)
        if interrupt and step==interrupt: os._exit(75)
    model.eval()
    with torch.no_grad():np.save(root/'predictions.npy',model(x).numpy(),allow_pickle=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--interrupt',type=int,default=0);p.add_argument('--omit',choices=['','rng','optimizer','order'],default='')
    p.add_argument('--learning-rate',type=float,default=.003);a=p.parse_args()
    train(a.output,a.resume,a.interrupt,a.omit,a.learning_rate)
