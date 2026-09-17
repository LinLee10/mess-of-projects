"""Separate checker: executes the worker but never imports its scoring or state loader."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import torch


def read_state(root):
    root=Path(root);r=json.loads((root/'LATEST.json').read_text());p=root/r['file']
    if hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Digest mismatch')
    return torch.load(p,map_location='cpu',weights_only=True)


def same(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def main(worker,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);commands=[]
    def run(name,*args,expected=0):
        cmd=[sys.executable,str(worker),'--output',str(out/name),*args]
        q=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
        commands.append({'argv':cmd,'exit_code':q.returncode,'stderr':q.stderr})
        (out/'commands.json').write_text(json.dumps(commands,indent=2))
        if q.returncode!=expected:raise AssertionError(f'{name}: {q.stderr}')
    run('reference')
    run('interrupted','--interrupt','7',expected=75)
    initial=read_state(out/'interrupted')
    if initial['complete'] or initial['step']!=7 or initial['cursor']==0:raise AssertionError('Must interrupt mid epoch')
    for name in ['rng','optimizer','order','corrupt','config','partial']:
        shutil.copytree(out/'interrupted',out/name)
    # A leftover uncommitted temporary checkpoint must not replace the last good pointer.
    (out/'partial'/'uncommitted.pt.tmp').write_bytes(b'incomplete write')
    run('interrupted','--resume');run('partial','--resume')
    ref=read_state(out/'reference');res=read_state(out/'interrupted')
    if not same(ref,res) or not same(ref,read_state(out/'partial')):raise AssertionError('Resume differs')
    if not np.array_equal(np.load(out/'reference/predictions.npy'),np.load(out/'interrupted/predictions.npy')):raise AssertionError('Predictions differ')
    for i in range(5):
        if sorted(ref['visited'][47*i:47*(i+1)])!=list(range(47)):raise AssertionError('Epoch did not visit every row')
    negative={}
    for name in ['rng','optimizer','order']:
        run(name,'--resume','--omit',name)
        negative[name]=not same(ref['weights'],read_state(out/name)['weights'])
        if not negative[name]:raise AssertionError('Omission control insensitive')
    rr=json.loads((out/'corrupt/LATEST.json').read_text());f=out/'corrupt'/rr['file'];f.write_bytes(f.read_bytes()+b'bad')
    run('corrupt','--resume',expected=1)
    run('config','--resume','--learning-rate','.004',expected=1)
    report={'status':'passed','control_only_not_scientific_training':True,'trained_fixture_rows':47,'epochs':5,
      'steps':ref['step'],'mid_epoch_interruption_exit':75,'all_state_equal':True,'prediction_arrays_exact':True,
      'missing_state_controls_changed_weights':negative,'corrupt_checkpoint_rejected':True,
      'config_mismatch_rejected':True,'uncommitted_temporary_write_ignored':True,'all_rows_visited_each_epoch':True,
      'independence':'Same assistant; separate checker process; production loader not imported',
      'scope':'Same CPU runtime continuation, not cross platform reproducibility'}
    (out/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.worker,a.output)
