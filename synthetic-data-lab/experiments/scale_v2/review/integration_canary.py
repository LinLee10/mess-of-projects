"""End to end software canary with explicit authored arrays, NOT research data."""
import json,sys,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import fullscale
from audit_scale import audit
from synthlab_next.common import array_hash

def fixture(name,cache):
    rng=np.random.default_rng(27 if name=='wine_red' else 73)
    x=rng.normal(size=(90,2));y=(x[:,0]+.3*x[:,1]>0).astype(np.int64)
    return x,y,{'dataset':name,'source_rows':90,'features':['a','b'],'target_names':['0','1'],'source_x_sha256':array_hash(x),'source_y_sha256':array_hash(y),'fixture_only':True,'real_rows_discarded':0}

with tempfile.TemporaryDirectory() as d:
    root=Path(d)/'canary'
    with patch.object(fullscale,'download_source',fixture),patch.object(fullscale,'CORE',['copula','ddpm']),patch.object(fullscale,'ABLATIONS',[]):
        fullscale.train_case('wine_red',root,Path(d)/'cache',smoke=True)
        fullscale.evaluate_case(root)
        result=audit(root)
    result['fixture_only']=True
    result['purpose']='Software integration, never empirical research evidence'
    print(json.dumps(result,indent=2))
