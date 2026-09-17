"""One bounded CPU replay. Generated evidence is new, not the historical archive."""
from pathlib import Path
import datetime
import json
import os
import subprocess
import sys


def main():
    root=Path(__file__).resolve().parents[1]
    env={**os.environ,'PYTHONPATH':str(root/'src'),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    for path in (root/'results',root/'models'):
        if path.exists():
            raise ValueError('Fresh checkout required; existing evidence must not be overwritten')
    commands=[
        [sys.executable,'-m','unittest','discover','-s','tests','-v'],
        [sys.executable,'-m','synthlab_next.cli','train','--config','configs/pilot.json','--output','results/pilot_v1'],
        [sys.executable,'-m','synthlab_next.cli','evaluate','--output','results/pilot_v1'],
        [sys.executable,'-m','synthlab_next.cli','analyze','--run','results/pilot_v1','--output','results/analysis_v1'],
        [sys.executable,'review/audit_artifacts.py','--run','results/pilot_v1','--output','results/independent_audit.json'],
        [sys.executable,'scripts/export_wine.py'],
        [sys.executable,'-m','synthlab_next.cli','screen-wine','--input','models/wine_demo/synthetic_wine_1000.csv',
         '--schema','models/wine_demo/selected_generator/schema.json','--output','models/wine_schema_screen'],
    ]
    record={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'source_commit':os.environ.get('GITHUB_SHA'),'run_id':os.environ.get('GITHUB_RUN_ID'),
            'run_attempt':os.environ.get('GITHUB_RUN_ATTEMPT'),
            'evidence_origin':'Fresh CI replay; not the original session artifact',
            'cpu_only':True,'pretrained_language_models':False,'commands':[]}
    log=root/'ci_execution.log'
    with log.open('w') as stream:
        for command in commands:
            stream.write(json.dumps(command)+'\n');stream.flush()
            result=subprocess.run(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=2100)
            record['commands'].append({'argv':command,'exit_code':result.returncode})
            if result.returncode:
                record['status']='failed';(root/'CI_EXECUTION.json').write_text(json.dumps(record,indent=2)+'\n')
                raise RuntimeError('Execution failed; inspect ci_execution.log')
    record['status']='completed';record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (root/'CI_EXECUTION.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
