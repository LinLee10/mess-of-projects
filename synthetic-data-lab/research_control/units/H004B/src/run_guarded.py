"""Canonical H004B entry: bound inputs and an external process wall deadline."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from admission import admit, digest

WALL_SECONDS = 1800

def run_bounded(command, timeout_seconds, stdout, stderr, env=None):
    if timeout_seconds <= 0:
        raise ValueError('Positive process budget required')
    child = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env,
                             start_new_session=True)
    timed_out = False
    try:
        code = child.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)
        code = 124
    return {'exit_code':code,'timed_out':timed_out,'pid':child.pid}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','audit','verification','output','execution'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    if sys.flags.optimize:
        raise RuntimeError('Optimized validation forbidden')
    started = time.monotonic()
    for existing in (args.output,args.execution):
        if os.path.lexists(existing):
            raise FileExistsError(existing)
    for target in (args.output,args.execution):
        for source in (args.inputs,args.audit,args.verification):
            if target.resolve().is_relative_to(source.resolve()):
                raise ValueError('Do not write beneath immutable inputs')
    admission = admit(args.audit,args.verification)
    runner = Path(__file__).with_name('run_utility.py')
    command = [sys.executable,'-B',str(runner)]
    for name in ('inputs','audit','verification','output'):
        command.extend(['--'+name,str(getattr(args,name).resolve())])
    args.execution.mkdir(parents=True,exist_ok=False)
    record = {'argv':command,'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'deadline_seconds':WALL_SECONDS,'admission':admission,
              'source_sha256':{p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}}
    with (args.execution/'START.json').open('x') as stream:
        json.dump(record,stream,indent=2,allow_nan=False)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    with (args.execution/'stdout.log').open('x') as out,(args.execution/'stderr.log').open('x') as err:
        record.update(run_bounded(command,WALL_SECONDS-(time.monotonic()-started),out,err,env))
    record.update(elapsed_seconds=time.monotonic()-started,
                  end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    with (args.execution/'EXECUTION.json').open('x') as stream:
        json.dump(record,stream,indent=2,allow_nan=False)
    raise SystemExit(record['exit_code'])

if __name__ == '__main__':
    main()
