"""Run protected recovery acceptance tests and retain real execution evidence."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = 'synthetic-data-lab/research_control/units/H003R/review/test_recovery.py'
EXTERNAL = 'review_acceptance/test_recovery_acceptance.py'
SUBJECT = 'synthetic-data-lab/research_control/units/H003R/review/recover_artifact.py'
PROTECTED = {
    ORIGINAL: '63a3e6d16caeea82986877af2e3007f4d01fde9465199ef41754b68499441b94',
    EXTERNAL: '8588320e2785ae4629e9f0f142047c2cdb50e6661197c52b2baed9728e888b42',
}
TESTS = [ORIGINAL, EXTERNAL, 'review_acceptance/test_recovery_cli.py']


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sys.flags.optimize:
        raise RuntimeError('Optimized Python is not an accepted validation environment')
    for name, expected in PROTECTED.items():
        if digest(ROOT / name) != expected:
            raise RuntimeError('Protected test changed: ' + name)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    watched = TESTS + [SUBJECT, 'review_acceptance/run_recovery_gate.py']
    before = {name: digest(ROOT / name) for name in watched}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    command = [sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *TESTS]
    started = time.monotonic()
    with (output / 'pytest.stdout.log').open('w') as stdout, (output / 'pytest.stderr.log').open('w') as stderr:
        try:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout,
                                    stderr=stderr, timeout=120)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
    after = {name: digest(ROOT / name) for name in watched}
    unchanged = before == after
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
                          text=True, timeout=15, check=True).stdout.strip()
    status = 'passed' if code == 0 and unchanged else 'failed'
    report = {
        'status': status, 'git_head': head, 'source_sha256': before[SUBJECT],
        'command': command, 'cwd': str(ROOT), 'exit_code': code,
        'elapsed_seconds': time.monotonic() - started, 'python': sys.version,
        'platform': platform.platform(), 'sha256_before': before, 'sha256_after': after,
        'protected_tests_unchanged': unchanged, 'scientific_acceptance': False,
        'scope': 'Recovery API and CLI controls only; no model training or holdout access',
    }
    for name in watched:
        target = output / 'subject_source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print((output / 'pytest.stdout.log').read_text(), end='')
    print((output / 'pytest.stderr.log').read_text(), end='', file=sys.stderr)
    print(json.dumps({'status': status, 'report': str(output / 'report.json')}))
    return 0 if status == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
