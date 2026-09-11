#!/usr/bin/env python3
"""Explicit publication helper. No network or writes unless --publish is supplied.

Preserves upstream history. Never pushes to Arhaan's repository, changes main,
uses force push, or puts this work in an unrelated existing repository.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

OWNER = 'LinLee10'
UPSTREAM = 'Arhaan2/synthetic-data-lab'
PIN = '039fe33a2a9aad00ca9fed3f7f0733be8a912732'
REPO = 'synthetic-data-lab'
BRANCH = 'research/executed-tabular-pilot-v1'


def run(args, cwd=None, check=True):
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=300)
    if check and result.returncode:
        # CLI output can contain local credentials in unusual configurations.
        # Do not copy arbitrary stderr into a public artifact.
        raise RuntimeError(f'{args[0]} {args[1]} failed with status {result.returncode}; inspect it locally')
    return result


def validate_destination(meta):
    if (meta.get('full_name') != f'{OWNER}/{REPO}' or not meta.get('fork')
            or meta.get('parent', {}).get('full_name') != UPSTREAM):
        raise ValueError('Destination is not the expected personal fork; refusing unrelated repository')
    if not meta.get('permissions', {}).get('push', False):
        raise ValueError('Authenticated account lacks push permission on the fork')


def copy_extension(source, target):
    source, target = Path(source).resolve(), Path(target).resolve()
    if target.exists():
        raise ValueError('Extension destination exists; no files will be overwritten')
    if source == target or source in target.parents:
        raise ValueError('Destination cannot be nested inside the source package')
    ignore = shutil.ignore_patterns('.git', '__pycache__', '*.pyc', '*.zip', '.DS_Store', '*.egg-info')
    # Reject symlinks and likely credential files before copying any content.
    for path in source.rglob('*'):
        if '.git' in path.parts or '__pycache__' in path.parts:
            continue
        if path.is_symlink():
            raise ValueError('Symlinks are not accepted for publication')
        if path.is_file() and (path.name == '.env' or path.suffix in {'.pem', '.key'}):
            raise ValueError('Potential credential file is present; publication refused')
    shutil.copytree(source, target, ignore=ignore)
    if any(p.stat().st_size >= 95_000_000 for p in target.rglob('*') if p.is_file()):
        raise ValueError('A file is too large for normal Git publication; separate artifact storage is required')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--publish', action='store_true', help='Authorize fork, new branch, commit and push')
    p.add_argument('--workdir', required=True, type=Path, help='A new checkout directory outside this package')
    args = p.parse_args(argv)
    source = Path(__file__).resolve().parents[1]
    target = args.workdir.expanduser().resolve()
    plan = {'upstream': UPSTREAM, 'pinned_commit': PIN, 'destination': f'{OWNER}/{REPO}',
            'branch': BRANCH, 'overlay': 'extensions/executed_tabular', 'publish_requested': args.publish}
    print(json.dumps(plan, indent=2))
    if not args.publish:
        return
    if target.exists() or source == target or source in target.parents:
        raise ValueError('Choose a new checkout directory outside this package')
    for command in ('git', 'gh'):
        if shutil.which(command) is None:
            raise RuntimeError(f'{command} is required; no remote changes made')
    user = run(['gh', 'api', 'user', '--jq', '.login']).stdout.strip()
    if user != OWNER:
        raise ValueError(f'GitHub CLI must be authenticated as {OWNER}; no remote changes made')
    # Ensure configured commit identity exists before any remote mutation.
    if not run(['git', 'config', 'user.name'], check=False).stdout.strip() or not run(['git', 'config', 'user.email'], check=False).stdout.strip():
        raise ValueError('Configure your Git commit name and email locally before publishing')
    found = run(['gh', 'api', f'repos/{OWNER}/{REPO}'], check=False)
    if found.returncode:
        # Do not interpret an arbitrary network/authorization failure as absence.
        if 'HTTP 404' not in found.stderr:
            raise RuntimeError('Destination lookup failed for a reason other than absence')
        run(['gh', 'repo', 'fork', UPSTREAM, '--clone=false', '--fork-name', REPO])
        found = run(['gh', 'api', f'repos/{OWNER}/{REPO}'])
    validate_destination(json.loads(found.stdout))
    remote = f'https://github.com/{OWNER}/{REPO}.git'
    existing = run(['git', 'ls-remote', '--heads', remote, BRANCH])
    if existing.stdout.strip():
        raise ValueError('Research branch already exists; no overwrite or force push will be attempted')
    run(['git', 'clone', '--no-checkout', remote, str(target)])
    run(['git', 'cat-file', '-e', PIN + '^{commit}'], cwd=target)
    run(['git', 'switch', '-c', BRANCH, PIN], cwd=target)
    copy_extension(source, target / 'extensions' / 'executed_tabular')
    notice = target / 'EXECUTED_TABULAR_EXTENSION.md'
    if notice.exists():
        raise ValueError('Integration notice already exists')
    notice.write_text('# Executed tabular research extension\n\n'
        'The original text methods and their historical not_run status are unchanged. '
        'Actual tabular generator and student experiments, provenance, limitations and '
        'results are in [extensions/executed_tabular](extensions/executed_tabular/README.md). '
        'This is a bounded pilot, not a reproduction or execution of the six text papers.\n')
    run(['git', 'add', '-f', '--', 'extensions/executed_tabular', notice.name], cwd=target)
    run(['git', 'commit', '-m', 'Add executed tabular research pilot with frozen evidence and model exports'], cwd=target)
    head = run(['git', 'rev-parse', 'HEAD'], cwd=target).stdout.strip()
    parent = run(['git', 'rev-parse', 'HEAD^'], cwd=target).stdout.strip()
    if parent != PIN:
        raise RuntimeError('Unexpected commit ancestry; push refused')
    run(['git', 'push', 'origin', f'HEAD:refs/heads/{BRANCH}'], cwd=target)
    observed = run(['git', 'ls-remote', '--heads', 'origin', BRANCH], cwd=target).stdout.split()
    if not observed or observed[0] != head:
        raise RuntimeError('Remote commit verification failed; inspect the remote before retrying')
    print(json.dumps({'published': True, 'commit': head,
        'url': f'https://github.com/{OWNER}/{REPO}/tree/{BRANCH}'}, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f'Publication stopped: {exc}', file=sys.stderr)
        sys.exit(1)
