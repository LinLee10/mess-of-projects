"""Verify an Actions ZIP and extract it without accepting path traversal."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import re
import shutil
import stat
import tempfile
import zipfile

def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def safe_name(name: str) -> bool:
    p = PurePosixPath(name)
    return bool(name) and not p.is_absolute() and '..' not in p.parts and '\\' not in name and ':' not in name

def recover(archive: Path, expected: str, destination: Path) -> dict:
    if not re.fullmatch('[0-9a-f]{64}', expected):
        raise ValueError('Expected SHA256 must be a lowercase hexadecimal digest')
    actual = digest(archive)
    if actual != expected:
        raise ValueError('Archive SHA256 mismatch')
    if destination.exists():
        raise FileExistsError('Refusing to replace existing evidence')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='recovery_', dir=destination.parent))
    try:
        with zipfile.ZipFile(archive) as z:
            members = z.infolist()
            if len(set(i.filename for i in members)) != len(members):
                raise ValueError('Duplicate archive paths')
            for item in members:
                if not safe_name(item.filename) or stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError('Unsafe archive member')
            z.extractall(staging)
        manifest = json.loads((staging/'ARTIFACT_SHA256.json').read_text())
        if not isinstance(manifest, dict):
            raise ValueError('Expected manifest mapping')
        files = {str(f.relative_to(staging)) for f in staging.rglob('*') if f.is_file()}
        if files != set(manifest) | {'ARTIFACT_SHA256.json'}:
            raise ValueError('Manifest does not exactly cover archive files')
        for name, expected_file in manifest.items():
            if not safe_name(name) or digest(staging/name) != expected_file:
                raise ValueError('Invalid internal digest: '+name)
        receipt = {'archive': archive.name, 'archive_sha256': actual,
                   'archive_bytes': archive.stat().st_size, 'internal_files_checked': len(manifest),
                   'status': 'verified', 'destination': str(destination)}
        staging.rename(destination)
        return receipt
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    parser.add_argument('sha256')
    parser.add_argument('destination', type=Path)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    result = recover(args.archive, args.sha256, args.destination)
    args.receipt.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
