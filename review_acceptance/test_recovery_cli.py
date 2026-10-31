"""Coordinator regression tests supplement, but never replace, the external tests."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'synthetic-data-lab/research_control/units/H003R/review/recover_artifact.py'
spec = importlib.util.spec_from_file_location('cli_recovery_subject', SOURCE)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


@pytest.mark.parametrize('name', ['', '.', './x', 'a//x', 'a/./x', 'a/../x', '/x', 'a\\x', 'C:x', 'a\x00x'])
def test_noncanonical_name_rejected(name):
    assert not subject.safe_name(name)


def make_archive(root, *, corrupt=False, directories=False, duplicate_key=False):
    payload = b'roundtrip evidence'
    digest = hashlib.sha256(payload).hexdigest()
    manifest = {'nested/payload.txt': '0' * 64 if corrupt else digest}
    text = json.dumps(manifest)
    if duplicate_key:
        text = '{"nested/payload.txt":"' + digest + '","nested/payload.txt":"' + digest + '"}'
    archive = root / 'input.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        if directories:
            z.writestr('nested/', b'')
        z.writestr('nested/payload.txt', payload)
        z.writestr('ARTIFACT_SHA256.json', text)
    return archive, hashlib.sha256(archive.read_bytes()).hexdigest(), payload


def test_explicit_directory_compatibility(tmp_path):
    archive, pin, payload = make_archive(tmp_path, directories=True)
    result = subject.recover(archive, pin, tmp_path / 'output')
    assert result['internal_files_checked'] == 1
    assert (tmp_path / 'output/nested/payload.txt').read_bytes() == payload


def test_duplicate_identical_manifest_key_rejected(tmp_path):
    archive, pin, _ = make_archive(tmp_path, duplicate_key=True)
    with pytest.raises(ValueError):
        subject.recover(archive, pin, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


@pytest.mark.parametrize('mode', ['valid', 'bad_outer_hash', 'bad_internal_hash'])
def test_cli_roundtrip_and_corruption_rejection(tmp_path, mode):
    archive, pin, payload = make_archive(tmp_path, corrupt=mode == 'bad_internal_hash')
    before = archive.read_bytes()
    output = tmp_path / 'output'
    receipt = tmp_path / 'receipt.json'
    command = [sys.executable, str(SOURCE), str(archive),
               '0' * 64 if mode == 'bad_outer_hash' else pin,
               str(output), '--receipt', str(receipt)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    assert archive.read_bytes() == before
    if mode == 'valid':
        assert result.returncode == 0, result.stderr
        record = json.loads(receipt.read_text())
        assert record['archive_sha256'] == pin
        assert record['status'] == 'verified'
        assert (output / 'nested/payload.txt').read_bytes() == payload
        repeated = subprocess.run(command, capture_output=True, text=True, timeout=15)
        assert repeated.returncode != 0
        assert json.loads(receipt.read_text()) == record
    else:
        assert result.returncode != 0
        assert not output.exists()
        assert not receipt.exists()
    assert not list(tmp_path.glob('recovery_*'))
