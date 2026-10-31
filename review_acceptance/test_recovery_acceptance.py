"""Six contract controls for unchanged recovery, using tiny synthetic ZIPs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

SOURCE = (Path(__file__).resolve().parents[1] /
          "synthetic-data-lab/research_control/units/H003R/review/recover_artifact.py")
spec = importlib.util.spec_from_file_location("acceptance_recover_artifact", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
recover = module.recover


def sha(data):
    return hashlib.sha256(data).hexdigest()


def archive(root, members, manifest):
    path = root / "input.zip"
    with zipfile.ZipFile(path, "w") as z:
        for name, data in members:
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            z.writestr(info, data)
        info = zipfile.ZipInfo("ARTIFACT_SHA256.json", date_time=(2020, 1, 1, 0, 0, 0))
        z.writestr(info, manifest)
    return path, sha(path.read_bytes())


def test_valid_archive_control(tmp_path):
    payload = b"checked evidence"
    manifest = json.dumps({"nested/payload.txt": sha(payload)})
    path, pin = archive(tmp_path, [("nested/payload.txt", payload)], manifest)
    original = path.read_bytes()
    destination = tmp_path / "output"
    receipt = recover(path, pin, destination)
    assert receipt["status"] == "verified"
    assert receipt["archive_sha256"] == pin
    assert receipt["internal_files_checked"] == 1
    assert (destination / "nested/payload.txt").read_bytes() == payload
    assert (destination / "ARTIFACT_SHA256.json").read_text() == manifest
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("recovery_*"))


def test_noncanonical_member_path_rejected(tmp_path):
    payload = b"noncanonical"
    path, pin = archive(tmp_path, [("./payload.txt", payload)],
                        json.dumps({"payload.txt": sha(payload)}))
    with pytest.raises(ValueError):
        recover(path, pin, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_distinct_member_alias_overwrite_rejected(tmp_path):
    first, last = b"original evidence", b"replacement evidence"
    path, pin = archive(tmp_path, [("payload.txt", first), ("./payload.txt", last)],
                        json.dumps({"payload.txt": sha(last)}))
    with pytest.raises(ValueError):
        recover(path, pin, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_duplicate_json_manifest_keys_rejected(tmp_path):
    payload = b"checked evidence"
    manifest = ('{"payload.txt": "' + "0" * 64 +
                '", "payload.txt": "' + sha(payload) + '"}')
    path, pin = archive(tmp_path, [("payload.txt", payload)], manifest)
    with pytest.raises(ValueError):
        recover(path, pin, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_existing_destination_preserved(tmp_path):
    payload = b"new evidence"
    path, pin = archive(tmp_path, [("payload.txt", payload)],
                        json.dumps({"payload.txt": sha(payload)}))
    destination = tmp_path / "output"
    destination.mkdir()
    sentinel = destination / "retained.txt"
    sentinel.write_bytes(b"existing evidence")
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        recover(path, pin, destination)
    assert list(destination.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"existing evidence"
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("recovery_*"))


def test_cleanup_after_rejected_input(tmp_path):
    payload = b"extracted before digest rejection"
    path, pin = archive(tmp_path, [("nested/payload.txt", payload)],
                        json.dumps({"nested/payload.txt": "0" * 64}))
    sentinel = tmp_path / "retained.txt"
    sentinel.write_bytes(b"unrelated retained evidence")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ValueError):
        recover(path, pin, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob("recovery_*"))
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
