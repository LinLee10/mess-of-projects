"""Bind the files actually consumed by H004B to independent H004A evidence."""
from pathlib import Path
import hashlib
import json
import re

METHODS = ('unit_beta', 'heuristic', 'empirical_median', 'empirical_quantile')
SEEDS = (43017, 43029, 43041)

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def strict_json(path):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError('Duplicate verification key')
            out[key] = value
        return out
    def invalid(value):
        raise ValueError('Nonfinite verification value')
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid)

def admit(audit, verification):
    """Require exact resolved identities and current bytes for all twelve tables.

    This admission control does not replace the numerical verifier or establish
    that a caller-supplied report is independently trustworthy.
    """
    audit = Path(audit).resolve(strict=True)
    verification = Path(verification).resolve(strict=True)
    report = strict_json(verification)
    if report.get('status') != 'verified' or report.get('fixture') is not False:
        raise ValueError('Full independent verification required')
    required = {'training_rows':348563, 'contexts':232,
                'audit_files':12, 'audit_rows':4182756}
    counts = report.get('counts', {})
    if any(type(counts.get(k)) is not int or counts[k] != v for k,v in required.items()):
        raise ValueError('Incomplete verified population')
    hashes = report.get('artifact_sha256')
    if not isinstance(hashes, dict):
        raise ValueError('Verified identity mapping required')
    indexed = {}
    for name, value in hashes.items():
        if not isinstance(name,str) or not Path(name).is_absolute():
            raise ValueError('Verified paths must be absolute')
        resolved = str(Path(name).resolve())
        if resolved in indexed:
            raise ValueError('Ambiguous verified path identity')
        if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):
            raise ValueError('Malformed verified digest')
        indexed[resolved] = value
    checked = []
    for method in METHODS:
        for seed in SEEDS:
            path = audit / f'{method}_{seed}.npz'
            if path.is_symlink() or not path.is_file():
                raise ValueError('Expected ordinary audit artifact')
            resolved = path.resolve(strict=True)
            if resolved.parent != audit or str(resolved) not in indexed:
                raise ValueError('Consumed audit path is not in verified evidence')
            actual = digest(resolved)
            if actual != indexed[str(resolved)]:
                raise ValueError('Consumed audit bytes changed')
            checked.append({'path':str(resolved),'sha256':actual})
    return {'status':'admitted','consumed_files':checked,
            'verification_path':str(verification),'verification_sha256':digest(verification)}
