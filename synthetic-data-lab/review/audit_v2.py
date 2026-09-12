"""Read only artifact audit with no imports from the implementation under review.

Metrics, checkpoint inference, selection and provenance are checked separately.
This is a second implementation by the same assistant, not an external reviewer.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np
import torch
from torch.nn import functional as F


class AuditFailure(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditFailure(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def array_digest(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(json.dumps(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def safe(root, rel):
    root = Path(root).resolve()
    require(isinstance(rel, str) and bool(rel) and not Path(rel).is_absolute(), "Invalid relative path")
    p = root / rel
    require(p.resolve().is_relative_to(root), "Path leaves artifact root")
    for part in [p, *p.parents]:
        if part == root:
            break
        require(not part.is_symlink(), "Symbolic manifest path")
    require(p.is_file(), f"Missing artifact: {rel}")
    return p


def verify_manifest(root, required=()):
    root = Path(root)
    manifest = read(root / "FROZEN.json")
    stored = manifest.pop("manifest_sha256", None)
    require(canonical(manifest) == stored, "Manifest digest mismatch")
    files = manifest.get("files")
    require(isinstance(files, dict) and len(files) > 0, "Empty or invalid manifest")
    require(set(required).issubset(files), "Required artifacts absent from manifest")
    for rel, expected in files.items():
        require(digest(safe(root, rel)) == expected, f"Artifact digest mismatch: {rel}")
    return {"manifest_sha256": stored, "file_count": len(files), "files": files}


def check_probabilities(y, p):
    y, p = np.asarray(y), np.asarray(p)
    require(y.ndim == 1 and len(y) > 0 and np.issubdtype(y.dtype, np.integer), "Labels must be integers")
    require(p.ndim == 2 and p.shape[0] == len(y) and p.shape[1] >= 2, "Probability shape mismatch")
    require(not np.iscomplexobj(p) and np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(),
            "Invalid probability entries")
    require((y >= 0).all() and (y < p.shape[1]).all(), "Labels outside declared classes")
    require(np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-5), "Probability rows do not sum to one")
    return y, p.astype(float)


def independent_metrics(y, p):
    y, p = check_probabilities(y, p)
    p = np.clip(p, 1e-12, 1)
    p = p / p.sum(axis=1, keepdims=True)
    pred = p.argmax(axis=1)
    k = p.shape[1]
    cm = np.zeros((k, k), dtype=np.int64)
    for actual, estimated in zip(y.tolist(), pred.tolist()):
        cm[actual, estimated] += 1
    row, col, diag = cm.sum(axis=1), cm.sum(axis=0), np.diag(cm)
    recall = [float(diag[c] / row[c]) if row[c] else None for c in range(k)]
    f1 = [float(2 * diag[c] / (row[c] + col[c])) if row[c] + col[c] else 0.0 for c in range(k)]
    losses = -np.log(p[np.arange(len(y)), y])
    return {"accuracy": float(diag.sum() / len(y)),
            "balanced_accuracy": float(np.mean([v for v in recall if v is not None])),
            "macro_f1": float(np.mean(f1)), "log_loss": float(losses.mean()),
            "per_class_loss": [float(losses[y == c].mean()) if row[c] else None for c in range(k)],
            "per_class_recall": recall, "confusion_matrix": cm.tolist(), "n": len(y)}


def compare_metrics(expected, actual, where):
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "log_loss"):
        require(abs(expected[key] - actual[key]) <= 1e-10, f"Metric mismatch at {where}: {key}; expected={expected[key]!r}, observed={actual[key]!r}")
    for key in ("confusion_matrix", "n"):
        require(expected[key] == actual[key], f"Count mismatch at {where}: {key}")
    for key in ("per_class_loss", "per_class_recall"):
        require(len(expected[key]) == len(actual[key]), f"Class count mismatch at {where}")
        for a, b in zip(expected[key], actual[key]):
            require((a is None and b is None) or (a is not None and b is not None and abs(a-b) <= 1e-10),
                    f"Class metric mismatch at {where}: {key}")


# Saved probabilities have a strict arithmetic contract. Recomputed inference
# is a different check: single precision kernels can vary across CPU models.
# These bounds never authorize changing a predicted class or a saved metric.
INFERENCE_ATOL = 1e-6
INFERENCE_RTOL = 1e-5


def compare_inference(recorded, reconstructed, where):
    recorded, reconstructed = np.asarray(recorded), np.asarray(reconstructed)
    require(recorded.shape == reconstructed.shape, f"Inference shape mismatch at {where}")
    require(np.isfinite(reconstructed).all(), f"Nonfinite inference at {where}")
    difference = float(np.max(np.abs(recorded - reconstructed)))
    require(np.allclose(recorded, reconstructed, rtol=INFERENCE_RTOL, atol=INFERENCE_ATOL),
            f"Checkpoint probabilities materially differ at {where}: max absolute difference {difference}")
    require(np.array_equal(recorded.argmax(1), reconstructed.argmax(1)),
            f"Checkpoint decisions changed at {where}")
    return {"path": where, "exact": bool(np.array_equal(recorded, reconstructed)),
            "maximum_absolute_probability_difference": difference}


def compare_reconstructed_metrics(expected, actual, where):
    """Probes/logistic dev do not store probabilities. Disclose this weaker check.

    Count and decision metrics remain exact. Losses permit explicitly bounded
    numerical drift from reconstructed inference, not from stored predictions.
    """
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "per_class_recall", "confusion_matrix", "n"):
        require(np.array_equal(expected[key], actual[key]), f"Reconstructed decisions changed at {where}: {key}")
    differences = []
    for a, b in [(expected["log_loss"], actual["log_loss"]), *zip(expected["per_class_loss"], actual["per_class_loss"])]:
        require((a is None) == (b is None), f"Reconstructed class coverage mismatch at {where}")
        if a is not None:
            require(np.isfinite(a) and np.isfinite(b), f"Nonfinite reconstructed loss at {where}")
            difference = abs(a-b)
            require(difference <= INFERENCE_ATOL + INFERENCE_RTOL * abs(a),
                    f"Reconstructed loss drift exceeds bound at {where}: expected={a!r}, observed={b!r}")
            differences.append(float(difference))
    return {"path": where, "maximum_absolute_loss_difference": max(differences, default=0.0),
            "saved_probability_array_available": False}


def state_digest(state):
    h = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def checkpoint_predict(path, x, kind):
    meta = read(path / "model.json")
    z = (x - np.asarray(meta["mean"])) / np.asarray(meta["scale"])
    require(np.isfinite(z).all(), "Invalid saved normalization")
    if kind == "logistic":
        scores = z @ np.asarray(meta["coef"]).T + np.asarray(meta["intercept"])
        if scores.shape[1] == 1:
            scores = np.column_stack((np.zeros(len(x)), scores[:, 0]))
        scores -= scores.max(axis=1, keepdims=True)
        p = np.exp(scores)
        return p / p.sum(axis=1, keepdims=True)
    require(kind == "mlp", "Unknown student type")
    state = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
    require(state_digest(state) == meta["audit"]["final_sha256"], "Student weight digest mismatch")
    require(meta["audit"]["initial_sha256"] != meta["audit"]["final_sha256"], "Unchanged student parameters")
    with torch.no_grad():
        value = torch.tensor(z, dtype=torch.float32)
        for i in (0, 2, 4):
            value = F.linear(value, state[f"layers.{i}.weight"], state[f"layers.{i}.bias"])
            if i != 4:
                value = F.relu(value)
        return F.softmax(value, dim=1).numpy().astype(float)


def audit(run):
    root = Path(run).resolve()
    torch.set_num_threads(1)
    frozen = verify_manifest(root, ("INDEX.json", "protocol.json", "environment.json"))
    final = verify_manifest(root / "final_evaluation", ("scores.json",))
    config, index = read(root / "protocol.json"), read(root / "INDEX.json")
    scores = read(root / "final_evaluation/scores.json")
    require(scores["frozen_manifest_sha256"] == frozen["manifest_sha256"], "Evaluation refers to another training run")
    require(scores.get("test_used_for_selection") is False, "Evaluation discloses test based selection")
    methods = config["methods"] + config["portfolios"]
    conditions = [("real_only", "real_only")] + [(m, r) for m in methods for r in config["regimes"]]
    expected = {(d, s, g, m, r, learner, seed)
                for d in config["datasets"] for s in config["split_seeds"] for g in config["generation_seeds"]
                for m, r in conditions for learner, seed in [("mlp", x) for x in config["student_seeds"]] + [("logistic", None)]}
    key = lambda v: tuple(v.get(k) for k in ("dataset", "split_seed", "generation_seed", "method", "regime", "student", "student_seed"))
    observed = [key(v) for v in scores["rows"]]
    require(len(observed) == len(set(observed)) and set(observed) == expected, "Missing, duplicate or unexpected evaluation conditions")
    splits = {}
    split_checks = 0
    from sklearn.datasets import load_wine, load_breast_cancer
    for dataset in config["datasets"]:
        source = {"wine": load_wine, "breast_cancer": load_breast_cancer}[dataset]()
        for seed in config["split_seeds"]:
            split = root / "splits" / dataset / str(seed)
            metadata = read(split / "metadata.json")
            parts = {}
            ids = []
            for name in ("train", "dev", "test", "unused"):
                with np.load(split / f"{name}.npz", allow_pickle=False) as data:
                    parts[name] = {k: data[k].copy() for k in ("x", "y", "ids")}
                part = parts[name]
                require(len(part["ids"]) == len(set(part["ids"].tolist())), "Duplicate partition IDs")
                require(len(part["ids"]) == metadata["counts"][name], "Partition count mismatch")
                require(array_digest(part["ids"]) == metadata["split_hashes"][name], "Partition ID digest mismatch")
                require(np.array_equal(part["x"], source.data[part["ids"]]), "Partition features do not match source IDs")
                require(np.array_equal(part["y"], source.target[part["ids"]]), "Partition labels do not match source IDs")
                ids.extend(part["ids"].tolist())
            require(len(ids) == len(set(ids)) == len(source.target) and set(ids) == set(range(len(source.target))),
                    "Partitions overlap or fail to cover the source")
            splits[(dataset, seed)] = parts
            split_checks += 1
    counters = Counter()
    exact = 0
    maximum_probability_error = 0.0
    development_inference_checks = []
    reconstructed_only_checks = []
    test_inference_checks = []
    grouped_development = defaultdict(list)
    aggregate = defaultdict(list)
    for row in scores["rows"]:
        parts = splits[(row["dataset"], row["split_seed"])]
        path = root / row["path"]
        require(row["path"] + "/model.json" in frozen["files"], "Student metadata not sealed")
        with np.load(safe(root, row["prediction_path"]), allow_pickle=False) as data:
            recorded = {k: data[k].copy() for k in ("ids", "labels", "probabilities")}
        require(np.array_equal(recorded["ids"], parts["test"]["ids"]), "Prediction IDs differ from frozen test")
        require(np.array_equal(recorded["labels"], parts["test"]["y"]), "Prediction answers differ from frozen test")
        measured = independent_metrics(recorded["labels"], recorded["probabilities"])
        compare_metrics(row["test"], measured, row["path"] + "/test")
        p = checkpoint_predict(path, parts["test"]["x"], row["student"])
        test_check = compare_inference(recorded["probabilities"], p, row["path"] + "/test")
        test_inference_checks.append(test_check)
        maximum_probability_error = max(test_check["maximum_absolute_probability_difference"], maximum_probability_error)
        exact += int(test_check["exact"])
        dev_p = checkpoint_predict(path, parts["dev"]["x"], row["student"])
        if row["student"] == "mlp":
            rel = row["path"] + "/development_predictions.npz"
            require(rel in frozen["files"], "Development probabilities are not sealed")
            with np.load(safe(root, rel), allow_pickle=False) as stored:
                saved_dev = stored["probabilities"].copy()
            dev_metrics = independent_metrics(parts["dev"]["y"], saved_dev)
            compare_metrics(row["development"], dev_metrics, row["path"] + "/saved_dev")
            development_inference_checks.append(compare_inference(saved_dev, dev_p, row["path"] + "/dev"))
        else:
            dev_metrics = independent_metrics(parts["dev"]["y"], dev_p)
            reconstructed_only_checks.append(compare_reconstructed_metrics(row["development"], dev_metrics, row["path"] + "/dev"))
        if row["student"] == "mlp":
            model = read(path / "model.json"); a = model["audit"]
            require(a["completed_steps"] == config["student_steps"] == len(a["losses"]), "Incomplete neural student training")
            require(np.isfinite(a["losses"]).all(), "Nonfinite student losses")
            table = parts["train"] if row["method"] == "real_only" else None
            if table is None:
                table_path = path.parents[3] / "datasets" / row["method"] / "training.npz"
                with np.load(table_path, allow_pickle=False) as data:
                    table = {"x": data["x"], "y": data["y"]}
            require(array_digest(table["x"]) == a["training_x_sha256"] and array_digest(table["y"]) == a["training_y_sha256"],
                    "Student training data lineage mismatch")
            grouped_development[(row["dataset"], row["split_seed"], row["generation_seed"], row["method"], row["regime"])].append(dev_metrics)
        counters[row["student"]] += 1
        aggregate[(row["dataset"], row["student"], row["method"], row["regime"])].append(measured["balanced_accuracy"])
    selection_count = 0
    generator_count = 0
    probe_count = 0
    for rel in index["replicates"]:
        rep_path = safe(root, rel)
        rep = read(rep_path)
        metadata = read(root / rep["split_path"] / "metadata.json")
        parts = splits[(metadata["dataset"], metadata["split_seed"])]
        rx, ry = parts["train"]["x"], parts["train"]["y"]
        ranked = []
        for method, regime in conditions:
            values = grouped_development[(metadata["dataset"], metadata["split_seed"], rep["generation_seed"], method, regime)]
            require(len(values) == len(config["student_seeds"]), "Incomplete development candidates")
            ranked.append({"method": method, "regime": regime,
                           "balanced_accuracy": float(np.mean([v["balanced_accuracy"] for v in values])),
                           "log_loss": float(np.mean([v["log_loss"] for v in values]))})
        ranked.sort(key=lambda v: (-v["balanced_accuracy"], v["log_loss"], v["method"], v["regime"]))
        require(ranked[0]["method"] == rep["selection"]["method"] and ranked[0]["regime"] == rep["selection"]["regime"],
                "Development winner cannot be reproduced")
        selection_count += 1
        for method in config["methods"]:
            folder = rep_path.parent / "generators" / method
            a = read(folder / "audit.json")
            require(a["training_x_sha256"] == array_digest(rx) and a["training_y_sha256"] == array_digest(ry), "Generator input lineage mismatch")
            with np.load(folder / "pool.npz", allow_pickle=False) as pool:
                require(array_digest(pool["x"]) == a["pool_x_sha256"] and array_digest(pool["y"]) == a["pool_y_sha256"], "Generator pool lineage mismatch")
            if (folder / "weights.pt").is_file():
                state = torch.load(folder / "weights.pt", map_location="cpu", weights_only=True)
                require(state_digest(state) == a["final_sha256"], "Generator weights do not match audit")
                require(a["initial_sha256"] != a["final_sha256"], "Unchanged generator weights")
                require(a["completed_steps"] == config["generator_steps"] == len(a["losses"]) and np.isfinite(a["losses"]).all(), "Incomplete generator training")
                generator_count += 1
        for p in (rep_path.parent / "probes").glob("*/*/model.json"):
            model = read(p); a = model["audit"]
            require(a["completed_steps"] == config["probe_steps"] == len(a["losses"]), "Incomplete probe training")
            predicted = checkpoint_predict(p.parent, parts["dev"]["x"], "mlp")
            reconstructed_only_checks.append(compare_reconstructed_metrics(
                read(p.parent / "development.json"), independent_metrics(parts["dev"]["y"], predicted), str(p.relative_to(root))))
            probe_count += 1
    require(selection_count == len(config["datasets"]) * len(config["split_seeds"]) * len(config["generation_seeds"]), "Replicate count mismatch")
    return {"status": "passed", "reviewer_independent_from_original_author": False,
            "production_modules_imported": any(k.startswith("synthlab_next") for k in sys.modules),
            "scope": "Separately implemented artifact, arithmetic, inference and development selection checks; not a scientific replication claim",
            "source_run": str(root), "training_manifest_sha256": frozen["manifest_sha256"],
            "final_manifest_sha256": final["manifest_sha256"],
            "sealed_training_files_verified": frozen["file_count"], "sealed_evaluation_files_verified": final["file_count"],
            "source_verified_disjoint_partitions": split_checks, "evaluation_conditions_checked": len(observed),
            "neural_generator_weight_and_training_ledgers_checked": generator_count,
            "neural_students_checked": counters["mlp"], "logistic_students_checked": counters["logistic"],
            "development_probe_checkpoints_checked": probe_count, "development_winners_recomputed": selection_count,
            "test_prediction_arrays_exact": exact, "maximum_checkpoint_probability_error": maximum_probability_error,
            "saved_development_probability_arrays_strictly_scored": len(development_inference_checks),
            "development_inference_arrays_exact": sum(int(v["exact"]) for v in development_inference_checks),
            "inference_tolerances": {"absolute": INFERENCE_ATOL, "relative": INFERENCE_RTOL, "class_decisions_must_match": True},
            "saved_metric_tolerance": 1e-10,
            "nonexact_test_inference": [v for v in test_inference_checks if not v["exact"]],
            "nonexact_development_inference": [v for v in development_inference_checks if not v["exact"]],
            "reconstructed_only_metrics_count": len(reconstructed_only_checks),
            "reconstructed_loss_drift": [v for v in reconstructed_only_checks if v["maximum_absolute_loss_difference"] > 1e-10],
            "metric_families_checked": ["accuracy", "balanced_accuracy", "macro_f1", "log_loss", "per_class_loss", "per_class_recall", "confusion_matrix", "n"],
            "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
            "method_means": [{"dataset": k[0], "student": k[1], "method": k[2], "regime": k[3], "conditions": len(v), "balanced_accuracy": float(np.mean(v))} for k,v in sorted(aggregate.items())],
            "limits": ["Internal artifact consistency does not prove that no test informed human choices", "Repeated splits overlap and do not constitute independent population samples", "No privacy, chemical validity, clinical or frontier superiority claim", "No pretrained language model methods are evaluated here", "Checkpoint reload is not complete training reproduction"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Audit output must be new; overwrite refused")
    require(not args.output.resolve().is_relative_to(args.run.resolve()), "Audit output must be outside immutable experiment")
    result = audit(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("method_means", "limits")}, indent=2))


if __name__ == "__main__":
    main()
