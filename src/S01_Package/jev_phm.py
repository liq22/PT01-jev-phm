"""Jev-inspired PHM probe. No Jev weights, RLCD reproduction, or text generation."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

PHMFACTORY_REVISION = "cdc0669167ca25ac848e34c880875ca39c6aed89"
FEATURES = ("rms", "std", "mean_abs", "peak_to_peak", "crest", "kurtosis")
ROLES = ("train", "tune", "cal", "test")


def unit_weights(groups: np.ndarray) -> np.ndarray:
    """Uniform independent unit, then uniform window within that unit."""
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    if not len(counts):
        raise ValueError("No independent units supplied")
    return 1.0 / (len(counts) * counts[inverse])


def check_logits(logits: np.ndarray, y: np.ndarray) -> None:
    if logits.ndim != 2 or len(logits) == 0 or logits.shape[1] < 2:
        raise ValueError("Expected nonempty [windows, classes] logits")
    if not np.isfinite(logits).all():
        raise ValueError("Non-finite logits")
    if y.shape != (len(logits),) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("Labels must be a one-dimensional integer array")
    if np.any(y < 0) or np.any(y >= logits.shape[1]):
        raise ValueError("Label outside declared class space")


def nll(logits: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    check_logits(logits, y)
    return float(weights @ (logsumexp(logits, axis=1) - logits[np.arange(len(y)), y]))


def temperature(logits: np.ndarray, y: np.ndarray, groups=None) -> float:
    """Fit only on calibration data; groups=None is ordinary pooled scaling."""
    w = np.full(len(y), 1.0 / len(y)) if groups is None else unit_weights(groups)
    result = minimize_scalar(lambda t: nll(logits / np.exp(t), y, w),
                             bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    if not result.success:
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    # Include T=1 explicitly rather than accepting numerical deterioration in-sample.
    return float(np.exp(result.x)) if result.fun < nll(logits, y, w) else 1.0


def signal_features(signal) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64).squeeze()
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("First probe requires one finite univariate vibration window")
    rms, sd = np.sqrt(np.mean(x*x)), np.std(x)
    crest = np.max(np.abs(x))/rms if rms > 0 else 0.0
    kurtosis = np.mean(((x-x.mean())/sd)**4) if sd > 0 else 0.0
    return np.array([rms, sd, np.abs(x).mean(), np.ptp(x), crest, kurtosis])


def check_data(data: dict) -> None:
    n = len(data["y"])
    for key in ("features", "y", "unit", "role", "record"):
        if len(data[key]) != n:
            raise ValueError(f"Length mismatch: {key}")
    if data["features"].shape != (n, len(FEATURES)) or not np.isfinite(data["features"]).all():
        raise ValueError("Invalid six-feature state")
    if len(set(data["record"])) != n:
        raise ValueError("Duplicate source windows")
    if set(data["role"]) != set(ROLES):
        raise ValueError("Need nonempty train/tune/cal/test roles")
    for unit in np.unique(data["unit"]):
        if len(set(data["role"][data["unit"] == unit])) != 1:
            raise ValueError(f"Independent unit leaks across roles: {unit}")
    classes = set(range(len(data["labels"])))
    if len(classes) < 2 or len(set(data["labels"])) != len(classes):
        raise ValueError("Need at least two uniquely named classes")
    for role in ROLES:
        if set(data["y"][data["role"] == role]) != classes:
            raise ValueError(f"Closed-set class coverage missing in {role}; do not window-split to repair it")
    check_logits(np.zeros((n, len(classes))), data["y"])


def export_factory(factory, units_path, label_map, signal_key, label_key) -> dict:
    """Reuse upstream datasets and transforms. Never load/window raw files here."""
    with open(units_path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    mapping = {r["file_id"]: r for r in rows}
    if len(mapping) != len(rows):
        raise ValueError("Each file_id needs one explicit unit/role assignment")
    labels = list(label_map.values())
    if len(set(labels)) != len(labels):
        raise ValueError("Label names must be unique; do not merge raw labels implicitly")
    encode = {str(raw): i for i, raw in enumerate(label_map)}
    result = {k: [] for k in ("features", "y", "unit", "role", "record")}
    seen_files = set()
    for split, allowed in (("train", {"train"}), ("val", {"tune", "cal"}), ("test", {"test"})):
        dataset = getattr(factory, f"{split}_dataset")
        for i in range(len(dataset)):
            item = dataset[i]
            file_id = str(item["file_id"])
            row = mapping[file_id]
            if row["role"] not in allowed or not row["unit_id"].strip():
                raise ValueError(f"Invalid independent-unit role for upstream {split}/{file_id}")
            raw_label = np.asarray(item[label_key])
            if raw_label.size != 1:
                raise ValueError("Expected a scalar class label, not a one-hot or sequence target")
            y = encode[str(raw_label.item())]
            result["features"].append(signal_features(item[signal_key]))
            result["y"].append(y)
            result["unit"].append(row["unit_id"])
            result["role"].append(row["role"])
            result["record"].append(f"{file_id}:{i}")
            seen_files.add(file_id)
    if seen_files != set(mapping):
        raise ValueError("units.csv must cover exactly the files selected by PHMFactory")
    data = {k: np.asarray(v) for k, v in result.items()}
    data["labels"] = np.asarray(labels)
    check_data(data)
    return data


def embed(features, revision: str, device: str, batch_size: int = 8) -> np.ndarray:
    """Frozen Qwen3 last-nonpadding representation; labels/IDs never enter text."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Supply the resolved Hugging Face commit, not a moving model alias")
    name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=False)
    tokenizer.padding_side = "right"
    model = AutoModel.from_pretrained(name, revision=revision, trust_remote_code=False).to(device).eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            texts = []
            for row in features[start:start+batch_size]:
                state = "; ".join(f"{k}={v:.8g}" for k, v in zip(FEATURES, row))
                texts.append(tokenizer.apply_chat_template(
                    [{"role": "user", "content": "Represent this vibration window for fault diagnosis. " + state}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False))
            inputs = tokenizer(texts, padding=True, truncation=False, return_tensors="pt").to(device)
            hidden = model(**inputs).last_hidden_state
            last = inputs["attention_mask"].sum(1)-1
            output.append(hidden[torch.arange(len(texts), device=device), last].float().cpu().numpy())
    return np.concatenate(output)


def typed_decisions(probabilities, labels, reject_cost):
    if not 0 < reject_cost < 1:
        raise ValueError("The declared 0/1-loss scenario requires 0 < reject_cost < 1")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or not np.allclose(probabilities.sum(1), 1):
        raise ValueError("Invalid categorical probabilities")
    best = probabilities.argmax(1)
    accept = 1-probabilities.max(1) < reject_cost  # Defer on a cost tie.
    return [{"kind": "diagnosis" if a else "defer", "label": str(labels[k]) if a else None,
             "probabilities": dict(zip(map(str, labels), map(float, p)))}
            for p, k, a in zip(probabilities, best, accept)]


def metrics(logits, y, groups, reject_cost):
    p = softmax(logits, axis=1)
    w = unit_weights(groups)
    predicted, confidence = p.argmax(1), p.max(1)
    error = predicted != y
    accepted = 1-confidence < reject_cost
    coverage = float(w @ accepted)
    bins = np.minimum((confidence*10).astype(int), 9)
    ece = sum(abs(float(w[bins == b] @ (confidence[bins == b] - (~error[bins == b])))) for b in range(10))
    return {"nll": nll(logits, y, w), "brier": float(w @ ((p-np.eye(p.shape[1])[y])**2).sum(1)),
            "accuracy": float(w @ (~error)), "macro_f1": float(f1_score(y, predicted, average="macro", sample_weight=w)),
            "ece_10": ece, "coverage": coverage,
            "selective_error": float(w @ (error & accepted))/coverage if coverage else None,
            "scenario_cost": float(w @ np.where(accepted, error, reject_cost))}


def run(data, embeddings, config, out):
    check_data(data)
    if not np.array_equal(embeddings["record"], data["record"]):
        raise ValueError("Embeddings are not aligned to PHMFactory rows")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)  # Never overwrite a successful or failed run.
    masks = {r: data["role"] == r for r in ROLES}
    report, arrays = {}, {k: data[k] for k in ("y", "unit", "role", "record", "labels")}
    for family, x in (("numeric", data["features"]), ("qwen", embeddings["hidden"])):
        if x.ndim != 2 or len(x) != len(data["y"]) or not np.isfinite(x).all():
            raise ValueError(f"Invalid {family} features")
        scaler = StandardScaler().fit(x[masks["train"]])
        x = scaler.transform(x)
        candidates = []
        for c in config["C_grid"]:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                clf = LogisticRegression(C=c, solver="lbfgs", max_iter=2000, tol=1e-7).fit(x[masks["train"]], data["y"][masks["train"]])
            val = clf.predict_log_proba(x[masks["tune"]])
            candidates.append((nll(val, data["y"][masks["tune"]], unit_weights(data["unit"][masks["tune"]])), c, clf))
        _, chosen_c, clf = min(candidates, key=lambda r: r[0])
        logits = clf.predict_log_proba(x)
        cal, test = masks["cal"], masks["test"]
        temps = {"raw": 1., "pooled": temperature(logits[cal], data["y"][cal]),
                 "unit": temperature(logits[cal], data["y"][cal], data["unit"][cal])}
        report[family] = {"C": chosen_c, "tuning": [{"nll": s, "C": c} for s, c, _ in candidates], "temperatures": temps}
        arrays[family+"_logits"] = logits
        arrays[family+"_scaler_mean"], arrays[family+"_scaler_scale"] = scaler.mean_, scaler.scale_
        arrays[family+"_coef"], arrays[family+"_intercept"] = clf.coef_, clf.intercept_
        for variant, t in temps.items():
            report[family][variant] = metrics(logits[test]/t, data["y"][test], data["unit"][test], config["reject_cost"])
            decisions = typed_decisions(softmax(logits[test]/t, axis=1), data["labels"], config["reject_cost"])
            with (out/f"{family}_{variant}.jsonl").open("w") as stream:
                for record, decision in zip(data["record"][test], decisions):
                    stream.write(json.dumps({"record": str(record), **decision})+"\n")
        # Conditional uncertainty of the paired NLL difference; no independent-window bootstrap.
        losses = []
        for variant in ("unit", "pooled"):
            z = logits[test]/temps[variant]
            losses.append(logsumexp(z, axis=1)-z[np.arange(test.sum()), data["y"][test]])
        groups = data["unit"][test]
        delta = np.array([(losses[0]-losses[1])[groups == g].mean() for g in np.unique(groups)])
        if len(delta) < 2:
            raise ValueError("Fewer than two independent test units; cannot estimate unit uncertainty")
        rng = np.random.default_rng(config["bootstrap_seed"])
        boot = rng.choice(delta, size=(config["bootstrap_repetitions"], len(delta)), replace=True).mean(1)
        report[family]["unit_minus_pooled_nll"] = {"mean": float(delta.mean()), "conditional_ci95": np.quantile(boot, [.025, .975]).tolist(), "units": len(delta)}
    np.savez_compressed(out/"predictions.npz", **arrays)
    (out/"metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    (out/"config.json").write_text(json.dumps(config, indent=2))
    (out/"provenance.json").write_text(json.dumps({"data": str(data["provenance"]), "embedding": str(embeddings["provenance"])}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("export")
    for flag in ("factory-root", "factory-json", "units-csv", "labels-json", "signal-key", "label-key", "out"):
        p.add_argument("--"+flag, required=True)
    p = sub.add_parser("embed")
    for flag in ("data", "revision", "out"):
        p.add_argument("--"+flag, required=True)
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("run")
    for flag in ("data", "embeddings", "config", "out"):
        p.add_argument("--"+flag, required=True)
    args = parser.parse_args()
    if args.command == "export":
        root = Path(args.factory_root).resolve()
        sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        if sha != PHMFACTORY_REVISION:
            raise ValueError("PHMFactory revision differs from the inspected contract")
        from src.data_factory import build_data
        import src.data_factory as source
        if not Path(source.__file__).resolve().is_relative_to(root):
            raise ValueError("Wrong src.data_factory import; set PYTHONPATH to the pinned checkout")
        cfg = json.loads(Path(args.factory_json).read_text())
        factory = build_data(SimpleNamespace(**cfg["data"]), SimpleNamespace(**cfg["task"]))
        data = export_factory(factory, args.units_csv, json.loads(Path(args.labels_json).read_text()), args.signal_key, args.label_key)
        data["provenance"] = np.asarray(json.dumps({"phmfactory_commit": sha, "factory_config": cfg, "units_csv": Path(args.units_csv).read_text()}))
        with open(args.out, "xb") as stream:
            np.savez_compressed(stream, **data)
    elif args.command == "embed":
        with np.load(args.data, allow_pickle=False) as archive:
            data = dict(archive)
        check_data(data)
        hidden = embed(data["features"], args.revision, args.device)
        with open(args.out, "xb") as stream:
            np.savez_compressed(stream, hidden=hidden, record=data["record"], provenance=np.asarray(json.dumps({"model": "Qwen/Qwen3-0.6B", "revision": args.revision, "device": args.device})))
    else:
        with np.load(args.data, allow_pickle=False) as d, np.load(args.embeddings, allow_pickle=False) as e:
            run(dict(d), dict(e), json.loads(Path(args.config).read_text()), args.out)


if __name__ == "__main__":
    main()
