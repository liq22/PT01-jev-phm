"""Jev-inspired PHM probe. No Jev weights, RLCD reproduction, or text generation."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
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
QWEN_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
DEFAULT_GPU_UUID = "GPU-d77295d4-8e79-f0fa-8d82-791c244a2148"
FEATURES = ("rms", "std", "mean_abs", "peak_to_peak", "crest", "kurtosis")
ROLES = ("train", "tune", "cal", "test")


def unit_weights(groups: np.ndarray) -> np.ndarray:
    """Uniform independent unit, then uniform window within that unit."""
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    if not len(counts):
        raise ValueError("No independent units supplied")
    return 1.0 / (len(counts) * counts[inverse])


def window_training_weights(file_id: np.ndarray, base_weight: float = 64.) -> np.ndarray:
    """Keep each source recording's training mass fixed as coverage changes."""
    _, inverse, counts = np.unique(file_id, return_inverse=True, return_counts=True)
    if not len(counts) or not np.isfinite(base_weight) or base_weight <= 0:
        raise ValueError("Need source recordings and a positive recording weight")
    return base_weight / counts[inverse]


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
    if "sample_weight" in data:
        weights = data["sample_weight"]
        if weights.shape != (n,) or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("Invalid training sample weights")


def export_factory(factory, units_path, label_map, signal_key, label_key, channel=None) -> dict:
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
    result = {k: [] for k in ("features", "y", "unit", "role", "record", "file_id", "window_start", "window_stop")}
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
            signal = np.asarray(item[signal_key])
            if channel is not None:
                if signal.ndim != 2 or not 0 <= channel < signal.shape[1]:
                    raise ValueError(f"Cannot select channel {channel} from upstream shape {signal.shape}")
                signal = signal[:, channel]
            sample = dataset.file_windows_list[i]
            source = dataset.dataset_dict[sample["file_id"]]
            start, stop = source.window_intervals[sample["window_id"]]
            if stop-start != len(signal):
                raise ValueError("Upstream window interval and transformed signal length disagree")
            result["features"].append(signal_features(signal))
            result["y"].append(y)
            result["unit"].append(row["unit_id"])
            result["role"].append(row["role"])
            result["record"].append(f"{file_id}:{start}:{stop}")
            result["file_id"].append(file_id)
            result["window_start"].append(start)
            result["window_stop"].append(stop)
            seen_files.add(file_id)
    if seen_files != set(mapping):
        raise ValueError("units.csv must cover exactly the files selected by PHMFactory")
    data = {k: np.asarray(v) for k, v in result.items()}
    data["labels"] = np.asarray(labels)
    data["sample_weight"] = window_training_weights(data["file_id"])
    check_data(data)
    return data


def build_queries(features: np.ndarray, labels: np.ndarray | list[str], healthy_index: int = 0) -> dict[str, list[str]]:
    """Matched taxonomy and observation; no per-sample answer is included."""
    if not 0 <= healthy_index < len(labels) or len(labels) < 3:
        raise ValueError("Question decomposition requires health and at least two fault classes")
    faults = [str(label) for i, label in enumerate(labels) if i != healthy_index]
    definition = (f"The mutually exclusive classes are {', '.join(map(str, labels))}. "
                  f"The healthy class is {labels[healthy_index]}; the defined fault classes are {', '.join(faults)}. ")
    if any("dominant" in str(label).lower() for label in labels):
        definition += ("A dominant fault label identifies the predominant annotated damage location and may include "
                       "mixed damage; it does not assert a pure isolated fault. ")
    questions = {
        "flat_a": "Which one of the defined classes describes this vibration window?",
        "flat_b": "Choose the most appropriate class from the complete defined diagnosis classes for this vibration window.",
        "presence": "Does this vibration window belong to any of the defined fault classes rather than the healthy class?",
        "identity": "Conditional on this vibration window belonging to one of the defined fault classes, which fault class describes it?",
        "legacy": "Represent this vibration window for fault diagnosis.",
    }
    result = {key: [] for key in questions}
    for row in features:
        state = "; ".join(f"{k}={v:.8g}" for k, v in zip(FEATURES, row))
        for key, question in questions.items():
            result[key].append(("" if key == "legacy" else definition) + question + " " + state)
    return result


def restrict_cuda(device: str, gpu_uuid: str = DEFAULT_GPU_UUID) -> dict:
    """Set one permitted physical GPU before any CUDA initialization."""
    if device == "cpu":
        return {"device": "cpu"}
    if device not in {"cuda", "cuda:0"}:
        raise ValueError("Select one physical GPU by UUID and use logical cuda:0")
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"], text=True)
    inventory = [row.split(",", 2) for row in rows.splitlines()]
    match = [(int(index.strip()), uuid.strip(), name.strip())
             for index, uuid, name in inventory if uuid.strip() == gpu_uuid]
    if len(match) != 1 or match[0][0] == 2:
        raise ValueError("Requested GPU is absent or is forbidden physical GPU 2")
    loaded = sys.modules.get("torch")
    if loaded is not None and loaded.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before the physical-device restriction")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    return {"device": "cuda:0", "physical_index": match[0][0], "uuid": gpu_uuid, "name": match[0][2]}


def embed_queries(features: np.ndarray, labels: np.ndarray | list[str], revision: str, device: str = "cpu", batch_size: int = 8,
                  query_names=("legacy", "flat_a", "flat_b", "presence", "identity"),
                  healthy_index: int = 0, dtype: str = "float32", gpu_uuid: str = DEFAULT_GPU_UUID,
                  progress: bool = True) -> dict:
    """Load frozen Qwen once; export each named last-nonpadding representation."""
    if revision != QWEN_REVISION:
        raise ValueError("This protocol fixes the inspected Qwen3-0.6B revision")
    if batch_size <= 0 or dtype not in {"float32", "bfloat16", "float16"}:
        raise ValueError("Invalid batch size or model precision")
    # Legacy E1 can still run on a binary taxonomy; E2 is checked separately.
    names = tuple(query_names)
    valid = {"legacy", "flat_a", "flat_b", "presence", "identity"}
    if not names or len(set(names)) != len(names) or not set(names) <= valid:
        raise ValueError("Unknown or duplicate query name")
    mapping = restrict_cuda(device, gpu_uuid)
    import torch
    from transformers import AutoModel, AutoTokenizer
    device = mapping["device"]
    if device != "cpu":
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Expected exactly one permitted visible physical GPU")
        logical_uuid = str(torch.cuda.get_device_properties(0).uuid)
        if logical_uuid.lower().removeprefix("gpu-") != gpu_uuid.lower().removeprefix("gpu-"):
            raise RuntimeError("Physical UUID and logical CUDA device do not match")
        mapping["logical_uuid"] = logical_uuid
    name = "Qwen/Qwen3-0.6B"
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=False,
                                              token=False, local_files_only=True)
    tokenizer.padding_side = "right"
    model = AutoModel.from_pretrained(name, revision=revision, trust_remote_code=False,
                                     token=False, local_files_only=True,
                                     torch_dtype=getattr(torch, dtype)).to(device).eval()
    if device != "cpu":
        torch.cuda.synchronize()
    metadata = {"model": name, "revision": revision, "dtype": dtype, "batch_size": batch_size,
                "mapping": mapping, "load_seconds": time.perf_counter()-started,
                "representation": "last_nonpadding_token", "queries": {}}
    result = {"features": np.asarray(features), "labels": np.asarray(labels)}
    with torch.inference_mode():
        for query in names:
            chunks, tokens, padded_tokens = [], [], 0
            began, forwards = time.perf_counter(), 0
            for start in range(0, len(features), batch_size):
                rows = features[start:start+batch_size]
                if query == "legacy":
                    texts = ["Represent this vibration window for fault diagnosis. " +
                             "; ".join(f"{k}={v:.8g}" for k, v in zip(FEATURES, row)) for row in rows]
                else:
                    texts = build_queries(rows, labels, healthy_index)[query]
                texts = [tokenizer.apply_chat_template([{"role": "user", "content": content}],
                         tokenize=False, add_generation_prompt=True, enable_thinking=False) for content in texts]
                inputs = tokenizer(texts, padding=True, truncation=False, return_tensors="pt").to(device)
                counts = inputs["attention_mask"].sum(1)
                hidden = model(**inputs, use_cache=False).last_hidden_state
                chunks.append(hidden[torch.arange(len(texts), device=device), counts-1].float().cpu().numpy())
                tokens.append(counts.cpu().numpy())
                padded_tokens += int(inputs["input_ids"].numel())
                forwards += 1
                if progress and (forwards == 1 or forwards % 100 == 0):
                    print(f"{query}: {start+len(rows)}/{len(features)} windows", flush=True)
            if device != "cpu":
                torch.cuda.synchronize()
            key = "hidden" if query == "legacy" else f"hidden_{query}"
            result[key] = np.concatenate(chunks)
            result[f"tokens_{query}"] = np.concatenate(tokens)
            metadata["queries"][query] = {"seconds": time.perf_counter()-began, "forward_batches": forwards,
                                          "model_calls_per_window": 1, "tokens": int(result[f"tokens_{query}"].sum()),
                                          "padded_tokens": padded_tokens, "windows": len(features)}
    metadata["total_seconds"] = time.perf_counter()-started
    result["metadata"] = metadata
    return result


def embed(features, revision: str, device: str, batch_size: int = 8) -> np.ndarray:
    return embed_queries(features, [], revision, device, batch_size, query_names=("legacy",))["hidden"]


def _fit_classifier(x: np.ndarray, y: np.ndarray, c: float,
                    sample_weight: np.ndarray | None = None) -> LogisticRegression:
    began = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier = LogisticRegression(C=c, solver="lbfgs", max_iter=2000, tol=1e-7,
                                        class_weight=None).fit(x, y, sample_weight=sample_weight)
    print(f"Head C={c:g}, rows={len(y)}, dimensions={x.shape[1]}, "
          f"iterations={classifier.n_iter_.tolist()}, seconds={time.perf_counter()-began:.2f}", flush=True)
    return classifier


def fit_head(x: np.ndarray, y: np.ndarray, C: float, sample_weight: np.ndarray | None = None) -> dict:
    if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all():
        raise ValueError("Invalid supervised head observations")
    scaler = StandardScaler().fit(x, sample_weight=sample_weight)
    classifier = _fit_classifier(scaler.transform(x), y, C, sample_weight)
    return {"scaler": scaler, "classifier": classifier}


def _head_log_probs(classifier, x: np.ndarray) -> np.ndarray:
    """Keep binary and multiclass probabilities in log space without underflow."""
    score = classifier.decision_function(x)
    if score.ndim == 1:
        score = np.column_stack((np.zeros(len(score)), score))
    return score-logsumexp(score, axis=1, keepdims=True)


def predict_head(head: dict, x: np.ndarray) -> np.ndarray:
    return _head_log_probs(head["classifier"], head["scaler"].transform(x))


def select_head(x: np.ndarray, y: np.ndarray, roles: np.ndarray, units: np.ndarray,
                C_grid: list[float], sample_weight: np.ndarray | None = None) -> tuple[dict, float, list[dict]]:
    """Select on tune units only; this function never reads test outcomes."""
    train, tune = roles == "train", roles == "tune"
    weights = None if sample_weight is None else np.asarray(sample_weight)[train]
    scaler = StandardScaler().fit(x[train], sample_weight=weights)
    train_x, tune_x = scaler.transform(x[train]), scaler.transform(x[tune])
    candidates = []
    for c in C_grid:
        clf = _fit_classifier(train_x, y[train], c, weights)
        score = nll(_head_log_probs(clf, tune_x), y[tune], unit_weights(units[tune]))
        candidates.append((score, c, clf))
    _, c, clf = min(candidates, key=lambda row: (row[0], row[1]))
    return {"scaler": scaler, "classifier": clf}, c, [{"nll": s, "C": c} for s, c, _ in candidates]


def compose_log_probs(presence_logits: np.ndarray, identity_logits: np.ndarray,
                      healthy_index: int = 0) -> np.ndarray:
    presence = np.asarray(presence_logits)
    identity = np.asarray(identity_logits)
    if (presence.ndim != 2 or identity.ndim != 2 or presence.shape != (len(identity), 2)
            or identity.shape[1] < 2 or not 0 <= healthy_index <= identity.shape[1]
            or not np.isfinite(presence).all() or not np.isfinite(identity).all()):
        raise ValueError("Need binary presence and at least two conditional fault categories")
    presence = presence-logsumexp(presence, axis=1, keepdims=True)
    identity = identity-logsumexp(identity, axis=1, keepdims=True)
    result = np.empty((len(identity), identity.shape[1]+1))
    result[:, healthy_index] = presence[:, 0]
    faults = [i for i in range(result.shape[1]) if i != healthy_index]
    result[:, faults] = presence[:, 1, None]+identity
    return result


def fit_factorized(x_presence: np.ndarray, x_identity: np.ndarray, y: np.ndarray, C: float,
                   sample_weight: np.ndarray | None = None, healthy_index: int = 0) -> dict:
    classes = np.unique(y)
    if len(classes) < 3 or not np.array_equal(classes, np.arange(len(classes))) or healthy_index not in classes:
        raise ValueError("Factorization needs contiguous health and at least two fault classes")
    fault = y != healthy_index
    fault_classes = classes[classes != healthy_index]
    identity_y = np.searchsorted(fault_classes, y[fault])
    weights = None if sample_weight is None else np.asarray(sample_weight)
    return {"presence": fit_head(x_presence, fault.astype(int), C, sample_weight=weights),
            "identity": fit_head(x_identity[fault], identity_y, C,
                                 sample_weight=None if weights is None else weights[fault]),
            "healthy_index": healthy_index, "fault_classes": fault_classes}


def predict_factorized(factor: dict, x_presence: np.ndarray, x_identity: np.ndarray) -> np.ndarray:
    # Every observation runs both heads; no true label participates in routing.
    presence = predict_head(factor["presence"], x_presence)
    identity = predict_head(factor["identity"], x_identity)
    return compose_log_probs(presence, identity, factor["healthy_index"])


def direct_fusion_logits(logits_a: np.ndarray, logits_b: np.ndarray) -> np.ndarray:
    if np.shape(logits_a) != np.shape(logits_b):
        raise ValueError("Direct questions must share the final category space")
    a = logits_a-logsumexp(logits_a, axis=1, keepdims=True)
    b = logits_b-logsumexp(logits_b, axis=1, keepdims=True)
    return np.logaddexp(a, b)-np.log(2.)


def risk_coverage(logits: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict[str, np.ndarray]:
    """Window prefixes with stable input-order confidence ties and unit weights."""
    check_logits(logits, y)
    p = softmax(logits, axis=1)
    confidence = p.max(1)
    order = np.argsort(-confidence, kind="stable")
    weight = unit_weights(groups)[order]
    coverage = np.cumsum(weight)
    risk = np.cumsum(weight*(p.argmax(1)[order] != y[order]))/coverage
    return {"coverage": coverage, "risk": risk, "confidence": confidence[order], "order": order}


def paired_unit_nll(logits_a: np.ndarray, logits_b: np.ndarray, y: np.ndarray, groups: np.ndarray,
                    repetitions: int = 2000, seed: int = 20260922) -> dict:
    """A minus B; interval conditions on the fitted models and this split."""
    loss_a = logsumexp(logits_a, axis=1)-logits_a[np.arange(len(y)), y]
    loss_b = logsumexp(logits_b, axis=1)-logits_b[np.arange(len(y)), y]
    names = np.unique(groups)
    delta = np.array([(loss_a-loss_b)[groups == g].mean() for g in names])
    if len(delta) < 2:
        raise ValueError("Need at least two independent units for paired uncertainty")
    boot = np.random.default_rng(seed).choice(delta, size=(repetitions, len(delta)), replace=True).mean(1)
    return {"mean": float(delta.mean()), "conditional_ci95": np.quantile(boot, [.025, .975]).tolist(),
            "units": len(delta), "unit_names": names.tolist(), "unit_deltas": delta.tolist(),
            "repetitions": repetitions, "seed": seed}


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
    curve = risk_coverage(logits, y, groups)
    return {"nll": nll(logits, y, w), "brier": float(w @ ((p-np.eye(p.shape[1])[y])**2).sum(1)),
            "accuracy": float(w @ (~error)), "macro_f1": float(f1_score(y, predicted, average="macro", sample_weight=w)),
            "ece_10": ece, "coverage": coverage,
            "selective_error": float(w @ (error & accepted))/coverage if coverage else None,
            "scenario_cost": float(w @ np.where(accepted, error, reject_cost)),
            "aurc": float(np.diff(np.r_[0., curve["coverage"]]) @ curve["risk"])}


def _sample_weights(data: dict) -> np.ndarray:
    if "sample_weight" in data:
        return np.asarray(data["sample_weight"])
    if "file_id" in data:
        return window_training_weights(data["file_id"])
    return np.ones(len(data["y"]))  # Existing numerical fixtures and original E1 archives.


def _save_head(arrays: dict, prefix: str, head: dict) -> None:
    scaler, clf = head["scaler"], head["classifier"]
    arrays[prefix+"_scaler_mean"], arrays[prefix+"_scaler_scale"] = scaler.mean_, scaler.scale_
    arrays[prefix+"_coef"], arrays[prefix+"_intercept"] = clf.coef_, clf.intercept_
    arrays[prefix+"_classes"] = clf.classes_
    arrays[prefix+"_n_iter"] = clf.n_iter_


def _select_pair(x_a, x_b, data, grid, factorized: bool, healthy_index: int):
    train, tune = data["role"] == "train", data["role"] == "tune"
    y, weights = data["y"], _sample_weights(data)
    fault = y != healthy_index
    fault_classes = np.array([i for i in range(len(data["labels"])) if i != healthy_index])
    selected_b = train & fault if factorized else train
    y_a = fault.astype(int) if factorized else y
    y_b = np.searchsorted(fault_classes, y[selected_b]) if factorized else y[selected_b]
    scaler_a = StandardScaler().fit(x_a[train], sample_weight=weights[train])
    scaler_b = StandardScaler().fit(x_b[selected_b], sample_weight=weights[selected_b])
    train_a, tune_a = scaler_a.transform(x_a[train]), scaler_a.transform(x_a[tune])
    train_b, tune_b = scaler_b.transform(x_b[selected_b]), scaler_b.transform(x_b[tune])
    candidates = []
    for c in grid:
        head_a = {"scaler": scaler_a, "classifier": _fit_classifier(train_a, y_a[train], c, weights[train])}
        head_b = {"scaler": scaler_b, "classifier": _fit_classifier(train_b, y_b, c, weights[selected_b])}
        a = _head_log_probs(head_a["classifier"], tune_a)
        b = _head_log_probs(head_b["classifier"], tune_b)
        z = compose_log_probs(a, b, healthy_index) if factorized else direct_fusion_logits(a, b)
        score = nll(z, y[tune], unit_weights(data["unit"][tune]))
        candidates.append((score, c, head_a, head_b))
    _, c, a, b = min(candidates, key=lambda row: (row[0], row[1]))
    return a, b, c, [{"nll": s, "C": c} for s, c, _, _ in candidates]


def run_e2(data: dict, embeddings: dict, config: dict, out: str | Path) -> dict:
    check_data(data)
    healthy_index = int(config.get("healthy_index", 0))
    if len(data["labels"]) < 3 or not 0 <= healthy_index < len(data["labels"]):
        raise ValueError("E2 needs a declared healthy class and at least two fault classes")
    if not np.array_equal(embeddings["record"], data["record"]):
        raise ValueError("Question embeddings are not aligned to observation rows")
    if "features" in embeddings and not np.array_equal(embeddings["features"], data["features"]):
        raise ValueError("Question embeddings describe a different observation protocol")
    if "labels" in embeddings and not np.array_equal(embeddings["labels"], data["labels"]):
        raise ValueError("Question embeddings use a different diagnostic taxonomy")
    for name in ("flat_a", "flat_b", "presence", "identity"):
        x = embeddings["hidden_"+name]
        if x.ndim != 2 or len(x) != len(data["y"]) or not np.isfinite(x).all():
            raise ValueError(f"Invalid {name} representation")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    arrays = {k: data[k] for k in ("y", "unit", "role", "record", "labels")}
    for key in ("file_id", "window_start", "window_stop"):
        if key in data:
            arrays[key] = data[key]
    arrays["sample_weight"] = _sample_weights(data)
    grid = config["C_grid"]
    numeric, flat_a = data["features"], embeddings["hidden_flat_a"]
    workflows = {
        "Numerical-flat": (numeric, None, False),
        "Numerical-factorized": (numeric, numeric, True),
        "Qwen-flat": (flat_a, None, False),
        "Qwen-factorized": (embeddings["hidden_presence"], embeddings["hidden_identity"], True),
        "Qwen-two-question-direct": (flat_a, embeddings["hidden_flat_b"], False),
        "Qwen-shared-factorized": (flat_a, flat_a, True),
    }
    cal, test = data["role"] == "cal", data["role"] == "test"
    report, chosen = {}, {}
    for name, (x_a, x_b, factorized) in workflows.items():
        print(f"Fitting {name}", flush=True)
        began = time.perf_counter()
        prefix = name.replace("-", "_")
        if x_b is None:
            head, c, tuning = select_head(x_a, data["y"], data["role"], data["unit"], grid, arrays["sample_weight"])
            logits = predict_head(head, x_a)
            _save_head(arrays, prefix, head)
            parameters = head["classifier"].coef_.size + head["classifier"].intercept_.size
        else:
            a, b, c, tuning = _select_pair(x_a, x_b, data, grid, factorized, healthy_index)
            z_a, z_b = predict_head(a, x_a), predict_head(b, x_b)
            logits = compose_log_probs(z_a, z_b, healthy_index) if factorized else direct_fusion_logits(z_a, z_b)
            arrays[prefix+"_branch_a_logits"], arrays[prefix+"_branch_b_logits"] = z_a, z_b
            _save_head(arrays, prefix+"_a", a)
            _save_head(arrays, prefix+"_b", b)
            parameters = sum(h["classifier"].coef_.size+h["classifier"].intercept_.size for h in (a, b))
        arrays[prefix+"_logits"] = logits
        temps = {"raw": 1., "pooled": temperature(logits[cal], data["y"][cal]),
                 "unit": temperature(logits[cal], data["y"][cal], data["unit"][cal])}
        report[name] = {"C": c, "tuning": tuning, "temperatures": temps,
                        "fit_evaluate_seconds": time.perf_counter()-began, "head_parameters": parameters}
        for variant, t in temps.items():
            z = logits[test]/t
            report[name][variant] = metrics(z, data["y"][test], data["unit"][test], config["reject_cost"])
            curve = risk_coverage(z, data["y"][test], data["unit"][test])
            for key in ("coverage", "risk", "confidence", "order"):
                arrays[f"{prefix}_{variant}_rc_{key}"] = curve[key]
            decisions = typed_decisions(softmax(z, axis=1), data["labels"], config["reject_cost"])
            with (out/f"{prefix}_{variant}.jsonl").open("w") as stream:
                for record, decision in zip(data["record"][test], decisions):
                    stream.write(json.dumps({"record": str(record), **decision}, allow_nan=False)+"\n")
        chosen[name] = logits[test]/temps["pooled"]
    pairs = [("Qwen-factorized", other) for other in
             ("Qwen-flat", "Qwen-shared-factorized", "Qwen-two-question-direct")]
    pairs += [("Numerical-factorized", "Numerical-flat"), ("Qwen-shared-factorized", "Qwen-flat")]
    report["contrasts"] = {f"{a}_minus_{b}": paired_unit_nll(chosen[a], chosen[b], data["y"][test], data["unit"][test],
                           config.get("bootstrap_repetitions", 2000), config.get("bootstrap_seed", 20260922)) for a, b in pairs}
    report["protocol"] = {"primary_calibration": "pooled", "healthy_index": healthy_index,
                          "training_objective": "weighted summed leaf NLL plus squared coefficients/(2C); intercepts unpenalized",
                          "sample_weight": "64/windows_in_source_record", "risk_coverage_ties": "stable observation order"}
    np.savez_compressed(out/"predictions.npz", **arrays)
    (out/"metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    (out/"config.json").write_text(json.dumps(config, indent=2, allow_nan=False))
    (out/"provenance.json").write_text(json.dumps({"data": str(data.get("provenance", "unknown")),
                                                  "embedding": str(embeddings.get("provenance", "unknown"))}, indent=2))
    return report


def run(data, embeddings, config, out):
    check_data(data)
    if not np.array_equal(embeddings["record"], data["record"]):
        raise ValueError("Embeddings are not aligned to PHMFactory rows")
    if "features" in embeddings and not np.array_equal(embeddings["features"], data["features"]):
        raise ValueError("Embeddings describe a different observation protocol")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)  # Never overwrite a successful or failed run.
    masks = {r: data["role"] == r for r in ROLES}
    report, arrays = {}, {k: data[k] for k in ("y", "unit", "role", "record", "labels")}
    arrays["sample_weight"] = _sample_weights(data)
    for key in ("file_id", "window_start", "window_stop"):
        if key in data:
            arrays[key] = data[key]
    for family, x in (("numeric", data["features"]), ("qwen", embeddings["hidden"])):
        if x.ndim != 2 or len(x) != len(data["y"]) or not np.isfinite(x).all():
            raise ValueError(f"Invalid {family} features")
        head, chosen_c, tuning = select_head(x, data["y"], data["role"], data["unit"], config["C_grid"], arrays["sample_weight"])
        logits = predict_head(head, x)
        cal, test = masks["cal"], masks["test"]
        temps = {"raw": 1., "pooled": temperature(logits[cal], data["y"][cal]),
                 "unit": temperature(logits[cal], data["y"][cal], data["unit"][cal])}
        report[family] = {"C": chosen_c, "tuning": tuning, "temperatures": temps}
        arrays[family+"_logits"] = logits
        _save_head(arrays, family, head)
        for variant, t in temps.items():
            report[family][variant] = metrics(logits[test]/t, data["y"][test], data["unit"][test], config["reject_cost"])
            for key, value in risk_coverage(logits[test]/t, data["y"][test], data["unit"][test]).items():
                arrays[f"{family}_{variant}_rc_{key}"] = value
            decisions = typed_decisions(softmax(logits[test]/t, axis=1), data["labels"], config["reject_cost"])
            with (out/f"{family}_{variant}.jsonl").open("w") as stream:
                for record, decision in zip(data["record"][test], decisions):
                    stream.write(json.dumps({"record": str(record), **decision})+"\n")
        # Conditional uncertainty of the paired NLL difference; no independent-window bootstrap.
        report[family]["unit_minus_pooled_nll"] = paired_unit_nll(logits[test]/temps["unit"], logits[test]/temps["pooled"],
            data["y"][test], data["unit"][test], config["bootstrap_repetitions"], config["bootstrap_seed"])
    np.savez_compressed(out/"predictions.npz", **arrays)
    (out/"metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    (out/"config.json").write_text(json.dumps(config, indent=2))
    (out/"provenance.json").write_text(json.dumps({"data": str(data["provenance"]), "embedding": str(embeddings["provenance"])}, indent=2))
    return report


def _namespace(value):
    # Native split fractions/partition_map must remain mappings.
    return SimpleNamespace(**value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("export")
    for flag in ("factory-root", "factory-json", "units-csv", "labels-json", "signal-key", "label-key", "out"):
        p.add_argument("--"+flag, required=True)
    p.add_argument("--channel", type=int)
    p = sub.add_parser("embed")
    for flag in ("data", "revision", "out"):
        p.add_argument("--"+flag, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    p.add_argument("--queries", nargs="+", default=["legacy"])
    p.add_argument("--healthy-index", type=int, default=0)
    p.add_argument("--gpu-uuid", default=DEFAULT_GPU_UUID)
    for command in ("run", "run-e2"):
        p = sub.add_parser(command)
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
        factory = build_data(_namespace(cfg["data"]), _namespace(cfg["task"]))
        data = export_factory(factory, args.units_csv, json.loads(Path(args.labels_json).read_text()), args.signal_key, args.label_key, args.channel)
        data["provenance"] = np.asarray(json.dumps({"phmfactory_commit": sha, "factory_config": cfg,
                                                    "channel": args.channel, "units_csv": Path(args.units_csv).read_text()}))
        with open(args.out, "xb") as stream:
            np.savez_compressed(stream, **data)
    elif args.command == "embed":
        destination = Path(args.out)
        metadata_path = destination.with_suffix(".json")
        if destination.exists() or metadata_path.exists():
            raise FileExistsError("Do not overwrite embedding arrays or metadata")
        with np.load(args.data, allow_pickle=False) as archive:
            data = dict(archive)
        check_data(data)
        output = embed_queries(data["features"], data["labels"], args.revision, args.device, args.batch_size,
                               args.queries, args.healthy_index, args.dtype, args.gpu_uuid)
        metadata = output.pop("metadata")
        output["record"] = data["record"]
        output["provenance"] = np.asarray(json.dumps(metadata, allow_nan=False))
        with open(args.out, "xb") as stream:
            np.savez_compressed(stream, **output)
        with metadata_path.open("x", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, allow_nan=False)
            stream.write("\n")
    else:
        with np.load(args.data, allow_pickle=False) as d, np.load(args.embeddings, allow_pickle=False) as e:
            runner = run_e2 if args.command == "run-e2" else run
            runner(dict(d), dict(e), json.loads(Path(args.config).read_text()), args.out)


if __name__ == "__main__":
    main()
