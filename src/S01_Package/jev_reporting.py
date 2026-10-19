"""Statistics, figures and timings from the approved study's saved outputs.

No data eligibility, model selection, target-driven threshold choice or fitting is
performed here. Numerical fixtures used to test this module are not PHM results.
"""
from __future__ import annotations

import csv
import io
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261017


def _write_text(path: Path, text: str) -> None:
    """Reuse an identical derived result; retain conflicting outputs for inspection."""
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f"Saved output disagrees with recomputation: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as handle:
            handle.write(text)


def _write_json(path: Path, value: object) -> None:
    _write_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot silently save an empty result: {path}")
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _write_text(path, text.getvalue())


def _weights(saved: dict) -> np.ndarray:
    from jev_learning import record_unit_weights
    weights = record_unit_weights(saved["unit"], saved["record"])
    if "weights" in saved and not np.allclose(weights, saved["weights"], rtol=0, atol=1e-12):
        raise ValueError("Saved weights differ from record-within-physical-unit weights")
    return weights


def prediction_distribution(saved: dict, *, calibrated: bool = False,
                            floor: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Return actual probabilities and scoring log probabilities.

    Only a raw probability's NLL uses the declared numerical floor. Its Brier,
    argmax and confidence retain the original distribution, including exact zeros.
    Neural logits use stable log softmax without a probability floor.
    """
    if not 0 < floor < 1:
        raise ValueError("Invalid probability floor")
    y = np.asarray(saved["y"])
    if y.ndim != 1 or not len(y) or not np.issubdtype(y.dtype, np.integer) or not np.isin(y, [0, 1, 2]).all():
        raise ValueError("Need nonempty integer H/I/O labels")
    if "raw_probabilities" in saved and not calibrated:
        probabilities = np.asarray(saved["raw_probabilities"], dtype=np.float64)
        if (probabilities.shape != (len(y), 3) or not np.isfinite(probabilities).all()
                or (probabilities < 0).any() or not np.allclose(probabilities.sum(1), 1, rtol=0, atol=1e-10)):
            raise ValueError("Invalid saved probability distribution")
        return probabilities, np.log(np.maximum(probabilities, floor))
    logits = np.asarray(saved["logits"], dtype=np.float64)
    if logits.shape != (len(y), 3) or not np.isfinite(logits).all():
        raise ValueError("Invalid saved class logits")
    temperature = float(saved["temperature"]) if calibrated else 1.0
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid source-fixed temperature")
    logits = logits / temperature
    logp = logits - logsumexp(logits, axis=1, keepdims=True)
    return np.exp(logp), logp


def score_saved(saved: dict, *, calibrated: bool = False, floor: float = 1e-12) -> dict:
    probabilities, logp = prediction_distribution(saved, calibrated=calibrated, floor=floor)
    y, weights = np.asarray(saved["y"]), _weights(saved)
    prediction = probabilities.argmax(1)
    confidence, correct = probabilities.max(1), prediction == y
    confusion = np.zeros((3, 3), dtype=float)
    np.add.at(confusion, (y, prediction), weights)
    denominator = confusion.sum(0) + confusion.sum(1)
    f1 = np.divide(2 * confusion.diagonal(), denominator, out=np.zeros(3), where=denominator > 0)
    # Ten equal-width bins: [0,.1), ... [.9,1], with confidence=1 in bin 9.
    bins = np.minimum((confidence * 10).astype(int), 9)
    ece = 0.0
    for b in range(10):
        mask = bins == b
        mass = weights[mask].sum()
        if mass > 0:
            ece += abs(float(weights[mask] @ correct[mask] - weights[mask] @ confidence[mask]))
    py = probabilities[np.arange(len(y)), y]
    return {"nll": float(weights @ -logp[np.arange(len(y)), y]),
            "error": float(weights @ ~correct), "macro_f1": float(f1.mean()),
            "brier": float(weights @ np.square(probabilities - np.eye(3)[y]).sum(1)),
            "ece": float(ece),
            "log_floor_weight": float(weights @ (py < floor)) if "raw_probabilities" in saved and not calibrated else 0.0}


def risk_coverage(probabilities: np.ndarray, y: np.ndarray, weights: np.ndarray) -> dict:
    """Retrospective weighted curve; tied confidence follows saved observation order."""
    confidence = probabilities.max(1)
    order = np.argsort(-confidence, kind="stable")
    coverage = np.cumsum(weights[order])
    errors = probabilities.argmax(1) != y
    risk = np.cumsum(weights[order] * errors[order]) / coverage
    return {"order": order, "coverage": np.r_[0.0, coverage],
            "accepted_error": np.r_[np.nan, risk], "confidence": confidence[order]}


def alarm_and_acceptance(probabilities: np.ndarray, y: np.ndarray, weights: np.ndarray,
                         *, alarm_threshold: float = .5, acceptance_threshold: float = 0.) -> dict:
    """Detection rates use ALL observations, including rejected diagnoses.

    Thresholds are fixed scenarios, not target-selected optima or validated costs.
    """
    fault = y != 0
    alarm = probabilities[:, 1:].sum(1) >= alarm_threshold
    accept = probabilities.max(1) >= acceptance_threshold
    healthy_mass, fault_mass = weights[~fault].sum(), weights[fault].sum()
    coverage = float(weights[accept].sum())
    accepted_error = None if coverage == 0 else float(weights @ (accept & (probabilities.argmax(1) != y)) / coverage)
    return {"alarm_threshold": alarm_threshold, "acceptance_threshold": acceptance_threshold,
            "coverage": coverage, "accepted_error": accepted_error,
            "false_alarm_rate": None if healthy_mass == 0 else float(weights @ (alarm & ~fault) / healthy_mass),
            "missed_fault_rate": None if fault_mass == 0 else float(weights @ (~alarm & fault) / fault_mass),
            "scenario_cost": None}


def _paired_arrays(left: dict, right: dict) -> None:
    for key in ("y", "unit", "record", "resampled_start", "resampled_stop"):
        if key not in left or key not in right or not np.array_equal(left[key], right[key]):
            raise ValueError(f"Paired outputs refer to different observations: {key}")


def paired_nll_effect(left: dict[int, dict], right: dict[int, dict], *, calibrated: bool = False,
                      floor: float = 1e-12) -> tuple[list[dict], dict]:
    """JEV minus reference, paired on declared units and conditional on saved fits.

    Average seed-specific effects within each unit BEFORE resampling units. The
    interval does not include model fitting, split or target-selection variability.
    """
    if set(left) != set(right) or not left:
        raise ValueError("Paired methods require the same nonempty prespecified seed set")
    rows, per_fit = [], []
    reference = next(iter(left.values()))
    unknown = False
    for seed in sorted(left):
        a, b = left[seed], right[seed]
        _paired_arrays(a, b)
        _paired_arrays(a, reference)
        weights = _weights(a)
        _weights(b)
        _, la = prediction_distribution(a, calibrated=calibrated, floor=floor)
        _, lb = prediction_distribution(b, calibrated=calibrated, floor=floor)
        y = a["y"]
        delta = -la[np.arange(len(y)), y] + lb[np.arange(len(y)), y]
        per_fit.append(float(weights @ delta))
        for unit in np.unique(a["unit"]):
            mask = a["unit"] == unit
            label = np.unique(y[mask])
            if len(label) != 1:
                raise ValueError("A declared P1 evaluation unit has inconsistent diagnostic states")
            identifier = str(unit).split(":")[-1].strip().lower()
            unknown |= identifier in {"", "unknown", "unverified", "tbd", "na", "null", "none", "nan"}
            if "unit_verified" in a:
                unknown |= not bool(np.asarray(a["unit_verified"])[mask].all())
            unit_weights = weights[mask] / weights[mask].sum()
            rows.append({"seed": seed, "unit": str(unit), "class_index": int(label[0]),
                         "difference": float(unit_weights @ delta[mask])})
    units = sorted({r["unit"] for r in rows})
    means = np.array([np.mean([r["difference"] for r in rows if r["unit"] == unit]) for unit in units])
    class_counts = {str(k): sum(r["class_index"] == k for r in rows if r["seed"] == min(left)) for k in range(3)}
    reason = ("Evaluation-unit identity is UNKNOWN/unverified" if unknown else
              "At least one class has fewer than two declared evaluation units" if min(class_counts.values()) < 2 else None)
    interval = None
    if reason is None:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        samples = means[rng.integers(0, len(means), size=(BOOTSTRAP_DRAWS, len(means)))].mean(1)
        interval = [float(x) for x in np.quantile(samples, [.025, .975])]
    return rows, {"mean_difference": float(means.mean()), "fit_sd": float(np.std(per_fit, ddof=1)) if len(per_fit) > 1 else None,
                  "fits": len(left), "evaluation_units": len(units), "units_per_class": class_counts,
                  "paired_unit_95_interval": interval, "interval_unavailable_reason": reason,
                  "bootstrap_draws": BOOTSTRAP_DRAWS if interval is not None else 0,
                  "bootstrap_seed": BOOTSTRAP_SEED if interval is not None else None,
                  "interval_scope": "Declared-unit sampling conditional on these fitted models and the recorded unit basis; excludes training/split variability. The interval does not verify physical independence."}


def _load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        return dict(saved)


def summarize_saved_predictions(cfg: dict, root: Path, run: Path) -> None:
    """Recompute probabilities, effects and operating curves without model calls."""
    from jev_execution import read_config
    rows, alarms, fits, saved_by_method = [], [], [], {}
    variant_rows, variant_unit_rows = [], []
    floor = cfg["evaluation"]["probability_floor"]
    report = run / "reporting"
    report.mkdir(exist_ok=True)
    unit_basis = {"test": {"dataset": "RM_027_PU"}, "external": {"dataset": "RM_007_MFPT"}}
    qualification = cfg.get("data", {}).get("qualification_file")
    evidence = []
    if qualification:
        path = Path(qualification).expanduser()
        with (path if path.is_absolute() else root / path).open(newline="") as handle:
            evidence = list(csv.DictReader(handle))
    for description in unit_basis.values():
        relevant = [r for r in evidence if r["Name"] == description["dataset"]]
        codes = sorted({r.get("unit_basis") or "basis_unprovided" for r in relevant}) or ["basis_unprovided"]
        description["unit_basis"] = codes
        description["physical_source_evidence"] = sorted({r["physical_source"] for r in relevant})
        description["native_value_units"] = sorted({r.get("value_unit") or "unspecified" for r in relevant})
        description["amplitude_scope"] = "Native values with the declared multiplier; no cross-source physical-amplitude calibration is established."
        if codes == ["documented_PU_bearing"]:
            description["basis"] = "Approved native PU bearing identities and frozen roles."
        elif codes == ["user_assumed_MFPT_experiment_bearing"]:
            description["basis"] = "One native experiment/file is one assumed bearing, as authorized by the user. Physical specimens are not independently verified; this does not establish machine generalization."
        else:
            description["basis"] = "Unit basis was not established in the supplied report inputs; this report makes no physical-specimen verification claim."
    for config_path in cfg["family_configs"]:
        family = read_config(root / config_path)
        name = family["name"]
        saved_by_method[name] = {}
        for role in ("test", "external"):
            saved_by_method[name][role] = {}
            for seed in family["seeds"]:
                saved = _load_npz(run / "evaluations" / name / str(seed) / f"{role}.npz")
                saved_by_method[name][role][seed] = saved
                for view, calibrated in (("raw", False), ("calibrated", True)):
                    metrics = score_saved(saved, calibrated=calibrated, floor=floor)
                    rows.append(dict(method=name, seed=seed, role=role, view=view, **metrics))
                    p, _ = prediction_distribution(saved, calibrated=calibrated, floor=floor)
                    weights = _weights(saved)
                    curve = risk_coverage(p, saved["y"], weights)
                    curve_path = report / f"{name}_{seed}_{role}_{view}_risk_coverage.npz"
                    if curve_path.exists():
                        previous = _load_npz(curve_path)
                        if set(previous) != set(curve) or any(not np.array_equal(previous[k], v, equal_nan=True) for k, v in curve.items()):
                            raise ValueError("Existing risk–coverage output differs from saved predictions")
                    else:
                        np.savez_compressed(curve_path, **curve)
                    for alarm_threshold in np.linspace(0., 1., 11):
                        for acceptance_threshold in np.linspace(0., 1., 11):
                            alarms.append(dict(method=name, seed=seed, role=role, view=view,
                                **alarm_and_acceptance(p, saved["y"], weights,
                                    alarm_threshold=float(alarm_threshold), acceptance_threshold=float(acceptance_threshold))))
                if "generated_class" in saved:
                    generated = np.asarray(saved["generated_class"])
                    if generated.shape != saved["y"].shape or not np.isin(generated, [-1, 0, 1, 2]).all():
                        raise ValueError("Saved generated classes must align; invalid outputs use -1")
                    weights = _weights(saved)
                    # Invalid formats remain in the denominator and count as errors.
                    rows.append(dict(method=name, seed=seed, role=role, view="generated",
                        nll=None, error=float(weights @ (generated != saved["y"])), macro_f1=None,
                        brier=None, ece=None, log_floor_weight=None))
                if name == "semantic":
                    for variant in cfg["evaluation"].get("semantic_variants", {}):
                        alternative = _load_npz(run / "evaluations" / name / str(seed) / f"{role}.variant_{variant}.npz")
                        _paired_arrays(saved, alternative)
                        if float(alternative["temperature"]) != 1.:
                            raise ValueError("Description sensitivity is raw, without variant-specific calibration")
                        base_score = score_saved(saved, floor=floor)
                        alternative_score = score_saved(alternative, floor=floor)
                        variant_rows.append(dict(method=name, seed=seed, role=role, variant=variant,
                            **alternative_score, delta_nll_vs_canonical=alternative_score["nll"] - base_score["nll"]))
                        _, canonical_logp = prediction_distribution(saved, floor=floor)
                        _, alternative_logp = prediction_distribution(alternative, floor=floor)
                        weights = _weights(saved)
                        _weights(alternative)
                        y = saved["y"]
                        delta = canonical_logp[np.arange(len(y)), y] - alternative_logp[np.arange(len(y)), y]
                        for unit in np.unique(saved["unit"]):
                            mask = saved["unit"] == unit
                            within_unit = weights[mask] / weights[mask].sum()
                            variant_unit_rows.append(dict(method=name, seed=seed, role=role, variant=variant,
                                unit=str(unit), delta_nll_vs_canonical=float(within_unit @ delta[mask])))
        for role in ("test", "external"):
            for view in ("raw", "calibrated", "generated"):
                selected = [r for r in rows if r["method"] == name and r["role"] == role and r["view"] == view]
                if not selected:
                    continue
                fit_summary = dict(method=name, role=role, view=view, fits=len(selected))
                for metric in ("nll", "error", "macro_f1", "brier", "ece"):
                    values = [r[metric] for r in selected if r[metric] is not None]
                    fit_summary[metric + "_mean"] = float(np.mean(values)) if values else None
                    fit_summary[metric + "_fit_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
                fits.append(fit_summary)
    _write_csv(run / "summary.csv", rows)
    _write_csv(report / "fit_summary.csv", fits)
    _write_csv(report / "fixed_threshold_scenarios.csv", alarms)
    if variant_rows:
        _write_csv(report / "semantic_variant_sensitivity.csv", variant_rows)
        _write_csv(report / "semantic_variant_unit_effects.csv", variant_unit_rows)
    differences, effects = [], {}
    for role in ("test", "external"):
        for view, calibrated in (("raw", False), ("calibrated", True)):
            paired, effect = paired_nll_effect(saved_by_method["semantic"][role], saved_by_method["indexed"][role], calibrated=calibrated, floor=floor)
            effect["unit_basis"] = unit_basis[role]
            effect["interval_scope"] += " " + unit_basis[role]["basis"]
            differences.extend(dict(role=role, view=view, **r) for r in paired)
            effects[role + "_" + view] = effect
    _write_csv(run / "paired_unit_effects.csv", differences)
    _write_json(report / "paired_effects.json", effects)
    _write_json(report / "definitions.json", {
        "weighting": "Equal declared evaluation units; equal original recordings within a unit; equal windows within a recording.",
        "unit_basis_by_role": unit_basis,
        "brier": "Sum of squared class-probability errors across the three classes, then weighted mean.",
        "ece": "Ten equal-width confidence bins [0,.1),...,[.9,1]; weighted confidence versus weighted accuracy.",
        "risk_coverage": "Retrospective weighted ranking, stable saved-order tie handling; first accepted error is NA at zero coverage. Not a source-selected deployment threshold.",
        "alarms": "Fixed fault-probability threshold .5 on all observations; acceptance thresholds are fixed sensitivity scenarios and never remove observations from FPR/FNR.",
        "cost": "NA: no approved scenario cost has been supplied.",
        "fit_sd": "Sample SD over actual fits (NA for one fit), distinct from declared-unit sampling intervals conditional on the recorded unit basis.",
        "primary": "External semantic minus indexed RAW unit-balanced NLL, conditional on the assumed experiment units; no target-driven seed/threshold selection."})


def _measure_operations(operations: dict, indices: np.ndarray, *, sync, warmup: int = 30,
                        samples: int = 200) -> list[dict]:
    if (warmup, samples) != (30, 200):
        raise ValueError("Approved timing requires 30 warmups and 200 measurements")
    for operation in operations.values():
        for i in range(warmup):
            operation(int(indices[i % len(indices)]))
    sync()
    rows = []
    names = list(operations)
    for block in range(5):
        for i in range(samples // 5):
            for name in names[block % len(names):] + names[:block % len(names)]:
                index = int(indices[(block * (samples // 5) + i) % len(indices)])
                sync()
                start = time.perf_counter()
                operations[name](index)
                sync()
                rows.append(dict(readout=name, block=block, measurement=i,
                                 input_index=index, milliseconds=1000 * (time.perf_counter() - start)))
    return rows


def time_selected(cfg: dict, run: Path, device) -> None:
    """Matched batch-one timing; loading, cold candidates and raw transforms separate."""
    import joblib
    import torch
    from jev_learning import exported_role, batch_tensor, signal_statistics, model_from_config
    from jev_semantic import SignalDecisionModel
    folder = run / "timing" / cfg["name"]
    folder.mkdir(parents=True, exist_ok=False)
    fit = run / "fits" / cfg["name"]
    chosen = json.loads((fit / "selection.json").read_text())["models"][0]
    x, meta, _ = exported_role(run, "train")
    scale = json.loads((run / "source_scale.json").read_text())
    indices = np.concatenate([np.flatnonzero(meta["y"] == k)[:1] for k in range(3)])
    if len(indices) != 3:
        raise ValueError("No fixed source timing input for each class")
    def sync():
        if device is not None:
            torch.cuda.synchronize(device)
    cold_ms = None
    load_started = time.perf_counter()
    predictions = []
    if cfg["model"]["kind"] == "rf":
        model = joblib.load(fit / chosen["path"])
        def predict(index):
            return model.predict_proba(signal_statistics(np.array(x[index:index+1], copy=True)))
        operations = {"native": predict}
    else:
        model = model_from_config(cfg, device)
        state = torch.load(fit / chosen["path"], map_location="cpu", weights_only=True)
        if isinstance(model, SignalDecisionModel):
            model.restore_adaptation(state)
        else:
            model.load_state_dict(state, strict=True)
        model.eval()
        cold_ms = getattr(model, "candidate_encoding_seconds", None)
        cold_ms = None if cold_ms is None else 1000 * cold_ms
        def predict(index):
            return model(batch_tensor(x, slice(index, index+1), scale, device)).float().cpu().numpy()
        operations = {cfg["model"].get("readout", "native"): predict}
        if cfg["model"].get("readout") == "token":
            def generate(index):
                inputs, mask = model._inputs(batch_tensor(x, slice(index, index+1), scale, device))
                ids = model.token_ids.tolist()
                return model._causal_lm.generate(inputs_embeds=inputs, attention_mask=mask,
                    max_new_tokens=1, do_sample=False, use_cache=True, repetition_penalty=1.0,
                    prefix_allowed_tokens_fn=lambda batch, sequence: ids).cpu().numpy()
            operations["shortest_legal_generation"] = generate
    sync()
    model_loading_ms = 1000 * (time.perf_counter() - load_started)
    with torch.no_grad():
        if device is not None:
            torch.cuda.reset_peak_memory_stats(device)
        rows = _measure_operations(operations, indices, sync=sync,
            warmup=cfg["evaluation"]["timing_warmup"], samples=cfg["evaluation"]["timing_samples"])
        for name, operation in operations.items():
            for index in indices:
                actual = np.asarray(operation(int(index)))
                row = dict(readout=name, input_index=int(index), true_class=int(meta["y"][index]),
                           output=actual.tolist())
                if name == "shortest_legal_generation":
                    token = int(actual.reshape(-1)[-1]) if actual.size else -1
                    ids = model.token_ids.tolist()
                    row.update(generated_token_id=token, predicted_class=ids.index(token) if token in ids else -1)
                else:
                    row["predicted_class"] = int(actual.argmax(1)[0])
                predictions.append(row)
        memory = None if device is None else torch.cuda.max_memory_allocated(device) / 1024**3
    _write_csv(folder / "observations.csv", rows)
    _write_json(folder / "timing_predictions.json", predictions)
    # These observations were measured during the real source export, not by
    # re-reading raw waves here. Different full-record sizes are not a matched
    # window-latency benchmark and must not be added to the window processing p95.
    with (run / "exports" / "train" / "record_lengths.csv").open() as handle:
        preprocessing = list(csv.DictReader(handle))
    if not preprocessing or any("resampling_seconds" not in r or r["Name"] != "RM_027_PU" for r in preprocessing):
        raise ValueError("Actual approved PU export preprocessing timings are missing")
    seconds = np.array([float(r["resampling_seconds"]) for r in preprocessing])
    if not np.isfinite(seconds).all() or (seconds < 0).any():
        raise ValueError("Invalid saved preprocessing observations")
    _write_csv(folder / "source_preprocessing_observations.csv", preprocessing)
    summary = {name: {"p50_ms": float(np.percentile([r["milliseconds"] for r in rows if r["readout"] == name], 50)),
                     "p95_ms": float(np.percentile([r["milliseconds"] for r in rows if r["readout"] == name], 95))} for name in operations}
    _write_json(folder / "summary.json", {"seed": chosen["seed"], "measurements": summary,
        "model_and_checkpoint_loading_ms": model_loading_ms,
        "cold_candidate_encoding_ms": cold_ms, "peak_allocated_GiB": memory,
        "source_preprocessing_records": len(preprocessing),
        "source_preprocessing_total_seconds": float(seconds.sum()),
        "source_export_record_preprocessing_p50_ms": float(np.percentile(seconds * 1000, 50)),
        "source_export_record_preprocessing_p95_ms": float(np.percentile(seconds * 1000, 95)),
        "scope": "Warm batch-one common-rate window through normalization/model/output, synchronized. Separate preprocessing contains actual PU full-record channel/scale/FIR observations saved during source export, excluding H5 I/O; its across-record percentiles mix recording sizes and are not a matched inference benchmark. Candidate encoding is initial uncached description compilation measured by the model loader on a resident backbone, excluding model load; not hardware-cold latency. Model/checkpoint loading includes initial compilation and is reported separately. Acquisition duration and raw-record preprocessing are not added to window p95. Timing predictions are source inputs, not target efficacy."})


def render_saved(cfg: dict, root: Path, run: Path) -> None:
    """Editable SVG and PDF figures read derived results only; never invoke a model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = run / "figures"
    figures.mkdir(exist_ok=True)
    with (run / "paired_unit_effects.csv").open() as handle:
        differences = list(csv.DictReader(handle))
    effects = json.loads((run / "reporting" / "paired_effects.json").read_text())
    def save(fig, name):
        fig.tight_layout(rect=(0, .09, 1, 1) if fig.texts else (0, 0, 1, 1))
        with matplotlib.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
            for extension in ("svg", "pdf"):
                fig.savefig(figures / f"{name}.{extension}")
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for index, role in enumerate(("test", "external")):
        effect = effects[role + "_raw"]
        interval = effect["paired_unit_95_interval"]
        axes[0].scatter(effect["mean_difference"], index)
        if interval is not None:
            axes[0].plot(interval, [index, index])
    axes[0].axvline(0, color="black", linewidth=.8)
    axes[0].set_yticks([0, 1], ["PU internal (exploratory)", "MFPT external"])
    axes[0].set_xlabel("JEV − indexed raw NLL (nats)")
    selected = [r for r in differences if r["role"] == "external" and r["view"] == "raw"]
    units = sorted({r["unit"] for r in selected})
    means = [np.mean([float(r["difference"]) for r in selected if r["unit"] == unit]) for unit in units]
    axes[1].scatter(np.arange(len(units)), means)
    axes[1].axhline(0, color="black", linewidth=.8)
    axes[1].set_xticks(np.arange(len(units)), units, rotation=60, ha="right")
    axes[1].set_ylabel("Per-unit mean ΔNLL over saved fits")
    axes[1].set_xlabel("Declared evaluation unit")
    fig.text(.01, .01, "MFPT units assume one experiment = one bearing; intervals condition on that assumption and saved fits.\nPhysical specimens are not independently verified; these results do not establish machine generalization.", fontsize=7)
    save(fig, "paired_diagnostic_effects")
    with (run / "summary.csv").open() as handle:
        metrics = list(csv.DictReader(handle))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for name in ("rf", "signal", "indexed", "semantic", "permuted", "token"):
        timing = run / "timing" / name / "summary.json"
        if not timing.exists():
            continue
        measured = json.loads(timing.read_text())
        for readout, values in measured["measurements"].items():
            view = "generated" if readout == "shortest_legal_generation" else "raw"
            errors = [float(r["error"]) for r in metrics if r["method"] == name and r["role"] == "external"
                      and r["view"] == view and int(r["seed"]) == measured["seed"]]
            if not errors:
                if view == "generated":
                    continue
                raise ValueError("Timing has no matching same-fit target error")
            ax.scatter(values["p95_ms"], np.mean(errors) * 100, label=name + " / " + readout)
    ax.set_xlabel("Warm batch-one window processing p95 (ms)")
    ax.set_ylabel("Full-coverage target error (%)")
    if ax.collections:
        ax.legend(fontsize=7)
    fig.text(.01, .01, "MFPT error uses user-assumed experiment units.\nIndependent physical specimens and machine generalization are not established.", fontsize=7)
    save(fig, "error_processing_cost")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    # All fit curves are visible; do not pick the most favorable seed or smooth.
    for name in ("rf", "signal", "indexed", "semantic", "token"):
        curves = sorted((run / "reporting").glob(f"{name}_*_external_raw_risk_coverage.npz"))
        for i, path in enumerate(curves):
            curve = _load_npz(path)
            axes[0].plot(curve["coverage"], curve["accepted_error"], linewidth=.8,
                         label=name if i == 0 else None)
    axes[0].set_xlabel("Accepted coverage (declared-unit weight)")
    axes[0].set_ylabel("Accepted classification error")
    axes[0].legend(fontsize=7)
    with (run / "reporting" / "fixed_threshold_scenarios.csv").open() as handle:
        alarm_rows = list(csv.DictReader(handle))
    names = sorted({r["method"] for r in alarm_rows})
    for i, name in enumerate(names):
        chosen = [r for r in alarm_rows if r["method"] == name and r["role"] == "external" and r["view"] == "raw" and float(r["acceptance_threshold"]) == 0 and float(r["alarm_threshold"]) == .5]
        for offset, metric, marker in ((-.12, "false_alarm_rate", "o"), (.12, "missed_fault_rate", "s")):
            values = [float(r[metric]) for r in chosen if r[metric] != ""]
            if values:
                axes[1].scatter([i + offset] * len(values), values, marker=marker,
                                label=metric if i == 0 else None)
    axes[1].set_xticks(np.arange(len(names)), names, rotation=45, ha="right")
    axes[1].set_ylabel("Full-population rate; fixed fault threshold .5")
    axes[1].legend(fontsize=7)
    fig.text(.01, .01, "MFPT weighting uses the user-authorized one-experiment/one-bearing assumption; physical specimens are not independently verified.", fontsize=7)
    save(fig, "selective_diagnosis_and_alarms")
