"""Shared supervised fitting and record-within-unit evaluation.

Inputs are arrays already exported by the native PHMFactory path. No raw-data
reader, split algorithm or paper-experiment routing is implemented here.
"""
from __future__ import annotations

import copy
import csv
import json
import math
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch
from scipy.signal import hilbert
from sklearn.ensemble import RandomForestClassifier
from torch.nn import functional as F

from jev_semantic import SignalDecisionModel, load_signal_model
from jev_metrics import record_unit_weights, score_logits, score_probabilities


def exported_role(run: Path, role: str):
    folder = run / "exports" / role
    x = np.load(folder / "x.npy", mmap_mode="r", allow_pickle=False)
    with np.load(folder / "observations.npz", allow_pickle=False) as z:
        meta = dict(z)
    if x.ndim != 3 or len(x) != len(meta["y"]) or set(meta["y"]) != {0, 1, 2}:
        raise ValueError(f"Invalid or incomplete closed-set factory export: {role}")
    return x, meta, record_unit_weights(meta["unit"], meta["record"])


def source_scale(x, weights):
    first = second = 0.0
    for start in range(0, len(x), 64):
        batch = np.asarray(x[start:start+64], dtype=np.float64)
        w = weights[start:start+64]
        first += float(w @ batch.mean(axis=(1, 2)))
        second += float(w @ np.square(batch).mean(axis=(1, 2)))
    sd = math.sqrt(max(0.0, second - first * first))
    if not math.isfinite(sd) or sd <= 0:
        raise ValueError("Source scaling is undefined; do not insert a default scale")
    return {"mean": first, "std": sd}


def batch_tensor(x, indices, scale, device):
    a = np.array(x[indices], dtype=np.float32, copy=True)
    a = (a - scale["mean"]) / scale["std"]
    return torch.as_tensor(a, device=device)


@torch.no_grad()
def infer_logits(model, x, scale, device, batch_size, *, candidate_vectors=None):
    """Batch a fixed predictor; optional candidates are source-frozen variants."""
    model.eval()
    options = {} if candidate_vectors is None else {"candidate_vectors": candidate_vectors}
    result = []
    for start in range(0, len(x), batch_size):
        a = batch_tensor(x, slice(start, start+batch_size), scale, device)
        result.append(model(a, **options).detach().float().cpu().numpy())
    return np.concatenate(result)


@torch.no_grad()
def infer_semantic_variant(model, x, scale, device, batch_size, descriptions, snapshot,
                           *, candidate_vectors=None):
    """Evaluate a declared wording variant, preserving the canonical checkpoint.

    Source verification compiles and saves the candidate vectors. Target inference
    supplies that artifact explicitly, without selecting descriptions or refitting.
    """
    from jev_semantic import compile_semantic_variant
    if not isinstance(model, SignalDecisionModel) or model.readout != "semantic":
        raise ValueError("Wording sensitivity requires the selected semantic model")
    seconds = 0.0
    if candidate_vectors is None:
        candidate_vectors, seconds = compile_semantic_variant(model, descriptions, snapshot, device)
    vectors = candidate_vectors.to(device=device, dtype=torch.float32)
    return infer_logits(model, x, scale, device, batch_size,
                        candidate_vectors=vectors), seconds


def train_neural(model, train, tune, scale, config, *, lr, weight_decay, seed, folder: Path, device):
    """One bounded fit; step zero remains a valid selected predictor.

    Checkpoints include adapted parameters only for a frozen decoder. A cap is
    recorded separately from a source-validation early stop; it is not convergence.
    """
    x, meta, w = train; tx, tm, tw = tune
    generator = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    evaluate = lambda: score_logits(infer_logits(model, tx, scale, device, config["batch_size"]), tm["y"], tw)["nll"]
    def state():
        if isinstance(model, SignalDecisionModel):
            return copy.deepcopy(model.adaptation_state())
        return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    started = time.monotonic()
    best, best_step, best_state = evaluate(), 0, state()
    trace = [(0, best)]
    stale, step, reason = 0, 0, "update_cap"
    elapsed = 0.0
    while step < config["max_updates"]:
        for offset in range(0, len(x), config["batch_size"]):
            if offset == 0:
                order = generator.permutation(len(x))
            indices = order[offset:offset+config["batch_size"]]
            # Batch objective is an unbiased uniformly sampled weighted empirical loss.
            model.train(); optimizer.zero_grad(set_to_none=True)
            inputs = batch_tensor(x, indices, scale, device)
            labels = torch.as_tensor(meta["y"][indices], dtype=torch.long, device=device)
            weights = torch.as_tensor(w[indices] * len(x), dtype=torch.float32, device=device)
            loss = (F.cross_entropy(model(inputs).float(), labels, reduction="none") * weights).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss is nonfinite")
            loss.backward(); optimizer.step(); step += 1
            if step % config["validate_every"] == 0 or step == config["max_updates"]:
                validation = evaluate(); trace.append((step, validation))
                if validation < best:
                    best, best_step, best_state, stale = validation, step, state(), 0
                else:
                    stale += 1
                if stale >= config["patience"]:
                    reason = "source_early_stop"
                    break
            elapsed = time.monotonic() - started
            if elapsed >= config["max_seconds"]:
                reason = "time_cap"
                break
            if step >= config["max_updates"]:
                break
        if reason in {"time_cap", "source_early_stop"} or step >= config["max_updates"]:
            break
    torch.save(best_state, folder / "model.pt")
    with (folder / "trace.csv").open("x", newline="") as f:
        writer = csv.writer(f); writer.writerow(["step", "source_nll"]); writer.writerows(trace)
    return {"source_nll": best, "selected_step": best_step, "last_step": step,
            "termination": reason, "seconds": time.monotonic()-started,
            "qualification": "bounded_source_selection_only" if reason == "source_early_stop" else "cap_reached_uncertain"}


def signal_statistics(x):
    """Fixed full-band time/envelope baseline; no fault-frequency or target tuning."""
    a = np.asarray(x, dtype=np.float64).squeeze(-1)
    def six(v):
        rms = np.sqrt(np.square(v).mean(1)); sd = v.std(1); centered = v-v.mean(1, keepdims=True)
        return np.column_stack((rms, sd, np.abs(v).mean(1), np.ptp(v, axis=1),
            np.divide(np.max(np.abs(v), axis=1), rms, out=np.zeros_like(rms), where=rms>0),
            np.divide(np.mean(centered**4, axis=1), sd**4, out=np.zeros_like(sd), where=sd>0)))
    envelope = np.abs(hilbert(a, axis=1))
    return np.column_stack((six(a), six(envelope)))


def feature_array(x):
    return np.concatenate([signal_statistics(np.asarray(x[i:i+64])) for i in range(0,len(x),64)])


def model_from_config(cfg, device):
    if cfg["model"]["kind"] == "decoder":
        return load_signal_model(cfg["model"], device)[0]
    if cfg["model"]["kind"] == "factory":
        from types import SimpleNamespace
        from src.model_factory.model_factory import model_factory
        return model_factory(SimpleNamespace(**cfg["model"]["factory"]), metadata=None).to(device)
    raise ValueError("Neural model kind is not registered")


def candidates(cfg):
    if cfg["model"]["kind"] == "rf":
        return [dict(max_depth=d, min_samples_leaf=l) for d,l in
                product(cfg["search"]["max_depth"], cfg["search"]["min_samples_leaf"])]
    return [dict(lr=a, weight_decay=b) for a,b in
            product(cfg["search"]["lr"], cfg["search"]["weight_decay"])]


def fit_candidate(cfg, run: Path, device, index: int, seed: int):
    """One bounded candidate, called through the existing CLI in an isolated process."""
    import joblib
    start = time.monotonic()
    grid = candidates(cfg)
    if index not in range(len(grid)) or seed not in cfg["seeds"]:
        raise ValueError("Candidate/seed is outside the configured search")
    parameters = grid[index]
    name = f"trial_{index:02d}" if seed == cfg["seeds"][0] else f"refit_{seed}"
    folder = run / "fits" / cfg["name"] / name
    folder.mkdir(exist_ok=False)
    try:
        torch.manual_seed(seed); np.random.seed(seed)
        train = exported_role(run, "train"); tune = exported_role(run, "tune")
        scale = json.loads((run / "source_scale.json").read_text())
        if cfg["model"]["kind"] == "rf":
            fx, ft = feature_array(train[0]), feature_array(tune[0])
            model = RandomForestClassifier(n_estimators=cfg["model"]["trees"], random_state=seed,
                                           n_jobs=cfg["runtime"]["cpu_threads"], **parameters)
            model.fit(fx, train[1]["y"], sample_weight=train[2] * len(fx))
            probabilities = model.predict_proba(ft)
            if list(model.classes_) != [0,1,2]:
                raise ValueError("RF output is not aligned to H/I/O")
            info = {"source_nll": score_probabilities(probabilities,tune[1]["y"],tune[2],
                    floor=cfg["evaluation"]["probability_floor"])["nll"],
                    "termination": "completed", "qualification": "bounded_source_selection_only", "file": "model.joblib"}
            joblib.dump(model, folder / info["file"])
        else:
            model = model_from_config(cfg, device)
            remaining = cfg["fit"]["max_seconds"] - (time.monotonic() - start) - 10
            if remaining <= 0:
                raise TimeoutError("No fitting time remains after loading the configured model")
            fit_config = {**cfg["fit"], "max_seconds": remaining}
            info = train_neural(model,train,tune,scale,fit_config,seed=seed,folder=folder,
                                device=device,**parameters)
            info["file"] = "model.pt"
        elapsed = time.monotonic() - start
        if elapsed > cfg["fit"]["max_seconds"]:
            raise TimeoutError("Per-candidate time cap exceeded")
        info.update(parameters=parameters,seed=seed,path=f"{name}/{info['file']}",seconds=elapsed)
        (folder / "result.json").write_text(json.dumps(info,indent=2))
        return info
    except Exception as exc:
        (folder / "failure.json").write_text(json.dumps({"type":type(exc).__name__,
            "message":str(exc),"parameters":parameters,"seed":seed},indent=2))
        raise


def fit_family(cfg, run: Path, device, root: Path):
    """Select candidates using the same CLI; subprocess timeouts enforce fit limits.

    This is serial isolation of one algorithmic fit, not another training runner.
    A timed-out process is killed, its files retained, and the study stops.
    """
    import subprocess
    import sys
    family = run / "fits" / cfg["name"]
    family.mkdir(parents=True,exist_ok=True)
    specfile = family / "config.json"
    if specfile.exists():
        if json.loads(specfile.read_text()) != cfg:
            raise ValueError("Cannot resume with a changed scientific configuration")
    else:
        specfile.write_text(json.dumps(cfg,indent=2))
    resultfile = family / "selection.json"
    if resultfile.exists():
        previous = json.loads(resultfile.read_text())
        for r in previous["models"]:
            if not (family / r["path"]).is_file():
                raise FileNotFoundError("Saved completion has a missing model")
        return previous
    grid = candidates(cfg)
    def one(index,seed):
        name = f"trial_{index:02d}" if seed == cfg["seeds"][0] else f"refit_{seed}"
        folder = family / name
        finished = folder / "result.json"
        if finished.exists():
            answer = json.loads(finished.read_text())
            if answer["parameters"] != grid[index] or answer["seed"] != seed or not (folder/answer["file"]).is_file():
                raise ValueError("Saved candidate disagrees with the active configuration")
            return answer
        if folder.exists():
            raise FileExistsError(f"Incomplete fit retained at {folder}; no implicit retry")
        log = family / f"{name}.process.log"
        # The same command remains the sole user-visible scientific runner.
        command = [sys.executable,str(root/"src/S03_Scripts/run_jev_study.py"),"semantic",
                   "fit-candidate","--config",str(specfile),"--run-root",str(run),
                   "--candidate-index",str(index),"--fit-seed",str(seed)]
        with log.open("x") as stream:
            try:
                subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT,
                               timeout=cfg["fit"]["max_seconds"])
            except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as exc:
                failure = family / f"{name}.process_failure.json"
                failure.write_text(json.dumps({"type":type(exc).__name__,"message":str(exc),
                    "command":command,"log":str(log),"scope":"failed candidate, not a scientific result"},indent=2))
                raise
        if not finished.is_file():
            raise RuntimeError("Candidate returned without the required result")
        return json.loads(finished.read_text())
    summaries = [one(i,cfg["seeds"][0]) for i in range(len(grid))]
    winner_index = min(range(len(summaries)),key=lambda i:(summaries[i]["source_nll"],i))
    if summaries[winner_index]["qualification"] != "bounded_source_selection_only":
        raise RuntimeError(
            f"Source-selected trial_{winner_index:02d} is not source-qualified "
            f"({summaries[winner_index]['qualification']}); preserve all completed candidates "
            "and stop before additional-seed refits or target release"
        )
    selected = [summaries[winner_index]] + [one(winner_index,seed) for seed in cfg["seeds"][1:]]
    outcome = {"models":selected,"trials":summaries,
               "scope":"Source-only bounded selection; cap and convergence limitations are explicit"}
    resultfile.write_text(json.dumps(outcome,indent=2))
    return outcome
