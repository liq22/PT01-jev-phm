"""P1 configuration loading and validation, independent of study execution.

Model and search mappings retain their existing replace-not-merge semantics.
This module reads configuration files only; it never accesses observations,
loads a model, or releases an external target.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml


def read_config(path: Path, _seen=None, _validate=True) -> dict:
    seen = set() if _seen is None else set(_seen)
    path = path.resolve()
    if path in seen:
        raise ValueError("Cyclic configuration inheritance")
    seen.add(path)
    with path.open(encoding="utf-8") as f:
        value = json.load(f) if path.suffix.lower() == ".json" else yaml.safe_load(f)
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a mapping")
    parent = value.pop("base", None)
    if parent:
        parentpath = (path.parent / parent).resolve()
        if parentpath == path.resolve():
            raise ValueError("Configuration cannot include itself")
        value = merge_config(read_config(parentpath, seen, False), value)
    if not _validate:
        return value
    required = {"name","runtime","data","signal","model","fit","search","seeds","evaluation","family_configs"}
    if set(value) != required:
        raise ValueError(f"Unexpected/missing config sections: {set(value)^required}")
    if not isinstance(value["model"], dict):
        raise ValueError("The base config is not an executable model configuration")
    if value["model"]["kind"] not in {"decoder","factory","rf"}:
        raise ValueError("Unknown algorithm kind")
    if value["signal"]["samples"] % value["model"].get("patch_size",1):
        raise ValueError("Signal length is not divisible by patch size")
    limits = value["runtime"]
    if (limits["gpu_seconds"], limits["cpu_seconds"], limits["cpu_threads"]) != (86400,43200,8):
        raise ValueError("Resource fields must match this reviewed Shell contract; no silently unconsumed budget overrides")
    if value["runtime"]["device"] != "cuda":
        raise ValueError("This approved real experiment cannot fall back to CPU")
    if value["runtime"]["forbidden_physical_gpu"] != 2:
        raise ValueError("The physical GPU 2 exclusion cannot be removed")
    if len(value["seeds"]) != len(set(value["seeds"])):
        raise ValueError("Repeated seed is not a new fit")
    if value["data"]["source"] != "RM_027_PU" or value["data"]["target"] != "RM_007_MFPT":
        raise ValueError("Dataset substitution is outside the approved experiment")
    expected = {
        "runtime": {"factory", "device", "gpu_uuid", "forbidden_physical_gpu", "dtype", "cpu_threads", "gpu_seconds", "cpu_seconds"},
        "data": {"root", "metadata", "source", "target", "pu_roles", "qualification_file"},
        "signal": {"sample_rate_hz", "samples", "kaiser_beta"},
        "fit": {"batch_size", "max_updates", "max_seconds", "validate_every", "patience"},
        "evaluation": {"probability_floor", "timing_warmup", "timing_samples", "semantic_variants"},
    }
    for section, keys in expected.items():
        if set(value[section]) != keys:
            raise ValueError(f"Unknown or missing {section} fields: {set(value[section]) ^ keys}")
    kind = value["model"]["kind"]
    model_keys = {"rf": {"kind", "trees"}, "factory": {"kind", "factory"},
                  "decoder": {"kind", "snapshot", "revision", "dtype", "patch_size", "readout", "temperature", "descriptions", "assignment", "codes"}}[kind]
    if set(value["model"]) != model_keys:
        raise ValueError("Unknown or missing model parameters")
    expected_search = {"max_depth", "min_samples_leaf"} if kind == "rf" else {"lr", "weight_decay"}
    if set(value["search"]) != expected_search or any(not isinstance(a, list) or not a for a in value["search"].values()):
        raise ValueError("Invalid or unconsumed search parameters")
    if kind == "decoder" and value["model"]["dtype"] != value["runtime"]["dtype"]:
        raise ValueError("Model and runtime precision differ")
    if kind == "factory" and set(value["model"]["factory"]) != {"type", "name", "input_dim", "num_classes", "width_multiplier", "dropout"}:
        raise ValueError("Factory baseline contains unknown or missing arguments")
    if any(not isinstance(value["fit"][k], int) or value["fit"][k] <= 0
           for k in ("batch_size", "max_updates", "validate_every", "patience")):
        raise ValueError("Training counts must be positive integers")
    if not 0 < value["fit"]["max_seconds"] <= (1200 if kind == "rf" else 1800):
        raise ValueError("Fit time allowance exceeds the approved maximum")
    return value


def merge_config(base, override):
    out = dict(base)
    for k,v in override.items():
        out[k] = merge_config(out[k],v) if k not in {"model", "search"} and k in out and isinstance(out[k],dict) and isinstance(v,dict) else v
    return out
