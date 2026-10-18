"""Config-driven handlers added to run_jev_study.py, not a second CLI runner.

The approved experiment is orchestrated in Shell/YAML. This module handles
operation types, never paper experiment numbers. Unknown physical provenance
blocks data export; no CPU, synthetic-data or model-download fallback exists.
"""
from __future__ import annotations

import argparse
import csv
import filecmp
import importlib.util
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml


def read_config(path: Path, _seen=None, _validate=True) -> dict:
    seen = set() if _seen is None else set(_seen)
    path = path.resolve()
    if path in seen:
        raise ValueError("Cyclic configuration inheritance")
    seen.add(path)
    with path.open(encoding="utf-8") as f:
        value = yaml.safe_load(f)
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


def configured_path(root, path):
    return Path(path).expanduser() if Path(path).is_absolute() else root/ path


def validate_matrix(cfg, root):
    """Keep this approved six-family, 46-fit study finite before any data access."""
    families = [read_config(root / path) for path in cfg["family_configs"]]
    expected = {"rf", "signal", "indexed", "semantic", "permuted", "token"}
    if len(families) != 6 or {family["name"] for family in families} != expected:
        raise ValueError("The approved matrix requires exactly six distinct families")
    total = 0
    for family in families:
        shared = ("runtime", "data", "signal", "evaluation", "family_configs")
        if any(family[key] != cfg[key] for key in shared):
            raise ValueError("Families do not share the approved input contract")
        count = 1
        for values in family["search"].values():
            count *= len(values)
        seeds = [101] if family["name"] == "rf" else [101, 202, 303]
        if family["seeds"] != seeds or count != 6:
            raise ValueError("Search or repetition count differs from the approved 46-fit matrix")
        total += count + len(seeds) - 1
    if total != 46:
        raise ValueError("The approved study has a 46-fit maximum")
    return families


def bind_gpu(spec):
    """Check physical identity before importing/initializing CUDA."""
    text = subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"], text=True)
    rows = [line.split(",") for line in text.strip().splitlines()]
    indices = {u.strip():int(i.strip()) for i,u in rows}
    uuid = spec["gpu_uuid"]
    if uuid not in indices or indices[uuid] == spec["forbidden_physical_gpu"]:
        raise ValueError("Requested GPU UUID is absent or identifies the forbidden GPU")
    current = os.environ.get("CUDA_VISIBLE_DEVICES")
    if current not in {None, uuid}:
        raise ValueError("CUDA visibility differs from the single approved UUID")
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before physical-device binding")
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("The bound CUDA device is unavailable; CPU fallback is forbidden")
    if spec["dtype"] == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Configured BF16 is not supported; precision is not changed automatically")
    torch.set_num_threads(spec["cpu_threads"])
    return torch.device("cuda:0")


def check_inputs(cfg, root, *, gpu=False):
    """Physical-qualification evidence is supplied by G0, not inferred by shape."""
    from jev_signal_data import FACTORY_REVISION
    issues=[]
    data_root=Path(cfg["data"]["root"])
    paths=[data_root/cfg["data"]["metadata"],data_root/"RM_027_PU.h5",data_root/"RM_007_MFPT.h5",
           root/cfg["data"]["pu_roles"]]
    if cfg["data"]["qualification_file"]:
        paths.append(configured_path(root,cfg["data"]["qualification_file"]))
    else:
        issues.append(_missing_qualification())
    for p in paths:
        if not p.is_file(): issues.append(f"Missing required input: {p}")
    factory=root/cfg["runtime"]["factory"]
    try:
        rev=subprocess.check_output(["git","-C",str(factory),"rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
        if rev!=FACTORY_REVISION: issues.append("PHMFactory revision differs from the inspected pin")
    except (OSError,subprocess.CalledProcessError): issues.append("Pinned PHMFactory checkout is unavailable")
    if cfg["model"]["kind"]=="decoder":
        snapshot=Path(cfg["model"]["snapshot"])
        if snapshot.name!=cfg["model"]["revision"] or not (snapshot/"config.json").is_file() or not list(snapshot.glob("*.safetensors")):
            issues.append(f"Declared local checkpoint is missing/incomplete: {snapshot}")
        if importlib.util.find_spec("transformers") is None: issues.append("transformers is absent from the selected interpreter")
    if gpu:
        try: bind_gpu(cfg["runtime"])
        except Exception as exc: issues.append(str(exc))
    if issues:
        raise RuntimeError("BLOCKED:\n"+"\n".join(issues))


def _csv_contents(path):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Missing or repeated CSV columns: {path}")
        return {"columns": reader.fieldnames, "rows": list(reader)}


def _missing_qualification():
    return ("data.qualification_file is unbound: original-record-to-physical-unit correspondence "
            "(MFPT specimens and repeated acquisitions), and documented signal channel/sampling-clock/scale provenance remain "
            "unverified. Native metadata availability does not establish scientific qualification")


def _metadata_contents(frame):
    """Keep only the approved native rows actually consumed, with missing values as null."""
    return {"columns": list(frame.columns),
            "rows": frame.astype(object).where(frame.notna(), None).to_dict(orient="records")}


def _study_contract(cfg, root):
    qualification = cfg["data"]["qualification_file"]
    if not qualification:
        raise ValueError("A completed G0 record map must be bound before study preparation")
    from jev_signal_data import read_approved_metadata
    frame = read_approved_metadata(root, cfg["data"])
    references = {
        "qualification": configured_path(root, qualification),
        "pu_roles": configured_path(root, cfg["data"]["pu_roles"]),
    }
    return {**{key: cfg[key] for key in ("runtime", "data", "signal", "evaluation", "family_configs")},
            "input_contents": {**{name: _csv_contents(path) for name, path in references.items()},
                               "metadata": _metadata_contents(frame)}}


def bind_study(cfg, run, root):
    """Refuse changed common inputs while allowing model-family configurations."""
    contract = _study_contract(cfg, root)
    path=run/"study.json"
    if path.exists():
        if json.loads(path.read_text())!=contract:
            raise ValueError("Run input contract changed; use a prospectively declared new study")
    else:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(contract, stream, indent=2, allow_nan=False)


def _check_bound_study(cfg, root, run):
    path = run / "study.json"
    if not path.is_file():
        raise RuntimeError("External target remains sealed: missing prepared study")
    if json.loads(path.read_text()) != _study_contract(cfg, root):
        raise ValueError("Run inputs or referenced record contents changed; preserve this study")


def _source_selection(family, run):
    base = run / "fits" / family["name"]
    selection = base / "selection.json"
    if not selection.is_file() or json.loads((base / "config.json").read_text()) != family:
        raise RuntimeError(f"External target remains sealed: unfinished {family['name']}")
    record = json.loads(selection.read_text())
    if sorted(item["seed"] for item in record["models"]) != sorted(family["seeds"]):
        raise RuntimeError("External target remains sealed: not every predeclared seed completed")
    if any(item["qualification"] != "bounded_source_selection_only" for item in record["models"]):
        raise RuntimeError("External target remains sealed: a selected fit is not source-qualified")
    for selected in record["models"]:
        if not (base / selected["path"]).is_file():
            raise FileNotFoundError("Selected model is missing; external target remains sealed")
    return base, record


def _verification_contract(family, selection, scale):
    return {"config": family, "selection": selection, "source_scale": scale}


def _checked_verification(family, run, selected, record, scale):
    base = run / "fits" / family["name"]
    verified = run / "source_verification" / family["name"] / str(selected["seed"])
    path = verified / "result.json"
    if not path.is_file():
        raise RuntimeError("External target remains sealed: source checkpoint verification incomplete")
    result = json.loads(path.read_text())
    if result.get("status") != "source_verified" or result.get("contract") != _verification_contract(family, record, scale):
        raise RuntimeError("External target remains sealed: source verification contract changed")
    original = base / selected["path"]
    snapshot = verified / original.name
    filecmp.clear_cache()
    if not snapshot.is_file() or not filecmp.cmp(original, snapshot, shallow=False):
        raise RuntimeError("External target remains sealed: verified checkpoint content changed")
    if not (verified / "cal.npz").is_file() or not np.isfinite(result["source_temperature"]) or result["source_temperature"] <= 0:
        raise RuntimeError("External target remains sealed: source calibration is incomplete")
    with np.load(verified / "cal.npz", allow_pickle=False) as calibration:
        if float(calibration["temperature"]) != result["source_temperature"]:
            raise RuntimeError("External target remains sealed: frozen source temperature changed")
    if family["name"] == "semantic":
        import torch
        for variant in family["evaluation"]["semantic_variants"]:
            if not (verified / f"cal.variant_{variant}.npz").is_file():
                raise RuntimeError("External target remains sealed: source semantic-variant check incomplete")
            vectors_path = verified / f"variant_{variant}.pt"
            if not vectors_path.is_file():
                raise RuntimeError("External target remains sealed: fixed semantic-variant vectors missing")
            vectors = torch.load(vectors_path, map_location="cpu", weights_only=True).float()
            frozen = result.get("semantic_variant_vectors", {}).get(variant)
            if frozen is None or not torch.equal(vectors, torch.tensor(frozen, dtype=torch.float32)):
                raise RuntimeError("External target remains sealed: fixed semantic-variant vectors changed")
    return verified, result


def release_target(cfg, root, run):
    """All source selections must exist before any external signal evaluation."""
    _check_bound_study(cfg, root, run)
    scale = json.loads((run / "source_scale.json").read_text())
    for family in validate_matrix(cfg, root):
        _, record = _source_selection(family, run)
        for selected in record["models"]:
            _checked_verification(family, run, selected, record, scale)


def add_subparser(sub):
    p=sub.add_parser("semantic",help="Config-driven qualified signal/semantic study")
    p.add_argument("stage",choices=("check","prepare","smoke","fit","fit-candidate","verify-source","infer","timing","summarize"))
    p.add_argument("--config",required=True,type=Path)
    p.add_argument("--run-root",required=True,type=Path)
    p.add_argument("--candidate-index",type=int)
    p.add_argument("--fit-seed",type=int)
    p.set_defaults(handler=dispatch)


def _require_smoke(cfg, run):
    path = run / "smoke" / "result.json"
    if not path.is_file():
        raise RuntimeError("Real source interface smoke has not passed")
    result = json.loads(path.read_text())
    family = result.get("families", {}).get(cfg["name"])
    if result.get("status") != "source_smoke_passed" or not family or family.get("config") != cfg:
        raise RuntimeError("Real source smoke does not verify this model/configuration")


def smoke_sources(cfg, root, run, device):
    """Exercise all real source interfaces before any efficacy fit is started."""
    import torch
    from jev_learning import exported_role, batch_tensor, model_from_config, signal_statistics
    from jev_semantic import SignalDecisionModel
    folder = run / "smoke"
    folder.mkdir(exist_ok=False)
    x, meta, _ = exported_role(run, "train")
    indices = np.asarray([np.flatnonzero(meta["y"] == label)[0] for label in range(3)])
    scale = json.loads((run / "source_scale.json").read_text())
    results = {}
    try:
        for family in validate_matrix(cfg, root):
            name = family["name"]
            area = folder / name
            area.mkdir()
            if family["model"]["kind"] == "rf":
                features = signal_statistics(np.asarray(x[indices]))
                if features.shape != (3, 12) or not np.isfinite(features).all():
                    raise ValueError("RF source feature interface is invalid")
                results[name] = {"config": family, "rows": 3, "feature_shape": list(features.shape),
                                 "scope": "Feature-only source interface; no RF fit"}
                continue
            torch.manual_seed(101)
            model = model_from_config(family, device)
            batch = batch_tensor(x, indices, scale, device)
            labels = torch.as_tensor(meta["y"][indices], device=device)
            model.train()
            logits = model(batch)
            if tuple(logits.shape) != (3, 3) or not torch.isfinite(logits).all():
                raise ValueError("Real source model did not produce finite H/I/O logits")
            optimizer = torch.optim.SGD([parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-4)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(logits.float(), labels)
            loss.backward()
            gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None]
            if not gradients or not all(torch.isfinite(value).all() for value in gradients) or not any(torch.any(value != 0) for value in gradients):
                raise RuntimeError("Source model did not receive finite nonzero trainable gradients")
            if any(parameter.grad is not None for parameter in model.parameters() if not parameter.requires_grad):
                raise RuntimeError("Frozen source-model parameters received gradients")
            optimizer.step()
            loss_value = float(loss.detach())
            model.eval()
            with torch.no_grad():
                expected = model(batch).detach().float().cpu()
            state = model.adaptation_state() if isinstance(model, SignalDecisionModel) else {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save(state, area / "model.pt")
            del optimizer, model, loss, logits, gradients
            torch.cuda.empty_cache()
            restored = model_from_config(family, device)
            state = torch.load(area / "model.pt", map_location="cpu", weights_only=True)
            if isinstance(restored, SignalDecisionModel):
                restored.restore_adaptation(state)
            else:
                restored.load_state_dict(state, strict=True)
            restored.eval()
            with torch.no_grad():
                actual = restored(batch).detach().float().cpu()
            if not torch.allclose(expected, actual, rtol=1e-5, atol=1e-6):
                raise ValueError("Source smoke checkpoint roundtrip changed predictions")
            torch.save({"logits": actual, "labels": labels.cpu()}, area / "output.pt")
            results[name] = {"config": family, "rows": 3, "logit_shape": [3, 3],
                             "loss": loss_value, "updates": 1,
                             "scope": "Source-only interface, gradient and checkpoint check; excluded from efficacy fits"}
            del restored, state
            torch.cuda.empty_cache()
        result = {"status": "source_smoke_passed", "families": results,
                  "scope": "All predeclared source interfaces; zero target access; no efficacy evidence"}
        with (folder / "result.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        print(json.dumps(result, indent=2))
    except Exception as exc:
        with (folder / "failure.json").open("x", encoding="utf-8") as stream:
            json.dump({"type": type(exc).__name__, "message": str(exc),
                       "completed_families": sorted(results), "scope": "Source interface failure"}, stream, indent=2)
        raise


def dispatch(args,root,upstream):
    root=Path(root);cfg=read_config(args.config.resolve());run=args.run_root.resolve()
    validate_matrix(cfg, root)
    gpu=args.stage in {"smoke", "verify-source"} or (args.stage in {"fit-candidate","infer","timing"} and cfg["model"]["kind"]!="rf")
    if args.stage=="check":
        from jev_signal_data import read_approved_metadata, summarize_native_metadata, qualify_records
        frame = read_approved_metadata(root, cfg["data"])
        print(json.dumps({"status": "native_metadata_observed",
                          "observations": summarize_native_metadata(frame),
                          "scientific_qualification": "UNVERIFIED",
                          "scope": "Approved native metadata only; no waveform or model access and no physical-unit or measurement qualification"},
                         indent=2, allow_nan=False), flush=True)
        if not cfg["data"]["qualification_file"]:
            raise RuntimeError("BLOCKED:\n" + _missing_qualification())
        check_inputs(cfg,root,gpu=False)
        upstream()
        frame,records=qualify_records(root,cfg["data"])
        print(json.dumps({"status":"metadata_contract_bound","records":len(records),
                          "scope":"Does not authenticate the physical source documents or run a model"},indent=2))
        return
    check_inputs(cfg,root,gpu=False)
    upstream()
    device=bind_gpu(cfg["runtime"]) if gpu else None
    if args.stage=="prepare":
        run.mkdir(parents=True,exist_ok=True)
        if (run/"study.json").exists():
            raise FileExistsError("Preparation already started; preserve partial output and use a new run")
    elif not (run/"study.json").is_file():
        raise FileNotFoundError("Run has no prepared source contract")
    bind_study(cfg,run,root)
    from jev_learning import exported_role,source_scale,model_from_config,fit_family,batch_tensor
    if args.stage=="prepare":
        from jev_signal_data import export_native
        export_native(root,cfg,run,target=False)
        x,meta,w=exported_role(run,"train")
        (run/"source_scale.json").write_text(json.dumps(source_scale(x,w),indent=2))
        print("Source factory export completed; external waveforms remain unopened")
    elif args.stage=="smoke":
        smoke_sources(cfg, root, run, device)
    elif args.stage=="fit":
        _require_smoke(cfg, run)
        print(json.dumps(fit_family(cfg,run,device,root),indent=2))
    elif args.stage=="fit-candidate":
        _require_smoke(cfg, run)
        if args.candidate_index is None or args.fit_seed is None: raise ValueError("Explicit candidate and seed required")
        from jev_learning import fit_candidate
        print(json.dumps(fit_candidate(cfg,run,device,args.candidate_index,args.fit_seed),indent=2))
    elif args.stage == "verify-source":
        verify_source(cfg, root, run, device)
    elif args.stage=="infer":
        release_target(cfg,root,run)
        from jev_signal_data import export_native
        if not (run/"exports"/"external"/"observations.npz").is_file():
            export_native(root,cfg,run,target=True)
        evaluate_selected(cfg,run,device)
    elif args.stage=="timing":
        release_target(cfg,root,run)
        from jev_reporting import time_selected
        time_selected(cfg,run,device)
    else:
        summarize_saved(cfg,root,run)
        from jev_reporting import render_saved
        render_saved(cfg,root,run)


def _load_predictor(cfg, family, selected, scale, device):
    """Reconstruct one saved model without accessing external observations."""
    import joblib
    import torch
    from jev_learning import infer_logits, feature_array, model_from_config
    from jev_semantic import SignalDecisionModel
    if cfg["model"]["kind"] == "rf":
        model = joblib.load(family / selected["path"])
        if list(model.classes_) != [0, 1, 2]:
            raise ValueError("Saved RF classes are incompatible with H/I/O")
        def predict(x):
            raw = model.predict_proba(feature_array(x))
            return np.log(np.maximum(raw, cfg["evaluation"]["probability_floor"])), raw
    else:
        model = model_from_config(cfg, device)
        state = torch.load(family / selected["path"], map_location="cpu", weights_only=True)
        if isinstance(model, SignalDecisionModel):
            model.restore_adaptation(state)
        else:
            model.load_state_dict(state, strict=True)
        def predict(x):
            return infer_logits(model, x, scale, device, cfg["fit"]["batch_size"]), None
    return model, predict


def _generated_predictions(model, x, scale, device, batch_size):
    """Measure the actual one-token output, including any illegal token as failure."""
    import torch
    from jev_learning import batch_tensor
    model.eval()
    ids = model.token_ids.tolist()
    classes = {token: index for index, token in enumerate(ids)}
    generated = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            inputs, mask = model._inputs(batch_tensor(x, slice(start, start + batch_size), scale, device))
            tokens = model._causal_lm.generate(inputs_embeds=inputs, attention_mask=mask,
                max_new_tokens=1, do_sample=False, use_cache=True, repetition_penalty=1.0,
                prefix_allowed_tokens_fn=lambda batch, sequence: ids)
            generated.extend(tokens[:, -1].detach().cpu().tolist())
    token_ids = np.asarray(generated, dtype=np.int64)
    return {"generated_token_id": token_ids,
            "generated_class": np.asarray([classes.get(token, -1) for token in generated], dtype=np.int64)}


def verify_source(cfg, root, run, device):
    """Load and roundtrip every selected checkpoint, then freeze source calibration."""
    import torch
    from scipy.optimize import minimize_scalar
    from jev_learning import exported_role, score_logits
    _check_bound_study(cfg, root, run)
    families = validate_matrix(cfg, root)
    selections = [(family, *_source_selection(family, run)) for family in families]
    scale = json.loads((run / "source_scale.json").read_text())
    x, meta, weights = exported_role(run, "cal")
    for family, base, record in selections:
        for selected in record["models"]:
            folder = run / "source_verification" / family["name"] / str(selected["seed"])
            if (folder / "result.json").is_file():
                _checked_verification(family, run, selected, record, scale)
                continue
            folder.mkdir(parents=True, exist_ok=False)
            try:
                local_device = None if family["model"]["kind"] == "rf" else device
                model, predict = _load_predictor(family, base, selected, scale, local_device)
                logits, raw = predict(x)
                generated = _generated_predictions(model, x, scale, local_device, family["fit"]["batch_size"]) if family["model"].get("readout") == "token" else {}
                variant_vectors = {}
                if family["name"] == "semantic":
                    from jev_semantic import compile_semantic_variant
                    from jev_learning import infer_semantic_variant
                    for variant, descriptions in family["evaluation"]["semantic_variants"].items():
                        vectors, seconds = compile_semantic_variant(model, descriptions, family["model"]["snapshot"], local_device)
                        vectors = vectors.detach().cpu()
                        torch.save(vectors, folder / f"variant_{variant}.pt")
                        frozen = torch.load(folder / f"variant_{variant}.pt", map_location="cpu", weights_only=True)
                        variant_vectors[variant] = frozen.float().tolist()
                        variant_logits, _ = infer_semantic_variant(model, x, scale, local_device,
                            family["fit"]["batch_size"], descriptions, family["model"]["snapshot"], candidate_vectors=frozen)
                        np.savez_compressed(folder / f"cal.variant_{variant}.npz", **meta,
                            logits=variant_logits, weights=weights, temperature=np.array(1.0),
                            variant=np.array(variant), descriptions=np.asarray(descriptions),
                            candidate_compile_seconds=np.array(seconds))
                del predict, model
                if local_device is not None:
                    torch.cuda.empty_cache()
                model, predict = _load_predictor(family, base, selected, scale, local_device)
                replay, replay_raw = predict(x)
                if not np.allclose(logits, replay, rtol=1e-5, atol=1e-6):
                    raise ValueError("Reloaded source predictions disagree with the selected checkpoint")
                if raw is not None and not np.array_equal(raw, replay_raw):
                    raise ValueError("Reloaded RF probabilities disagree")
                del predict, model
                if local_device is not None:
                    torch.cuda.empty_cache()
                optimum = minimize_scalar(lambda t: score_logits(logits / np.exp(t), meta["y"], weights)["nll"],
                    bounds=(np.log(.05), np.log(20)), method="bounded")
                if not optimum.success:
                    raise RuntimeError("Source-only temperature optimization failed")
                temperature = float(np.exp(optimum.x)) if optimum.fun < score_logits(logits, meta["y"], weights)["nll"] else 1.0
                values = {**meta, "logits": logits, "weights": weights,
                          "temperature": np.array(temperature), **generated}
                if raw is not None:
                    values["raw_probabilities"] = raw
                np.savez_compressed(folder / "cal.npz", **values)
                shutil.copyfile(base / selected["path"], folder / Path(selected["path"]).name)
                result = {"status": "source_verified", "seed": selected["seed"],
                          "source_temperature": temperature, "rows": len(x),
                          "contract": _verification_contract(family, record, scale),
                          "semantic_variant_vectors": variant_vectors,
                          "scope": "Selected checkpoint loaded twice and evaluated only on source calibration observations; no target access"}
                with (folder / "result.json").open("x", encoding="utf-8") as stream:
                    json.dump(result, stream, indent=2, allow_nan=False)
            except Exception as exc:
                with (folder / "failure.json").open("x", encoding="utf-8") as stream:
                    json.dump({"type": type(exc).__name__, "message": str(exc),
                               "scope": "Source verification failure, not a target diagnostic result"}, stream, indent=2)
                raise
    release_target(cfg, root, run)
    print("Every source-selected checkpoint and source temperature verified; external waveform access is now eligible")


def evaluate_selected(cfg,run,device):
    import torch
    from jev_learning import exported_role, score_logits, score_probabilities
    family=run/"fits"/cfg["name"]
    selection=json.loads((family/"selection.json").read_text())
    scale=json.loads((run/"source_scale.json").read_text())
    for selected in selection["models"]:
        out=run/"evaluations"/cfg["name"]/str(selected["seed"])
        if (out/"metrics.json").exists():
            for role in ("cal","test","external"):
                if not (out/f"{role}.npz").is_file(): raise FileNotFoundError("Incomplete saved evaluation")
                if cfg["name"] == "semantic":
                    for variant in cfg["evaluation"]["semantic_variants"]:
                        if not (out / f"{role}.variant_{variant}.npz").is_file():
                            raise FileNotFoundError("Incomplete saved semantic-variant evaluation")
            continue
        out.mkdir(parents=True,exist_ok=False)
        verified, result = _checked_verification(cfg, run, selected, selection, scale)
        temperature = result["source_temperature"]
        model, predict = _load_predictor(cfg, family, selected, scale, device)
        rows=[]
        for role in ("cal","test","external"):
            if role == "cal":
                with np.load(verified / "cal.npz", allow_pickle=False) as saved:
                    values = dict(saved)
                logits, raw, meta, w = values["logits"], values.get("raw_probabilities"), values, values["weights"]
            else:
                x,meta,w=exported_role(run,role);logits,raw=predict(x)
                values={**meta,"logits":logits,"weights":w,"temperature":np.array(temperature)}
                if raw is not None: values["raw_probabilities"]=raw
                if cfg["model"].get("readout") == "token":
                    values.update(_generated_predictions(model, x, scale, device, cfg["fit"]["batch_size"]))
            np.savez_compressed(out/f"{role}.npz",**values)
            if cfg["name"] == "semantic":
                from jev_learning import infer_semantic_variant
                for variant, descriptions in cfg["evaluation"]["semantic_variants"].items():
                    destination = out / f"{role}.variant_{variant}.npz"
                    if role == "cal":
                        shutil.copyfile(verified / f"cal.variant_{variant}.npz", destination)
                    else:
                        variant_logits, seconds = infer_semantic_variant(model, x, scale, device,
                            cfg["fit"]["batch_size"], descriptions, cfg["model"]["snapshot"],
                            candidate_vectors=torch.load(verified / f"variant_{variant}.pt", map_location="cpu", weights_only=True))
                        np.savez_compressed(destination, **meta, logits=variant_logits, weights=w,
                            temperature=np.array(1.0), variant=np.array(variant),
                            descriptions=np.asarray(descriptions), candidate_compile_seconds=np.array(seconds))
            rows.append({"role":role,"raw":score_probabilities(raw,meta["y"],w,floor=cfg["evaluation"]["probability_floor"]) if raw is not None else score_logits(logits,meta["y"],w),
                         "calibrated":score_logits(logits/temperature,meta["y"],w)})
        (out/"metrics.json").write_text(json.dumps({"seed":selected["seed"],"source_temperature":temperature,"rows":rows},indent=2))
        del predict, model
        if device is not None: torch.cuda.empty_cache()


def summarize_saved(cfg,root,run):
    """Recompute primary metrics from native outputs; no model calls."""
    from jev_reporting import summarize_saved_predictions
    summarize_saved_predictions(cfg, root, run)
