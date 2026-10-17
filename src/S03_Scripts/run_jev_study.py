"""Run the frozen PU study through PHMFactory and the sole Jev-PHM implementation.

Pilot stages are explicit: prepare, export, select, embed, evaluate. Approved P1
operations use the semantic subcommand. Large arrays stay in
the ignored run directory. This driver never implements a loader or a model.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "S01_Package"))
import jev_phm as jev

FACTORY_ROOT = ROOT / "external" / "PHM-Vibench"
DEFAULT_DATA_ROOT = Path("/home/user/data/PHMbenchdata/PHM-Vibench")
LABEL_MAP = {"0.0": "healthy", "1.0": "inner-race-dominant", "2.0": "outer-race-dominant"}


def resolve_window_plan(lengths: Mapping[str, int], window_size: int,
                        max_stride: int | None = None) -> dict:
    """Choose the upstream evenly-spaced count needed to cover every recording."""
    if not lengths or isinstance(window_size, bool) or int(window_size) != window_size or window_size < 2:
        raise ValueError("Need recording lengths and an integer window size >= 2")
    stride = window_size if max_stride is None else max_stride
    if isinstance(stride, bool) or int(stride) != stride or not 1 <= stride <= window_size:
        raise ValueError("Coverage maximum stride must be an integer in [1, window_size]")
    values = np.asarray(list(lengths.values()))
    if not np.issubdtype(values.dtype, np.integer) or np.any(values < window_size):
        raise ValueError("Every actual recording must be at least as long as the window")
    count = 1 + math.ceil((int(values.max()) - window_size) / stride)
    if count > int(values.min()) - window_size + 1:
        raise ValueError("Common evenly-spaced count would duplicate window starts in a shorter recording")
    return {"window_size": int(window_size), "max_stride": int(stride),
            "num_window": count, "min_length": int(values.min()), "max_length": int(values.max())}


def validate_export_coverage(data: dict, lengths: Mapping[str, int], plan: dict) -> dict:
    """Check upstream intervals, without reimplementing upstream sampling."""
    ids = np.asarray(data["file_id"]).astype(str)
    starts, stops = np.asarray(data["window_start"]), np.asarray(data["window_stop"])
    if starts.shape != ids.shape or stops.shape != ids.shape or ids.ndim != 1:
        raise ValueError("Window intervals must align with file IDs")
    if not np.issubdtype(starts.dtype, np.integer) or not np.issubdtype(stops.dtype, np.integer):
        raise ValueError("Window intervals must use integer sample positions")
    if set(ids) != set(lengths):
        raise ValueError("Exported recording population differs from the frozen inventory")
    gap_min, gap_max, min_overlap, max_overlap = [], [], [], []
    for file_id in sorted(lengths):
        mask = ids == file_id
        begin, end = starts[mask], stops[mask]
        order = np.argsort(begin, kind="stable")
        begin, end = begin[order], end[order]
        expected = 1 if lengths[file_id] == plan["window_size"] else plan["num_window"]
        if len(begin) != expected or begin[0] != 0 or end[-1] != lengths[file_id]:
            raise ValueError(f"Incomplete first/last coverage or incorrect window count: {file_id}")
        if np.any(end - begin != plan["window_size"]):
            raise ValueError(f"Window width differs from the frozen plan: {file_id}")
        gaps = np.diff(begin)
        if np.any(gaps <= 0) or np.any(gaps > plan["max_stride"]):
            raise ValueError(f"Duplicate starts or an uncovered interval: {file_id}")
        if len(gaps):
            gap_min.append(int(gaps.min())); gap_max.append(int(gaps.max()))
            min_overlap.append(int(plan["window_size"] - gaps.max()))
            max_overlap.append(int(plan["window_size"] - gaps.min()))
    return {"files": len(lengths), "windows": len(ids), "coverage_fraction": 1.0,
            "minimum_actual_stride": min(gap_min) if gap_min else None,
            "maximum_actual_stride": max(gap_max) if gap_max else None,
            "minimum_overlap_samples": min(min_overlap) if min_overlap else None,
            "maximum_overlap_samples": max(max_overlap) if max_overlap else None}


def assemble_split_roles(unit_labels: Mapping[str, int], seed: int = 20260922) -> dict[str, dict[str, str]]:
    """Freeze one primary split and six predetermined healthy-bearing rotations."""
    units = {int(k): sorted((u for u, y in unit_labels.items() if y == k),
                            key=lambda u: (u.rsplit(" ", 1)[-1], u)) for k in (0, 1, 2)}
    if set(unit_labels.values()) != {0, 1, 2} or [len(units[k]) for k in (0, 1, 2)] != [6, 13, 13]:
        raise ValueError("The declared PU protocol requires exactly 6/13/13 physical bearings")
    rng = np.random.default_rng(seed)
    shuffled = {k: list(rng.permutation(units[k])) for k in (0, 1, 2)}
    primary = {}
    for k, counts in ((0, (2, 1, 1, 2)), (1, (7, 1, 2, 3)), (2, (7, 1, 2, 3))):
        offset = 0
        for role, count in zip(jev.ROLES, counts):
            primary.update({str(u): role for u in shuffled[k][offset:offset + count]})
            offset += count
    result = {"primary": primary}
    buckets = {k: np.array_split(shuffled[k], 6) for k in (1, 2)}
    for fold in range(6):
        roles = {u: "train" for u in unit_labels}
        health = shuffled[0]
        roles[str(health[fold])] = "test"
        roles[str(health[(fold + 1) % 6])] = "cal"
        roles[str(health[(fold + 2) % 6])] = "tune"
        for k in (1, 2):
            test = set(map(str, buckets[k][fold]))
            remaining = [str(u) for u in shuffled[k] if u not in test]
            roles.update({u: "test" for u in test})
            roles.update({u: "cal" for u in remaining[:2]})
            roles[remaining[2]] = "tune"
        result[f"rotation_{fold + 1}"] = roles
    return result


def assign_roles(data: dict, unit_roles: Mapping[str, str]) -> dict:
    """Reuse role-independent observations under another previously frozen split."""
    if set(map(str, np.unique(data["unit"]))) != set(unit_roles):
        raise ValueError("Protocol bearings differ from the exported population")
    result = dict(data)
    result["role"] = np.asarray([unit_roles[str(u)] for u in data["unit"]])
    origin = json.loads(str(data["provenance"])) if "provenance" in data else {}
    result["provenance"] = np.asarray(json.dumps({**origin, "evaluation_unit_roles": dict(unit_roles)}))
    jev.check_data(result)
    return result


def _write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def _upstream() -> None:
    sha = subprocess.check_output(["git", "-C", str(FACTORY_ROOT), "rev-parse", "HEAD"], text=True).strip()
    if sha != jev.PHMFACTORY_REVISION:
        raise ValueError("PHMFactory revision differs from the inspected contract")
    sys.path.insert(0, str(FACTORY_ROOT))
    import src.data_factory as source
    if not Path(source.__file__).resolve().is_relative_to(FACTORY_ROOT):
        raise ValueError("Wrong src.data_factory import; use the pinned checkout")


def prepare(args) -> None:
    _upstream()
    from phmfactory.config import analyze_config
    from src.data_factory.data_utils import read_metadata_table
    from src.data_factory.splitting import _sha256_file
    import h5py

    output = Path(args.run_root).resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "inputs"
    inputs.mkdir()
    for name in ("cache", "exports", "embeddings", "evaluations"):
        (output / name).mkdir()
    probe = _load_json(Path(args.probe_config))
    data_root = Path(args.data_root).resolve()
    metadata_path = data_root / "metadata.xlsx"
    frame = read_metadata_table(metadata_path)
    selected = frame[frame["Dataset_id"] == 20].copy()
    if len(selected) != 2559 or set(selected["Name"]) != {"RM_027_PU"}:
        raise ValueError("Actual PU metadata population differs from 2,559 declared records")
    if selected["Id"].duplicated().any() or selected["Label_Description"].isna().any():
        raise ValueError("Duplicate IDs or absent physical-bearing identity")
    unit_labels = {}
    for unit, group in selected.groupby("Label_Description"):
        labels = set(group["Label"].astype(float))
        if len(labels) != 1 or not labels.issubset({0., 1., 2.}):
            raise ValueError(f"Ambiguous declared bearing class: {unit}")
        unit_labels[str(unit)] = int(next(iter(labels)))
    protocols = assemble_split_roles(unit_labels, probe["split_seed"])
    inventory, lengths = [], {}
    with h5py.File(data_root / "RM_027_PU.h5", "r") as source:
        if set(source.keys()) != set(selected["Id"].astype(str)):
            raise ValueError("H5 and selected metadata contain different recording IDs")
        for row in selected.sort_values("Id").to_dict("records"):
            file_id = str(row["Id"])
            shape = source[file_id].shape
            if len(shape) != 3 or shape[1:] != (3, 1):
                raise ValueError(f"Unexpected upstream PU channel layout: {file_id}: {shape}")
            lengths[file_id] = int(shape[0])
            inventory.append({"file_id": file_id, "unit_id": str(row["Label_Description"]),
                              "raw_label": float(row["Label"]), "domain_id": int(row["Domain_id"]),
                              "file": str(row["File"]), "sampling_rate_hz": int(row["Sample_rate"]),
                              "actual_length": int(shape[0]), "metadata_length": int(row["Sample_lenth"]),
                              "channels": int(shape[1])})
    if set(row["sampling_rate_hz"] for row in inventory) != {64000}:
        raise ValueError("PU sampling rates differ from the declared 64 kHz")
    window_sizes = args.window_sizes or probe["window_candidates"]
    stride = args.max_stride if args.max_stride is not None else probe["coverage_max_stride"]
    if len(set(window_sizes)) != len(window_sizes):
        raise ValueError("Window candidate lengths must be unique")
    if probe["training_record_weight"] != 64.:
        raise ValueError("This frozen study fixes the source-record training weight at 64")
    probe.update({"window_candidates": window_sizes, "coverage_max_stride": stride})
    plans = {str(w): resolve_window_plan(lengths, w, stride) for w in window_sizes}
    _write_json(inputs / "inventory.json", inventory)
    _write_json(inputs / "window_plans.json", plans)
    _write_json(inputs / "labels.json", LABEL_MAP)
    _write_json(inputs / "unit_roles.json", protocols)
    _write_json(inputs / "probe.json", probe)
    native_metadata_checksum = _sha256_file(metadata_path)
    for protocol, roles in protocols.items():
        folder = inputs / protocol
        folder.mkdir()
        with (folder / "units.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["file_id", "unit_id", "role"])
            writer.writeheader()
            writer.writerows({"file_id": row["file_id"], "unit_id": row["unit_id"],
                              "role": roles[row["unit_id"]]} for row in inventory)
        partitions = {name: {"ids": [row["file_id"] for row in inventory if
                                            ("val" if roles[row["unit_id"]] in {"tune", "cal"} else roles[row["unit_id"]]) == name]}
                      for name in ("train", "val", "test")}
        _write_json(folder / "native_split.json", {"schema_version": 1, "group_key": "Label_Description",
                    "stratify_key": "Label", "metadata_file_sha256": native_metadata_checksum,
                    "partitions": partitions})
    analysis = analyze_config(Path(args.factory_config).resolve(), override_values=[
        f"data.data_dir={data_root}", f"data.cache_dir={output / 'cache'}",
        f"data.split.manifest_path={inputs / 'primary' / 'native_split.json'}",
        f"environment.output_dir={output}"])
    config = analysis.runtime_config()
    _write_json(inputs / "factory_public_config.json", config)
    bridge = {"manifest_mode": "read_only", "partition_map": {"train": "train", "val": "val", "test": "test"}}
    _write_json(inputs / "native_split_adapter.json", {"runtime_additions": bridge,
        "reason": "Pinned public DataSplitConfig rejects native read_only/partition_map fields. "
                  "The inspected native constructor accepts these explicit frozen protocol fields. "
                  "Public analyze_config validates the complete base configuration first; all additions are serialized before construction.",
        "metadata_file_sha256": "Mandatory native manifest field computed solely by upstream _sha256_file."})
    _write_json(inputs / "run_spec.json", {"data_root": str(data_root), "factory_root": str(FACTORY_ROOT),
        "phmfactory_revision": jev.PHMFACTORY_REVISION, "qwen_revision": jev.QWEN_REVISION,
        "gpu_uuid": jev.DEFAULT_GPU_UUID, "forbidden_physical_gpu": 2,
        "window_candidates": window_sizes, "coverage_max_stride": stride, "channel": 2,
        "signal_key": "x", "label_key": "y", "training_record_weight": 64.,
        "code_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "command": sys.argv, "python": sys.version, "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    environment = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (inputs / "environment.txt").write_text(environment, encoding="utf-8")
    print(json.dumps({"run_root": str(output), "records": len(inventory), "bearings": len(unit_labels),
                      "window_plans": plans, "protocols": list(protocols)}, indent=2), flush=True)


def _ensure_cache(run_root: Path, data_root: Path) -> None:
    source, target = data_root / "RM_027_PU.h5", run_root / "cache" / "RM_027_PU.h5"
    if target.exists():
        if target.is_symlink() or target.stat().st_ino == source.stat().st_ino:
            raise ValueError("Writable factory cache must be an independent file copy")
        if target.stat().st_size != source.stat().st_size:
            raise ValueError("Incomplete existing PU cache copy; retain failed run and use a fresh run directory")
        return
    temporary = target.with_suffix(".copying")
    if temporary.exists():
        raise FileExistsError("Incomplete cache copy retained; use a fresh run directory")
    print(f"Copy immutable PU source to independent cache: {source} -> {target}", flush=True)
    shutil.copy2(source, temporary)
    temporary.rename(target)


def export(args) -> None:
    _upstream()
    from src.data_factory import build_data
    run_root = Path(args.run_root).resolve()
    spec = _load_json(run_root / "inputs" / "run_spec.json")
    plans = _load_json(run_root / "inputs" / "window_plans.json")
    inventory = _load_json(run_root / "inputs" / "inventory.json")
    lengths = {row["file_id"]: row["actual_length"] for row in inventory}
    _ensure_cache(run_root, Path(spec["data_root"]))
    if args.normalization == "per_window_standardization":
        selection = _load_json(run_root / "inputs" / "selection.json")
        selected = [str(selection["primary"]["window_size"])]
    else:
        selected = list(plans) if not args.window_size else [str(args.window_size)]
    for width in selected:
        plan = plans[width]
        stem = f"w{width}_{'raw' if args.normalization == 'none' else 'normalized'}"
        destination = run_root / "exports" / f"{stem}.npz"
        if destination.exists():
            raise FileExistsError(f"Do not overwrite an export: {destination}")
        config = _load_json(run_root / "inputs" / "factory_public_config.json")
        config["data"].update({"window_size": plan["window_size"], "num_window": plan["num_window"],
                               "normalization": args.normalization})
        config["data"]["split"].update(_load_json(run_root / "inputs" / "native_split_adapter.json")["runtime_additions"])
        _write_json(run_root / "exports" / f"{stem}_factory_runtime.json", config)
        began = time.perf_counter()
        # Native split fractions/partition_map are mappings, not namespaces.
        factory = build_data(SimpleNamespace(**config["data"]), SimpleNamespace(**config["task"]))
        data = jev.export_factory(factory, run_root / "inputs" / "primary" / "units.csv", LABEL_MAP,
                                  spec["signal_key"], spec["label_key"], channel=spec["channel"])
        coverage = validate_export_coverage(data, lengths, plan)
        data["provenance"] = np.asarray(json.dumps({"phmfactory_commit": jev.PHMFACTORY_REVISION,
            "factory_config": config, "window_plan": plan, "channel": spec["channel"],
            "normalization": args.normalization, "original_split": "primary",
            "input_inventory": str(run_root / "inputs" / "inventory.json")}, allow_nan=False))
        with destination.open("xb") as stream:
            np.savez_compressed(stream, **data)
        coverage.update({"seconds": time.perf_counter() - began,
            "features": {name: {"mean": float(data["features"][:, k].mean()),
                                 "std": float(data["features"][:, k].std()),
                                 "minimum": float(data["features"][:, k].min()),
                                 "maximum": float(data["features"][:, k].max())}
                         for k, name in enumerate(jev.FEATURES)}})
        _write_json(run_root / "exports" / f"{stem}_coverage.json", coverage)
        print(json.dumps({"export": str(destination), **coverage}, indent=2), flush=True)
        del factory, data


def select(args) -> None:
    run_root = Path(args.run_root).resolve()
    probe = _load_json(run_root / "inputs" / "probe.json")
    protocols = _load_json(run_root / "inputs" / "unit_roles.json")
    plans = _load_json(run_root / "inputs" / "window_plans.json")
    selection, table = {}, []
    for protocol, roles in protocols.items():
        rows = []
        for width, plan in plans.items():
            data = assign_roles(_load_npz(run_root / "exports" / f"w{width}_raw.npz"), roles)
            _, _, tuning = jev.select_head(data["features"], data["y"], data["role"], data["unit"],
                                          probe["C_grid"], data["sample_weight"])
            for row in tuning:
                rows.append({"protocol": protocol, "window_size": int(width),
                             "num_window": plan["num_window"], "C": row["C"], "tune_nll": row["nll"]})
            del data
        chosen = min(rows, key=lambda r: (r["tune_nll"], r["num_window"], r["C"]))
        selection[protocol] = {k: chosen[k] for k in ("window_size", "num_window", "C", "tune_nll")}
        table.extend(rows)
        print(f"Frozen {protocol}: {selection[protocol]}", flush=True)
    _write_json(run_root / "inputs" / "selection.json", selection)
    with (run_root / "inputs" / "window_tuning.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)


def embed(args) -> None:
    run_root = Path(args.run_root).resolve()
    selection = _load_json(run_root / "inputs" / "selection.json")
    if args.normalization == "per_window_standardization":
        widths = [selection["primary"]["window_size"]]
        names = ("legacy",)
    else:
        widths = sorted({row["window_size"] for row in selection.values()})
        names = ("legacy", "flat_a", "flat_b", "presence", "identity")
    if args.window_size is not None:
        if args.window_size not in widths:
            raise ValueError("Embedding window was not selected by a frozen protocol")
        widths = [args.window_size]
    if args.queries:
        names = tuple(args.queries)
    if len(widths) > 1:
        # Each model load gets a fresh process: the core deliberately rejects
        # changing device visibility after CUDA has already been initialized.
        for width in widths:
            command = [sys.executable, str(Path(__file__).resolve()), "embed", "--run-root", str(run_root),
                       "--window-size", str(width), "--normalization", args.normalization,
                       "--device", args.device, "--dtype", args.dtype, "--batch-size", str(args.batch_size),
                       "--gpu-uuid", args.gpu_uuid, "--queries", *names]
            subprocess.run(command, check=True)
        return
    for width in widths:
        kind = "raw" if args.normalization == "none" else "normalized"
        data = _load_npz(run_root / "exports" / f"w{width}_{kind}.npz")
        suffix = "_".join(names)
        destination = run_root / "embeddings" / f"w{width}_{kind}_{suffix}.npz"
        if destination.exists():
            raise FileExistsError(f"Do not overwrite embeddings: {destination}")
        result = jev.embed_queries(data["features"], data["labels"], jev.QWEN_REVISION,
            device=args.device, batch_size=args.batch_size, query_names=names,
            healthy_index=0, dtype=args.dtype, gpu_uuid=args.gpu_uuid)
        metadata = result.pop("metadata")
        result["record"] = data["record"]
        result["provenance"] = np.asarray(json.dumps({**metadata,
            "observation_provenance": json.loads(str(data["provenance"]))}, allow_nan=False))
        with destination.open("xb") as stream:
            np.savez_compressed(stream, **result)
        _write_json(destination.with_suffix(".json"), metadata)
        print(f"Saved {destination}", flush=True)


def _embedding_archive(run_root: Path, width: int, kind: str, e2: bool) -> dict:
    pattern = f"w{width}_{kind}_*.npz"
    needed = {"hidden_flat_a", "hidden_flat_b", "hidden_presence", "hidden_identity"} if e2 else {"hidden"}
    archives = sorted((run_root / "embeddings").glob(pattern))
    result, sources = {}, []
    for path in archives:
        archive = _load_npz(path)
        if not needed.intersection(archive):
            continue
        if "record" in result and not np.array_equal(result["record"], archive["record"]):
            raise ValueError("Embedding archives contain different observation rows")
        for key in set(result).intersection(archive) - {"provenance"}:
            if not np.array_equal(result[key], archive[key]):
                raise ValueError(f"Conflicting saved embedding field: {key}")
        result.update(archive)
        sources.append({"path": str(path), "provenance": json.loads(str(archive["provenance"]))})
    if not needed.issubset(result):
        raise FileNotFoundError(f"Missing frozen embeddings for {width}/{kind}: {sorted(needed - set(result))}")
    result["provenance"] = np.asarray(json.dumps({"archives": sources}, allow_nan=False))
    return result


def evaluate(args) -> None:
    if args.head_workers < 1:
        raise ValueError("Head worker count must be positive")
    run_root = Path(args.run_root).resolve()
    probe = _load_json(run_root / "inputs" / "probe.json")
    selection = _load_json(run_root / "inputs" / "selection.json")
    protocols = _load_json(run_root / "inputs" / "unit_roles.json")
    if set(selection) != set(protocols):
        raise ValueError("Freeze window selection for all protocols before inspecting any test result")
    names = args.protocols or list(protocols)
    for protocol in names:
        if protocol not in protocols:
            raise ValueError(f"Unknown protocol: {protocol}")
        width = selection[protocol]["window_size"]
        kind = "normalized" if args.normalization == "per_window_standardization" else "raw"
        if kind == "normalized" and protocol != "primary":
            raise ValueError("The predeclared normalization E1 comparison is primary-split only")
        data = assign_roles(_load_npz(run_root / "exports" / f"w{width}_{kind}.npz"), protocols[protocol])
        embeddings = _embedding_archive(run_root, width, kind, e2=args.experiment == "e2")
        folder = run_root / "evaluations" / f"{protocol}_{kind}_{args.experiment}"
        config = {**probe, "healthy_index": 0, "protocol": protocol,
                  "window_selection": selection[protocol], "normalization": kind,
                  "head_workers": args.head_workers}
        runner = jev.run_e2 if args.experiment == "e2" else jev.run
        runner(data, embeddings, config, folder)
        print(f"Saved {folder}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    from jev_execution import add_subparser
    add_subparser(sub)
    p = sub.add_parser("prepare")
    p.add_argument("--run-root", required=True)
    p.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    p.add_argument("--factory-config", default=str(ROOT / "src/S02_Configs/jev_factory.yaml"))
    p.add_argument("--probe-config", default=str(ROOT / "src/S02_Configs/jev_probe.json"))
    p.add_argument("--window-sizes", type=int, nargs="+")
    p.add_argument("--max-stride", type=int)
    p = sub.add_parser("export")
    p.add_argument("--run-root", required=True)
    p.add_argument("--window-size", type=int)
    p.add_argument("--normalization", choices=("none", "per_window_standardization"), default="none")
    p = sub.add_parser("select")
    p.add_argument("--run-root", required=True)
    p = sub.add_parser("embed")
    p.add_argument("--run-root", required=True)
    p.add_argument("--window-size", type=int)
    p.add_argument("--normalization", choices=("none", "per_window_standardization"), default="none")
    p.add_argument("--queries", nargs="+", choices=("legacy", "flat_a", "flat_b", "presence", "identity"))
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--gpu-uuid", default=jev.DEFAULT_GPU_UUID)
    p = sub.add_parser("evaluate")
    p.add_argument("--run-root", required=True)
    p.add_argument("--experiment", choices=("e1", "e2"), required=True)
    p.add_argument("--normalization", choices=("none", "per_window_standardization"), default="none")
    p.add_argument("--protocols", nargs="+")
    p.add_argument("--head-workers", type=int, default=1,
                   help="Independent C candidates fitted in isolated CPU worker processes")
    args = parser.parse_args()
    if hasattr(args, "handler"):
        try:
            args.handler(args, ROOT, _upstream)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            if args.stage == "check":
                parser.exit(2, f"{exc}\n")
            raise
    else:
        globals()[args.command](args)


if __name__ == "__main__":
    main()
