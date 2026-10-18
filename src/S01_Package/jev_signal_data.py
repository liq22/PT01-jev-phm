"""Explicit PHMFactory extension for qualified, common-rate signal windows.

No waveform reader, H5 loader, split sampler or raw data repair is defined here. Raw arrays
and grouped partitions are owned by the pinned PHMFactory. Resampling is an
explicit project transform, not an undocumented upstream YAML capability.
"""
from __future__ import annotations

import csv
import json
import math
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
from time import perf_counter
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.signal import resample_poly

FACTORY_REVISION = "cdc0669167ca25ac848e34c880875ca39c6aed89"
DATASETS = {"RM_027_PU", "RM_007_MFPT"}
RAW_TO_CANONICAL = {"RM_027_PU": {0: 0, 1: 1, 2: 2},
                    "RM_007_MFPT": {0: 0, 1: 2, 2: 1}}


def _integer_measurement(value: object, name: str, *, minimum: int) -> int:
    number = float(value)
    if not math.isfinite(number) or not number.is_integer() or number < minimum:
        raise ValueError(f"{name} must be a finite integer >= {minimum}; no rounding")
    return int(number)


def _measurement_binding(sample_rate_hz: object, channel_index: object,
                         scale_to_g: object) -> tuple[int, int, float]:
    rate = _integer_measurement(sample_rate_hz, "sample_rate_hz", minimum=1)
    channel = _integer_measurement(channel_index, "channel_index", minimum=0)
    scale = float(scale_to_g)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale_to_g must be finite and positive")
    return rate, channel, scale


def common_rate_signal(raw, *, sample_rate_hz, channel_index, scale_to_g, target_rate_hz, beta):
    """Declared-channel conversion and polyphase FIR before window generation."""
    sample_rate_hz, channel_index, scale_to_g = _measurement_binding(
        sample_rate_hz, channel_index, scale_to_g)
    target_rate_hz = _integer_measurement(target_rate_hz, "target_rate_hz", minimum=1)
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("Invalid documented FIR parameter")
    x = np.asarray(raw)
    if x.ndim == 3 and x.shape[2] == 1:
        x = x[:, :, 0]
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or not 0 <= channel_index < x.shape[1]:
        raise ValueError("Declared channel is inconsistent with the native reader/cache shape")
    if not np.isfinite(x).all():
        raise ValueError("Raw observation contains nonfinite samples")
    x = x[:, channel_index].astype(np.float64) * scale_to_g
    g = math.gcd(int(sample_rate_hz), int(target_rate_hz))
    y = resample_poly(x, int(target_rate_hz)//g, int(sample_rate_hz)//g,
                     window=("kaiser", beta), padtype="constant")
    return y.astype(np.float32)[:, None]


def _pu_roles(root: Path, config: dict) -> list[dict]:
    with (root / config["pu_roles"]).open(encoding="utf-8", newline="") as handle:
        roles = list(csv.DictReader(handle))
    if (len(roles) != 29 or len({row["physical_unit"] for row in roles}) != 29
            or sum(int(row["source_record_count"]) for row in roles) != 2319
            or Counter(int(row["class_index"]) for row in roles) != {0: 6, 1: 11, 2: 12}):
        raise ValueError("The approved PU role table must describe exactly 29 units and 2319 records")
    for row in roles:
        raw = _integer_measurement(row["raw_label"], "PU raw_label", minimum=0)
        if RAW_TO_CANONICAL["RM_027_PU"].get(raw) != int(row["class_index"]):
            raise ValueError("PU role table contradicts the native label mapping")
    return roles


def _worksheet_rows(stream):
    """Yield raw row bytes; excluded rows are never passed to an XML decoder."""
    buffer = b""
    while True:
        chunk = stream.read(65536)
        if chunk:
            buffer += chunk
        while True:
            start = re.search(rb"<row\b", buffer)
            if start is None:
                buffer = buffer[-4:]
                break
            buffer = buffer[start.start():]
            opening_end = buffer.find(b">")
            if opening_end < 0:
                break
            if buffer[:opening_end].rstrip().endswith(b"/"):
                end = opening_end + 1
            else:
                closing = buffer.find(b"</row>", opening_end)
                if closing < 0:
                    break
                end = closing + len(b"</row>")
            yield buffer[:end]
            buffer = buffer[end:]
        if not chunk:
            if re.search(rb"<row\b", buffer):
                raise ValueError("Unterminated row in the native metadata worksheet")
            return


def _decode_metadata_row(raw: bytes) -> dict[str, object]:
    row = ET.fromstring(raw)
    result = {}
    for cell in row:
        if cell.tag.rsplit("}", 1)[-1] != "c":
            continue
        reference = re.fullmatch(r"([A-Z]+)[0-9]+", cell.attrib.get("r", ""))
        if reference is None:
            raise ValueError("Native metadata cell lacks an explicit column reference")
        if cell.attrib.get("t") == "s":
            raise ValueError("Shared-string metadata is outside the inspected inline-string workbook contract")
        if cell.attrib.get("t") == "inlineStr":
            value = "".join(node.text or "" for node in cell.iter()
                            if node.tag.rsplit("}", 1)[-1] == "t")
        else:
            value = next((node.text for node in cell if node.tag.rsplit("}", 1)[-1] == "v"), None)
            if value is not None and cell.attrib.get("t") in {None, "n"}:
                number = float(value)
                value = int(number) if math.isfinite(number) and number.is_integer() else number
        result[reference.group(1)] = value
    return result


def _read_approved_workbook(path: Path):
    import pandas as pd
    with zipfile.ZipFile(path) as workbook:
        sheets = [name for name in workbook.namelist()
                  if re.fullmatch(r"xl/worksheets/sheet[0-9]+\.xml", name)]
        if sheets != ["xl/worksheets/sheet1.xml"] or "xl/sharedStrings.xml" in workbook.namelist():
            raise ValueError("Expected the inspected single-sheet inline-string native workbook")
        with workbook.open(sheets[0]) as worksheet:
            raw_rows = _worksheet_rows(worksheet)
            header = _decode_metadata_row(next(raw_rows, b"<row/>"))
            names = list(header.values())
            if "Name" not in names or len(set(names)) != len(names):
                raise ValueError("Native metadata header is missing or ambiguous")
            name_column = next(column for column, value in header.items() if value == "Name")
            name_cell = re.compile(rb'<c\b(?=[^>]*\br="' + name_column.encode("ascii")
                                   + rb'[0-9]+\")[^>]*>.*?</c>', re.DOTALL)
            approved = []
            for raw in raw_rows:
                # Routing inspects only raw bytes of the Name cell. In
                # particular, malformed excluded cells are never XML-parsed.
                cell = name_cell.search(raw)
                if cell is None or not any(b"<t>" + name.encode("ascii") + b"</t>" in cell.group()
                                           for name in DATASETS):
                    continue
                values = _decode_metadata_row(raw)
                item = {name: values.get(column) for column, name in header.items()}
                if item["Name"] not in DATASETS:
                    raise ValueError("Ambiguous native metadata Name routing")
                approved.append(item)
    return pd.DataFrame(approved, columns=names)


def read_approved_metadata(root: Path, config: dict):
    """Read native XLSX or an explicit UTF-8 CSV projection without raw-data writes.

    This metadata-only route does not load signals or qualify physical
    measurements; waveform readers, construction and splitting remain upstream.
    """
    roles = _pu_roles(root, config)
    path = Path(config["root"]) / config["metadata"]
    if path.suffix.lower() == ".xlsx":
        frame = _read_approved_workbook(path)
    elif path.suffix.lower() == ".csv":
        import pandas as pd
        frame = pd.read_csv(path, encoding="utf-8-sig")
        if "Name" not in frame or not frame["Name"].isin(DATASETS).all():
            raise ValueError("P1 metadata CSV contains an excluded dataset")
    else:
        raise ValueError("P1 metadata must be the native .xlsx or a PU/MFPT-only .csv projection")
    needed = {"Id", "Name", "File", "Label", "Label_Description"}
    if not needed.issubset(frame.columns):
        raise ValueError(f"Native metadata projection needs {sorted(needed)}")
    approved_units = {row["physical_unit"] for row in roles}
    keep = ((frame["Name"].eq("RM_027_PU") & frame["Label_Description"].isin(approved_units))
            | (frame["Name"].eq("RM_007_MFPT") & frame["Label"].isin([0, 1, 2])))
    selected = frame.loc[keep].copy()
    if selected["Id"].isna().any() or selected["Id"].duplicated().any():
        raise ValueError("Native P1 Id population is missing or ambiguous")
    selected["Id"] = [_integer_measurement(value, "metadata Id", minimum=0) for value in selected["Id"]]
    if selected["Id"].duplicated().any():
        raise ValueError("Native P1 Id population is missing or ambiguous after integer conversion")
    if Counter(selected["Name"]) != {"RM_027_PU": 2319, "RM_007_MFPT": 20}:
        raise ValueError("Native P1 population must contain 2319 approved PU and 20 known-label MFPT records")
    pu = selected.loc[selected["Name"].eq("RM_027_PU")]
    for role in roles:
        unit_rows = pu.loc[pu["Label_Description"].eq(role["physical_unit"])]
        if (len(unit_rows) != int(role["source_record_count"])
                or not unit_rows["Label"].eq(float(role["raw_label"])).all()):
            raise ValueError(f"Native PU population contradicts the approved role table: {role['physical_unit']}")
    mfpt = selected.loc[selected["Name"].eq("RM_007_MFPT")]
    if Counter(mfpt["Label"]) != {0: 3, 1: 10, 2: 7}:
        raise ValueError("Native MFPT known-label population must be baseline=3, outer=10, inner=7")
    descriptions = {0: "baseline", 1: "outer race fault", 2: "inner race fault"}
    if any(str(row["Label_Description"]) != descriptions[int(row["Label"])]
           for _, row in mfpt.iterrows()):
        raise ValueError("MFPT native label descriptions disagree with baseline/outer/inner label meanings")
    if (selected["File"].isna().any() or selected["File"].astype(str).str.strip().eq("").any()
            or selected.duplicated(["Name", "File"]).any()):
        raise ValueError("Native P1 original-file correspondence is missing or ambiguous")
    return selected


def summarize_native_metadata(frame) -> dict:
    def counts(values):
        def key(value):
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return "UNKNOWN"
            return str(int(value)) if isinstance(value, (int, float)) and float(value).is_integer() else str(value)
        return dict(Counter(key(value) for value in values))
    result = {"records": len(frame), "datasets": {},
              "basis": "native metadata projection only; physical-unit and measurement qualification not established"}
    for name in sorted(DATASETS):
        rows = frame.loc[frame["Name"].eq(name)]
        result["datasets"][name] = {
            "records": len(rows), "raw_to_canonical": {str(k): v for k, v in RAW_TO_CANONICAL[name].items()},
            "raw_label_counts": counts(rows["Label"]),
            "canonical_label_counts": counts(rows["Label"].map(RAW_TO_CANONICAL[name])),
            "label_description_counts": counts(rows["Label_Description"]),
            "nominal_sample_rate_counts": counts(rows["Sample_rate"]) if "Sample_rate" in rows else {},
            "metadata_channel_count_counts": counts(rows["Channel"]) if "Channel" in rows else {},
            "approved_physical_unit_count": rows["Label_Description"].nunique() if name == "RM_027_PU" else "UNKNOWN",
        }
    return result


def qualify_records(root: Path, config: dict) -> tuple[object, dict]:
    """Bind metadata to a completed G0 record map; never infer specimen identity."""
    path = config["qualification_file"]
    if not path:
        raise FileNotFoundError("G0 is incomplete: data.qualification_file is unbound")
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = root / path
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"Id", "Name", "File", "unit_id", "role", "class_index", "raw_label",
                "sample_rate_hz", "channel_index", "scale_to_g", "physical_source",
                "measurement_source", "license_source"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"G0 needs record bindings: {sorted(required)}")
    bindings, units = {}, {}
    for r in rows:
        if r["Name"] not in DATASETS or any(
                r[k] is None or r[k].strip().lower() in {"", "unknown", "tbd", "na", "null", "none", "nan"}
                for k in required):
            raise ValueError("Unqualified record or excluded dataset in G0")
        if r["Id"] in bindings:
            raise ValueError("Repeated metadata Id in G0")
        if r["role"] not in {"train", "tune", "cal", "test", "external"}:
            raise ValueError("Unknown source/evaluation role")
        if (r["Name"] == "RM_007_MFPT") != (r["role"] == "external"):
            raise ValueError("MFPT must be entirely external; source roles must belong to PU")
        r = dict(r)
        r["class_index"] = _integer_measurement(r["class_index"], "class_index", minimum=0)
        r["sample_rate_hz"], r["channel_index"], r["scale_to_g"] = _measurement_binding(
            r["sample_rate_hz"], r["channel_index"], r["scale_to_g"])
        if r["class_index"] not in range(3):
            raise ValueError("The approved closed task has exactly H/I/O classes")
        raw_label = _integer_measurement(r["raw_label"], "raw_label", minimum=0)
        if RAW_TO_CANONICAL[r["Name"]].get(raw_label) != r["class_index"]:
            raise ValueError(f"G0 contradicts the native raw-to-canonical label mapping: {r['Name']}")
        u = (r["Name"], r["unit_id"])
        description = (r["role"], r["class_index"])
        if u in units and units[u] != description:
            raise ValueError("A physical unit crosses roles or incompatible fault states")
        units[u] = description
        bindings[r["Id"]] = r
    counts = Counter(r["Name"] for r in bindings.values())
    if counts != {"RM_027_PU": 2319, "RM_007_MFPT": 20}:
        raise ValueError(f"Population differs from the approved record inventory: {counts}")
    frozen = _pu_roles(root, config)
    for f in frozen:
        matched = [r for r in bindings.values() if r["Name"] == "RM_027_PU" and r["unit_id"] == f["physical_unit"]]
        if len(matched) != int(f["source_record_count"]) or any(
                r["role"] != f["role"] or r["class_index"] != int(f["class_index"])
                or float(r["raw_label"]) != float(f["raw_label"]) for r in matched):
            raise ValueError(f"G0 contradicts the approved PU role map: {f['physical_unit']}")
    selected = read_approved_metadata(root, config)
    if (len(selected) != len(bindings) or selected["Id"].duplicated().any()
            or set(selected["Id"].astype(str)) != set(bindings)):
        raise ValueError("G0 and native metadata contain different Id populations")
    for _, item in selected.iterrows():
        r = bindings[str(item["Id"])]
        if str(item["Name"]) != r["Name"] or str(item["File"]) != r["File"] or float(item["Label"]) != float(r["raw_label"]):
            raise ValueError(f"Record correspondence changed: {item['Id']}")
        if r["Name"] == "RM_027_PU" and str(item["Label_Description"]) != r["unit_id"]:
            raise ValueError(f"Native PU physical identity differs from G0: {item['Id']}")
    return selected, bindings


def register_adapter() -> None:
    """Use the public registry; do not replace the existing Default_task adapter."""
    from src.data_factory import register_dataset_adapter, DATASET_ADAPTERS
    from src.data_factory.dataset_task.Default_dataset import Default_dataset

    class CommonRateDataset(Default_dataset):
        def __init__(self, data, metadata, args_data, args_task, mode="train"):
            key = next(iter(data))
            row = metadata[key]
            transform_start = perf_counter()
            transformed = common_rate_signal(data[key], sample_rate_hz=row["JevRate"],
                channel_index=int(row["JevChannel"]), scale_to_g=row["JevScale"],
                target_rate_hz=args_data.target_rate_hz, beta=args_data.resample_beta)
            resampling_seconds = perf_counter() - transform_start
            super().__init__({key: transformed}, metadata, args_data, args_task, mode)
            self.resampled_intervals = self.window_intervals
            retained = self.resampled_intervals[-1][1]
            self.record_lengths = {
                "input_samples": int(np.asarray(data[key]).shape[0]),
                "input_rate_hz": int(row["JevRate"]),
                "resampled_samples": int(transformed.shape[0]),
                "resampled_rate_hz": int(args_data.target_rate_hz),
                "retained_samples": int(retained),
                "discarded_tail_samples": int(transformed.shape[0] - retained),
                "discarded_tail_seconds": (transformed.shape[0] - retained) / args_data.target_rate_hz,
                "resampling_seconds": resampling_seconds,
                "resampling_scope": "whole native-cache record channel/scale/polyphase transform; excludes H5 IO",
            }
            # FIR support is not the stored sample interval. Do not mislabel it raw.
            self.window_intervals = None

    globals()["CommonRateDataset"] = CommonRateDataset
    key = ("Default_task", "jev_common_rate")
    if key not in DATASET_ADAPTERS:
        register_dataset_adapter(*key, module=__name__, attribute="CommonRateDataset")
    elif DATASET_ADAPTERS[key].module != __name__:
        raise ValueError("Dataset adapter name is already owned by another module")


def build_native_factory(root: Path, cfg: dict, run: Path, *, target: bool):
    from src.data_factory import build_data
    from src.data_factory.splitting import _sha256_file
    register_adapter()
    frame, bindings = qualify_records(root, cfg["data"])
    # Source preparation never opens the target H5. Target export retains real
    # PU train/validation partitions because the native factory requires them.
    names = DATASETS if target else {"RM_027_PU"}
    chosen = frame[frame["Name"].isin(names)].copy()
    if target:
        chosen = chosen[~chosen["Id"].astype(str).map(lambda k: bindings[k]["role"] == "test")].copy()
    for column, field in (("JevRate", "sample_rate_hz"), ("JevChannel", "channel_index"),
                          ("JevScale", "scale_to_g"), ("Label", "class_index"), ("Label_Description", "unit_id")):
        chosen[column] = chosen["Id"].astype(str).map(lambda k: bindings[k][field])
    area = run / ("factory_target" if target else "factory_source")
    area.mkdir(exist_ok=False)
    metadata = area / "metadata.csv"
    chosen.to_csv(metadata, index=False)
    partitions = {k: {"ids": []} for k in ("train", "val", "test")}
    for key in chosen["Id"].astype(str):
        role = bindings[key]["role"]
        split = "val" if role in {"tune", "cal"} else "test" if role == "external" else role
        partitions[split]["ids"].append(key)
    manifest = area / "native_split.json"
    manifest.write_text(json.dumps({"schema_version": 1, "group_key": "Label_Description",
        "stratify_key": "Label", "metadata_file_sha256": _sha256_file(metadata),
        "partitions": partitions}, indent=2))
    cache = area / "cache"
    cache.mkdir()
    for name in sorted(names):
        source = Path(cfg["data"]["root"]) / (name + ".h5")
        if not source.is_file():
            raise FileNotFoundError(source)
        # Inspect cache metadata only. Missing IDs must fail, not trigger a raw-reader fallback.
        import h5py
        expected = set(chosen.loc[chosen["Name"] == name, "Id"].astype(str))
        with h5py.File(source, "r") as stored:
            if not expected.issubset(stored.keys()):
                raise ValueError(f"Incomplete qualified cache for {name}; raw fallback is forbidden")
            for key in expected:
                shape = stored[key].shape
                channels = shape[1] if len(shape) in {2,3} else 1 if len(shape)==1 else 0
                if not channels or not 0 <= bindings[key]["channel_index"] < channels:
                    raise ValueError(f"Qualified channel does not match stored shape: {key}")
        # Native factory may write cache files; never give it raw-file links.
        temporary = cache / (name + ".copying")
        shutil.copy2(source, temporary)
        temporary.rename(cache / (name + ".h5"))
    data = dict(factory_name="default", data_dir=cfg["data"]["root"], metadata_file=str(metadata),
        use_cache=True, cache_dir=str(cache), normalization="none", window_size=cfg["signal"]["samples"],
        num_window=2**31-1, window_sampling_strategy="sequential", stride=cfg["signal"]["samples"],
        batch_size=cfg["fit"]["batch_size"], num_workers=0, dtype="float32",
        target_rate_hz=cfg["signal"]["sample_rate_hz"], resample_beta=cfg["signal"]["kaiser_beta"],
        split=dict(strategy="grouped_metadata", group_key="Label_Description", stratify_key="Label",
                   seed=20261017, test_policy="partition", manifest_mode="read_only",
                   partition_map={k:k for k in partitions}, manifest_path=str(manifest),
                   fractions={"train":0.5,"val":0.25,"test":0.25}))
    task = dict(type="Default_task", name="jev_common_rate", target_system_id=list(chosen.Dataset_id.unique()),
                batch_size=cfg["fit"]["batch_size"], num_workers=0, pin_memory=False)
    return build_data(SimpleNamespace(**data), SimpleNamespace(**task)), bindings


def export_native(root: Path, cfg: dict, run: Path, *, target: bool) -> None:
    """Export the existing native datasets, not raw H5 or reimplemented windows."""
    factory, binding = build_native_factory(root, cfg, run, target=target)
    parts = ("test",) if target else ("train", "val", "test")
    for split in parts:
        ds = getattr(factory, split + "_dataset")
        roles = ("external",) if target else ("tune", "cal") if split == "val" else (split,)
        for role in roles:
            indices = [i for i, s in enumerate(ds.file_windows_list) if binding[str(s["file_id"])]["role"] == role]
            if not indices:
                raise ValueError(f"Native dataset produced no observations for {role}")
            folder = run / "exports" / role
            folder.mkdir(parents=True, exist_ok=False)
            x = np.lib.format.open_memmap(folder / "x.npy", mode="w+", dtype=np.float32,
                                         shape=(len(indices), cfg["signal"]["samples"], 1))
            rows = []
            record_lengths = {}
            for j, index in enumerate(indices):
                item = ds[index]
                key = str(item["file_id"]); info = binding[key]
                if int(item["y"]) != info["class_index"]:
                    raise ValueError("Factory label differs from qualified event")
                x[j] = item["x"]
                s = ds.file_windows_list[index]
                begin, end = ds.dataset_dict[s["file_id"]].resampled_intervals[s["window_id"]]
                rows.append((info["class_index"], info["Name"]+":"+info["unit_id"], key, begin, end))
                if key not in record_lengths:
                    record_lengths[key] = dict(Id=key, Name=info["Name"], unit_id=info["unit_id"],
                        **ds.dataset_dict[s["file_id"]].record_lengths)
            x.flush(); del x
            np.savez(folder / "observations.npz", y=np.array([r[0] for r in rows]),
                unit=np.array([r[1] for r in rows]), record=np.array([r[2] for r in rows]),
                resampled_start=np.array([r[3] for r in rows]), resampled_stop=np.array([r[4] for r in rows]))
            with (folder / "record_lengths.csv").open("x", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(next(iter(record_lengths.values()))))
                writer.writeheader()
                writer.writerows(record_lengths.values())
