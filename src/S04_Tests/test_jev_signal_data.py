"""Pure code fixtures for P1 data contracts; no PHM payloads or model calls."""
from __future__ import annotations

import copy
import csv
import os
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/S01_Package"))
import jev_signal_data as signal_data


class SignalQualificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.unit = "Real Inner Ring Damage KI21"
        # Synthetic records reuse the approved role assignments to test the
        # population contract; they provide no G0 or measurement evidence.
        with (ROOT / "paper/assets/tables/jev_p1_pu_unit_roles.csv").open(newline="") as handle:
            self.roles = list(csv.DictReader(handle))
        self.rows = []
        source_roles = sorted(self.roles, key=lambda row: row["physical_unit"] != self.unit)
        for role in source_roles:
            for index in range(int(role["source_record_count"])):
                self.rows.append(dict(Id=str(len(self.rows) + 1), Name="RM_027_PU",
                    File=f"fixture/record_{len(self.rows)}.mat", unit_id=role["physical_unit"],
                    role=role["role"], class_index=role["class_index"], raw_label=role["raw_label"],
                    sample_rate_hz="64000", channel_index="2", scale_to_g="0.5",
                    physical_source="fixture-only", measurement_source="fixture-only", license_source="fixture-only"))
        for raw_label in [0] * 3 + [1] * 10 + [2] * 7:
            self.rows.append(dict(Id=str(len(self.rows) + 1), Name="RM_007_MFPT",
                File=f"fixture/record_{len(self.rows)}.mat", unit_id=f"MFPT-fixture-{raw_label}",
                role="external", class_index=str(signal_data.RAW_TO_CANONICAL["RM_007_MFPT"][raw_label]),
                raw_label=str(raw_label), sample_rate_hz="64000", channel_index="0", scale_to_g="0.5",
                physical_source="fixture-only", measurement_source="fixture-only", license_source="fixture-only"))
        descriptions = {0: "baseline", 1: "outer race fault", 2: "inner race fault"}
        self.frame = pd.DataFrame([dict(Id=int(r["Id"]), Dataset_id=20 if r["Name"] == "RM_027_PU" else 7,
            Name=r["Name"], File=r["File"], Label=int(float(r["raw_label"])),
            Label_Description=r["unit_id"] if r["Name"] == "RM_027_PU" else descriptions[int(r["raw_label"])],
            Sample_rate=64000, Channel=3 if r["Name"] == "RM_027_PU" else 1) for r in self.rows])
        self.config = dict(qualification_file="qualification.csv", pu_roles="roles.csv",
                           root=str(self.root), metadata="metadata_pu_mfpt.csv")
        self.write_csv("roles.csv", self.roles)

    def write_csv(self, name, rows):
        with (self.root / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def qualify(self):
        self.write_csv("qualification.csv", self.rows)
        self.frame.to_csv(self.root / "metadata_pu_mfpt.csv", index=False)
        return signal_data.qualify_records(self.root, self.config)

    def write_workbook(self, *, frame=None, excluded_rows=()):
        """Minimal inline-string worksheet fixture, never a real workbook."""
        frame = self.frame if frame is None else frame
        def row_xml(index, values):
            cells = []
            for column, value in enumerate(values):
                reference = f"{chr(65 + column)}{index}"
                if isinstance(value, str):
                    cells.append(f'<c r="{reference}" t="inlineStr"><is><t>{escape(value)}</t></is></c>')
                elif value is not None:
                    cells.append(f'<c r="{reference}" t="n"><v>{value}</v></c>')
            return f'<row r="{index}">' + "".join(cells) + '</row>'
        parts = [row_xml(1, list(frame.columns)), *excluded_rows]
        parts.extend(row_xml(index + 100, values) for index, values in enumerate(frame.itertuples(index=False, name=None)))
        path = self.root / "metadata.xlsx"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as workbook:
            workbook.writestr("xl/worksheets/sheet1.xml",
                '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
                + "".join(parts) + '</sheetData></worksheet>')
        self.config["metadata"] = path.name
        return path

    def test_preprojected_csv_retains_exact_approved_population(self):
        selected, bindings = self.qualify()
        self.assertEqual(len(selected), 2339)
        self.assertEqual(bindings["1"]["channel_index"], 2)
        self.assertEqual(bindings["1"]["scale_to_g"], 0.5)

    def test_workbook_decodes_only_header_and_allowlisted_name_cells(self):
        # This malformed excluded row would break any whole-sheet XML reader.
        # Name is column C; an approved name in a different cell must not route it.
        excluded = ('<row r="2"><c r="C2" t="inlineStr"><is><t>EXCLUDED_FIXTURE</t></is></c>'
                    '<c r="D2" t="inlineStr"><is><t>RM_027_PU</t></is></c><broken & invalid></row>')
        path = self.write_workbook(excluded_rows=(excluded,))
        original = path.read_bytes()
        with patch.object(signal_data.ET, "fromstring", wraps=signal_data.ET.fromstring) as decode:
            with patch.object(pd, "read_excel", side_effect=AssertionError("Whole workbook forbidden")):
                selected = signal_data.read_approved_metadata(self.root, self.config)
        self.assertEqual(len(selected), 2339)
        self.assertEqual(decode.call_count, 2340)
        self.assertTrue(all(b"EXCLUDED_FIXTURE" not in call.args[0] for call in decode.call_args_list))
        self.assertEqual(path.read_bytes(), original)

    def test_workbook_can_qualify_correct_mapping_without_upstream_initialization(self):
        self.write_workbook()
        self.write_csv("qualification.csv", self.rows)
        selected, binding = signal_data.qualify_records(self.root, self.config)
        self.assertEqual(len(selected), 2339)
        self.assertEqual(binding["2323"]["raw_label"], "1")
        self.assertEqual(binding["2323"]["class_index"], 2)

    def test_mixed_bearing_cannot_masquerade_as_fixed_single_ring_unit(self):
        self.frame.loc[0, "Label_Description"] = "Real Inner Ring Damage KB23"
        with self.assertRaisesRegex(ValueError, "Native P1 population"):
            self.qualify()

    def test_approved_pu_units_cannot_swap_record_identity(self):
        other = next(index for index, row in self.frame.iterrows()
                     if row["Name"] == "RM_027_PU" and row["Label"] == 1 and row["Label_Description"] != self.unit)
        self.frame.loc[0, "Label_Description"], self.frame.loc[other, "Label_Description"] = (
            self.frame.loc[other, "Label_Description"], self.frame.loc[0, "Label_Description"])
        with self.assertRaisesRegex(ValueError, "Native PU physical identity"):
            self.qualify()

    def test_original_file_identity_must_match(self):
        self.frame.loc[0, "File"] = "fixture/a_different_record.mat"
        with self.assertRaisesRegex(ValueError, "Record correspondence"):
            self.qualify()

    def test_raw_label_cannot_disagree_with_frozen_pu_role(self):
        self.rows[0]["raw_label"] = "2"
        self.frame.loc[0, "Label"] = 2
        with self.assertRaisesRegex(ValueError, "raw-to-canonical label mapping"):
            self.qualify()

    def test_extraneous_native_rows_are_not_silently_filtered(self):
        extra = self.frame.iloc[[0]].copy()
        extra["Id"] = 9000
        self.frame = pd.concat([self.frame, extra], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Native P1 population"):
            self.qualify()

    def test_native_duplicate_record_id_stops_projection(self):
        self.frame.loc[1, "Id"] = self.frame.loc[0, "Id"]
        with self.assertRaisesRegex(ValueError, "missing or ambiguous"):
            self.qualify()

    def test_integer_id_conversion_cannot_hide_duplicate_identity(self):
        self.frame.loc[1, "Id"] = "0001"
        self.write_workbook()
        with self.assertRaisesRegex(ValueError, "ambiguous after integer conversion"):
            signal_data.read_approved_metadata(self.root, self.config)

    def test_blank_original_file_is_not_a_qualified_correspondence(self):
        self.frame.loc[0, "File"] = "   "
        with self.assertRaisesRegex(ValueError, "original-file correspondence"):
            self.qualify()

    def test_mfpt_outer_and_inner_labels_cannot_use_pu_number_meanings(self):
        self.rows[2322]["class_index"] = "1"  # MFPT raw 1 is outer, canonical 2.
        with self.assertRaisesRegex(ValueError, "raw-to-canonical label mapping"):
            self.qualify()

    def test_mfpt_native_description_conflict_is_not_relabelled(self):
        for description in ("inner race fault", "outerracefault"):
            with self.subTest(description=description):
                self.frame.loc[2322, "Label_Description"] = description
                with self.assertRaisesRegex(ValueError, "MFPT native label descriptions"):
                    self.qualify()

    def test_summary_keeps_native_counts_separate_from_physical_qualification(self):
        selected, _ = self.qualify()
        summary = signal_data.summarize_native_metadata(selected)
        mfpt = summary["datasets"]["RM_007_MFPT"]
        self.assertEqual(mfpt["raw_to_canonical"], {"0": 0, "1": 2, "2": 1})
        self.assertEqual(mfpt["canonical_label_counts"], {"0": 3, "2": 10, "1": 7})
        self.assertEqual(mfpt["approved_physical_unit_count"], "UNKNOWN")
        self.assertEqual(summary["datasets"]["RM_027_PU"]["approved_physical_unit_count"], 29)
        self.assertIn("not established", summary["basis"])

    def test_unknown_g0_and_unbound_g0_stop(self):
        self.config["qualification_file"] = None
        with self.assertRaisesRegex(FileNotFoundError, "G0 is incomplete"):
            signal_data.qualify_records(self.root, self.config)
        self.config["qualification_file"] = "qualification.csv"
        self.rows[0]["physical_source"] = "UNKNOWN"
        with self.assertRaisesRegex(ValueError, "Unqualified record"):
            self.qualify()

    def test_invalid_measurements_stop_before_metadata_and_waveforms(self):
        first = copy.deepcopy(self.rows[0])
        for field, bad_value in (("sample_rate_hz", "64000.5"), ("sample_rate_hz", "inf"),
                                 ("channel_index", "-1"), ("channel_index", "0.5"),
                                 ("scale_to_g", "inf"), ("scale_to_g", "0")):
            with self.subTest(field=field, value=bad_value):
                self.rows[0] = dict(first, **{field: bad_value})
                self.write_csv("qualification.csv", self.rows)
                with patch.object(pd, "read_csv") as native_reader:
                    with self.assertRaises(ValueError):
                        signal_data.qualify_records(self.root, self.config)
                    native_reader.assert_not_called()

    def test_target_cannot_be_assigned_a_source_role(self):
        self.rows[-1]["role"] = "train"
        with self.assertRaisesRegex(ValueError, "entirely external"):
            self.qualify()


class SignalTailTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / "external/PHM-Vibench/src/data_factory/__init__.py").is_file(),
                         "Pinned PHMFactory source not initialized; native adapter integration NOT RUN")
    def test_actual_pinned_factory_adapter_with_numeric_fixture(self):
        """Actual adapter API compatibility, not G0 or a PHM experiment."""
        code = textwrap.dedent("""
            import sys
            from pathlib import Path
            from types import SimpleNamespace
            import numpy as np

            root = Path(sys.argv[1])
            factory_root = root / 'external/PHM-Vibench'
            sys.path.insert(0, str(factory_root))
            sys.path.insert(0, str(root / 'src/S01_Package'))
            import src.data_factory as factory
            from src.data_factory.dataset_task.Default_dataset import Default_dataset
            from src.data_factory.dataset_task.adapters import resolve_dataset_adapter
            import jev_signal_data as signal_data

            assert Path(factory.__file__).resolve().is_relative_to(factory_root.resolve())
            signal_data.register_adapter()
            adapter = resolve_dataset_adapter('Default_task', 'jev_common_rate')
            assert issubclass(adapter, Default_dataset)
            args_data = SimpleNamespace(window_size=8, num_window=2**31-1,
                window_sampling_strategy='sequential', stride=8,
                split={'strategy': 'grouped_metadata'}, normalization='none',
                target_rate_hz=25, resample_beta=5.)
            args_task = SimpleNamespace(type='Default_task', name='jev_common_rate')
            numeric_fixture = np.arange(83*3, dtype=np.float64).reshape(83, 3, 1)
            metadata_fixture = {'numeric-fixture-only': dict(
                Label=1, JevRate=64, JevChannel=2, JevScale=.5)}
            dataset = adapter({'numeric-fixture-only': numeric_fixture},
                metadata_fixture, args_data, args_task, mode='train')
            expected = signal_data.common_rate_signal(numeric_fixture,
                sample_rate_hz=64, channel_index=2, scale_to_g=.5,
                target_rate_hz=25, beta=5.)
            assert expected.shape == (33, 1)
            assert len(dataset) == 4
            assert dataset.resampled_intervals == [(0,8), (8,16), (16,24), (24,32)]
            assert dataset.window_intervals is None
            for index, (begin, end) in enumerate(dataset.resampled_intervals):
                np.testing.assert_array_equal(dataset[index]['x'], expected[begin:end])
                assert dataset[index]['y'] == 1
            report = dataset.record_lengths
            assert report['input_samples'] == 83
            assert report['resampled_samples'] == 33
            assert report['retained_samples'] == 32
            assert report['discarded_tail_samples'] == 1
            assert report['discarded_tail_seconds'] == 1/25
            assert report['resampling_seconds'] >= 0
            assert 'excludes H5 IO' in report['resampling_scope']
        """)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
                           OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        result = subprocess.run([sys.executable, "-c", code, str(ROOT)], env=environment,
                                text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_adapter_records_resampled_tail_without_calling_it_raw_support(self):
        registry = {}
        factory = types.ModuleType("src.data_factory")
        factory.DATASET_ADAPTERS = registry
        factory.register_dataset_adapter = lambda *key, **kwargs: registry.update(
            {key: SimpleNamespace(**kwargs)})
        module = types.ModuleType("src.data_factory.dataset_task.Default_dataset")

        class WindowBoundaryFixture:
            """Injected native boundary: one known interval, no real data IO."""
            def __init__(self, data, metadata, args_data, args_task, mode):
                self.window_intervals = [(0, 4)]

        module.Default_dataset = WindowBoundaryFixture
        with patch.dict(sys.modules, {"src.data_factory": factory,
                        "src.data_factory.dataset_task.Default_dataset": module}):
            signal_data.register_adapter()
            dataset = signal_data.CommonRateDataset({"fixture": np.arange(11)[:, None]},
                {"fixture": dict(JevRate=10, JevChannel=0, JevScale=1)},
                SimpleNamespace(target_rate_hz=5, resample_beta=5), SimpleNamespace())
        lengths = dataset.record_lengths
        self.assertEqual(lengths["input_samples"], 11)
        self.assertEqual(lengths["resampled_samples"], 6)
        self.assertEqual(lengths["discarded_tail_samples"], 2)
        self.assertAlmostEqual(lengths["discarded_tail_seconds"], 0.4)
        self.assertGreaterEqual(lengths["resampling_seconds"], 0)
        self.assertIn("excludes H5 IO", lengths["resampling_scope"])
        self.assertEqual(dataset.resampled_intervals, [(0, 4)])
        self.assertIsNone(dataset.window_intervals)

    def test_export_saves_one_tail_disclosure_per_original_record(self):
        class ExportBoundaryFixture:
            file_windows_list = [dict(file_id="fixture", window_id=0),
                                 dict(file_id="fixture", window_id=1)]
            dataset_dict = {"fixture": SimpleNamespace(resampled_intervals=[(0, 4), (4, 8)],
                record_lengths=dict(input_samples=19, input_rate_hz=10, resampled_samples=10,
                    resampled_rate_hz=5, retained_samples=8, discarded_tail_samples=2,
                    discarded_tail_seconds=0.4, resampling_seconds=0.125,
                    resampling_scope="whole native-cache record channel/scale/polyphase transform; excludes H5 IO"))}

            def __getitem__(self, index):
                return dict(file_id="fixture", y=1, x=np.full((4, 1), index, dtype=np.float32))

        factory = SimpleNamespace(test_dataset=ExportBoundaryFixture())
        binding = {"fixture": dict(Name="RM_007_MFPT", unit_id="unit-fixture",
                                   role="external", class_index=1)}
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            with patch.object(signal_data, "build_native_factory", return_value=(factory, binding)):
                signal_data.export_native(run, dict(signal=dict(samples=4)), run, target=True)
            folder = run / "exports/external"
            with (folder / "record_lengths.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["discarded_tail_samples"], "2")
            self.assertEqual(rows[0]["input_samples"], "19")
            self.assertEqual(rows[0]["resampling_seconds"], "0.125")
            np.testing.assert_array_equal(np.load(folder / "observations.npz")["resampled_stop"], [4, 8])


if __name__ == "__main__":
    unittest.main()
