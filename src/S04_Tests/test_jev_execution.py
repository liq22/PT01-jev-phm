"""Source-release and resume tests use local arrays and files, never PHM/model data."""
from __future__ import annotations

import argparse
import copy
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/S01_Package"))
import jev_execution as execution


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "run"
        self.run.mkdir()
        for name in ("g0.csv", "roles.csv", "metadata_pu_mfpt.csv"):
            (self.root / name).write_text("Id,unit_id,role\n1,bearing-a,train\n", encoding="utf-8")
        self.native_frame = pd.DataFrame({"Id": [1, 2], "Name": ["RM_027_PU", "RM_007_MFPT"],
            "Label": [0, 1], "Label_Description": ["bearing-fixture", "outer-fault-fixture"],
            "Sample_rate": [64000, 97656], "Channel": [1, 1], "unavailable": [None, None]})
        self.native_frame.to_csv(self.root / "metadata_pu_mfpt.csv", index=False)
        reader_patch = patch("jev_signal_data.read_approved_metadata", side_effect=self.metadata_fixture, create=True)
        self.native_reader = reader_patch.start()
        self.addCleanup(reader_patch.stop)
        self.cfg = {"name": "rf", "runtime": {},
                    "data": {"root": str(self.root), "metadata": "metadata_pu_mfpt.csv",
                             "qualification_file": "g0.csv", "pu_roles": "roles.csv"},
                    "signal": {}, "evaluation": {"probability_floor": 1e-12},
                    "family_configs": ["rf.yaml"], "seeds": [101],
                    "model": {"kind": "rf"}, "fit": {"batch_size": 3}}
        self.scale = {"mean": 0., "std": 1.}
        (self.run / "source_scale.json").write_text(json.dumps(self.scale))
        execution.bind_study(self.cfg, self.run, self.root)

    def metadata_fixture(self, root, config):
        if config["metadata"].endswith(".csv"):
            return pd.read_csv(Path(config["root"]) / config["metadata"])
        return self.native_frame.copy(deep=True)

    def selection(self):
        base = self.run / "fits" / self.cfg["name"]
        (base / "trial_00").mkdir(parents=True)
        (base / "trial_00" / "model.joblib").write_bytes(b"fixture-checkpoint")
        (base / "config.json").write_text(json.dumps(self.cfg))
        record = {"models": [{"seed": 101, "qualification": "bounded_source_selection_only",
                              "path": "trial_00/model.joblib"}], "trials": []}
        (base / "selection.json").write_text(json.dumps(record))
        return base, record

    def source_arrays(self):
        x = np.zeros((3, 8, 1), dtype=np.float32)
        meta = {"y": np.arange(3), "unit": np.array(["a", "b", "c"]),
                "record": np.array(["r1", "r2", "r3"]),
                "resampled_start": np.zeros(3, dtype=int), "resampled_stop": np.full(3, 8)}
        return x, meta, np.ones(3) / 3

    def predictor(self, *args):
        logits = np.array([[2., 0., 0.], [0., 2., 0.], [0., 0., 2.]])
        probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        return object(), lambda x: (np.log(probabilities), probabilities)

    def verify(self):
        with patch.object(execution, "validate_matrix", return_value=[self.cfg]), \
             patch("jev_learning.exported_role", return_value=self.source_arrays()), \
             patch.object(execution, "_load_predictor", side_effect=self.predictor) as loader:
            execution.verify_source(self.cfg, self.root, self.run, None)
            return loader.call_count

    def test_same_path_g0_content_change_blocks_resume(self):
        (self.root / "g0.csv").write_text("Id,unit_id,role\n1,bearing-a,test\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            execution.bind_study(self.cfg, self.run, self.root)

    def test_same_path_role_content_change_blocks_resume(self):
        (self.root / "roles.csv").write_text("Id,unit_id,role\n1,bearing-b,train\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            execution.bind_study(self.cfg, self.run, self.root)

    def test_native_workbook_uses_approved_metadata_api_for_contract(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["data"]["metadata"] = "metadata.xlsx"
        contract = execution._study_contract(cfg, self.root)
        self.native_reader.assert_called_with(self.root, cfg["data"])
        rows = contract["input_contents"]["metadata"]["rows"]
        self.assertEqual([row["Name"] for row in rows], ["RM_027_PU", "RM_007_MFPT"])
        self.assertIsNone(rows[0]["unavailable"])
        json.dumps(contract, allow_nan=False)

    def test_native_metadata_content_change_blocks_resume(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["data"]["metadata"] = "metadata.xlsx"
        (self.run / "study.json").write_text(json.dumps(execution._study_contract(cfg, self.root)))
        execution.bind_study(cfg, self.run, self.root)
        self.native_frame.loc[1, "Sample_rate"] = 12345
        with self.assertRaisesRegex(ValueError, "changed"):
            execution.bind_study(cfg, self.run, self.root)

    def test_metadata_observations_precede_missing_physical_qualification(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["data"]["qualification_file"] = None
        cfg["data"]["metadata"] = "metadata.xlsx"
        args = argparse.Namespace(config=self.root / "config.yaml", run_root=self.root / "not-created", stage="check")
        summary = {"records": 2, "datasets": {"RM_027_PU": {"records": 1}, "RM_007_MFPT": {"records": 1}},
                   "basis": "Native metadata fixture only; not qualification"}
        output = io.StringIO()
        upstream = Mock()
        with patch.object(execution, "read_config", return_value=cfg), \
             patch.object(execution, "validate_matrix"), \
             patch("jev_signal_data.summarize_native_metadata", return_value=summary, create=True), \
             patch("jev_signal_data.qualify_records") as qualify, \
             patch.object(execution, "check_inputs") as inputs, \
             patch.object(execution, "bind_gpu") as gpu, \
             contextlib.redirect_stdout(output):
            with self.assertRaisesRegex(RuntimeError, "original-record-to-physical-unit"):
                execution.dispatch(args, self.root, upstream)
        observed = json.loads(output.getvalue())
        self.assertEqual(observed["status"], "native_metadata_observed")
        self.assertEqual(observed["observations"], summary)
        self.assertEqual(observed["scientific_qualification"], "UNVERIFIED")
        qualify.assert_not_called()
        inputs.assert_not_called()
        gpu.assert_not_called()
        upstream.assert_not_called()
        self.assertFalse(args.run_root.exists())

    def test_native_metadata_error_is_not_hidden_by_unbound_qualification(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["data"]["qualification_file"] = None
        args = argparse.Namespace(config=self.root / "config.yaml", run_root=self.root / "not-created", stage="check")
        with patch.object(execution, "read_config", return_value=cfg), \
             patch.object(execution, "validate_matrix"), \
             patch("jev_signal_data.summarize_native_metadata", create=True), \
             patch("jev_signal_data.read_approved_metadata", side_effect=ValueError("Native metadata mismatch")):
            with self.assertRaisesRegex(ValueError, "Native metadata mismatch"):
                execution.dispatch(args, self.root, lambda: None)
        self.assertFalse(args.run_root.exists())

    def test_existing_checkpoints_do_not_release_target_without_verification(self):
        self.selection()
        with patch.object(execution, "validate_matrix", return_value=[self.cfg]):
            with self.assertRaisesRegex(RuntimeError, "verification incomplete"):
                execution.release_target(self.cfg, self.root, self.run)

    def test_failed_source_gate_precedes_target_export(self):
        self.selection()
        args = argparse.Namespace(config=self.root / "config.yaml", run_root=self.run, stage="infer")
        with patch.object(execution, "read_config", return_value=self.cfg), \
             patch.object(execution, "validate_matrix", return_value=[self.cfg]), \
             patch.object(execution, "check_inputs"), patch.object(execution, "bind_gpu"), \
             patch("jev_signal_data.export_native") as export:
            with self.assertRaisesRegex(RuntimeError, "verification incomplete"):
                execution.dispatch(args, self.root, lambda: None)
            export.assert_not_called()

    def test_source_roundtrip_calibration_and_resume_without_refitting(self):
        self.selection()
        self.assertEqual(self.verify(), 2)
        result = json.loads((self.run / "source_verification/rf/101/result.json").read_text())
        self.assertGreater(result["source_temperature"], 0)
        with np.load(self.run / "source_verification/rf/101/cal.npz", allow_pickle=False) as saved:
            self.assertEqual(float(saved["temperature"]), result["source_temperature"])
            np.testing.assert_array_equal(saved["unit"], ["a", "b", "c"])
        self.assertEqual(self.verify(), 0)

    def test_changed_checkpoint_blocks_target_after_verification(self):
        base, _ = self.selection()
        self.verify()
        (base / "trial_00/model.joblib").write_bytes(b"changed-checkpoint")
        with patch.object(execution, "validate_matrix", return_value=[self.cfg]):
            with self.assertRaisesRegex(RuntimeError, "checkpoint content changed"):
                execution.release_target(self.cfg, self.root, self.run)

    def test_semantic_variant_is_source_compiled_and_target_gate_checks_vectors(self):
        # Reuse the explicitly numerical TinyDecoder fixture, never pretrained weights.
        import torch
        from test_jev_semantic import make_model
        self.cfg["name"] = "semantic"
        self.cfg["model"] = {"kind": "decoder", "snapshot": "fixture-only"}
        self.cfg["evaluation"]["semantic_variants"] = {"race_wording": ["H", "I", "O"]}
        (self.run / "study.json").unlink()
        execution.bind_study(self.cfg, self.run, self.root)
        self.selection()
        vectors = torch.eye(3, 4)
        def predictor(*args):
            _, predict = self.predictor()
            return make_model(), predict
        with patch.object(execution, "validate_matrix", return_value=[self.cfg]), \
             patch("jev_learning.exported_role", return_value=self.source_arrays()), \
             patch.object(execution, "_load_predictor", side_effect=predictor), \
             patch("jev_semantic.compile_semantic_variant", return_value=(vectors, .01)) as compile_vectors:
            execution.verify_source(self.cfg, self.root, self.run, torch.device("cpu"))
        compile_vectors.assert_called_once()
        area = self.run / "source_verification/semantic/101"
        with np.load(area / "cal.variant_race_wording.npz", allow_pickle=False) as saved:
            self.assertEqual(float(saved["temperature"]), 1.)
            self.assertEqual(saved["logits"].shape, (3, 3))
        torch.save(vectors + 1, area / "variant_race_wording.pt")
        with patch.object(execution, "validate_matrix", return_value=[self.cfg]):
            with self.assertRaisesRegex(RuntimeError, "variant vectors changed"):
                execution.release_target(self.cfg, self.root, self.run)

    def test_inconsistent_reload_retains_failure_and_prevents_implicit_retry(self):
        self.selection()
        with patch.object(execution, "validate_matrix", return_value=[self.cfg]), \
             patch("jev_learning.exported_role", return_value=self.source_arrays()), \
             patch.object(execution, "_load_predictor", side_effect=[self.predictor(),
                   (object(), lambda x: (np.zeros((3, 3)), None))]):
            with self.assertRaisesRegex(ValueError, "Reloaded source predictions disagree"):
                execution.verify_source(self.cfg, self.root, self.run, None)
        self.assertTrue((self.run / "source_verification/rf/101/failure.json").is_file())
        with patch.object(execution, "validate_matrix", return_value=[self.cfg]), \
             patch("jev_learning.exported_role", return_value=self.source_arrays()), \
             patch.object(execution, "_load_predictor") as loader:
            with self.assertRaises(FileExistsError):
                execution.verify_source(self.cfg, self.root, self.run, None)
            loader.assert_not_called()

    def test_matrix_requires_approved_46_fits(self):
        cfg = execution.read_config(ROOT / "src/S02_Configs/real/semantic.yaml")
        families = execution.validate_matrix(cfg, ROOT)
        self.assertEqual(len(families), 6)
        families = copy.deepcopy(families)
        families[0]["search"]["max_depth"].append(16)
        with patch.object(execution, "read_config", side_effect=families):
            with self.assertRaisesRegex(ValueError, "46-fit matrix"):
                execution.validate_matrix(cfg, ROOT)

    def test_saved_json_configuration_preserves_numeric_contract(self):
        cfg = execution.read_config(ROOT / "src/S02_Configs/real/rf.yaml")
        saved = self.root / "config.json"
        saved.write_text(json.dumps(cfg), encoding="utf-8")
        restored = execution.read_config(saved)
        self.assertIsInstance(restored["evaluation"]["probability_floor"], float)
        self.assertEqual(restored["evaluation"]["probability_floor"], 1e-12)
        self.assertEqual(restored, cfg)
        self.assertEqual(len(execution.validate_matrix(restored, ROOT)), 6)

    def test_verify_source_stage_is_in_existing_cli(self):
        parser = argparse.ArgumentParser()
        execution.add_subparser(parser.add_subparsers(required=True))
        args = parser.parse_args(["semantic", "verify-source", "--config", "x.yaml", "--run-root", "x"])
        self.assertEqual(args.stage, "verify-source")

    def test_file_presence_does_not_establish_model_smoke(self):
        (self.run / "smoke").mkdir()
        (self.run / "smoke/result.json").write_text(json.dumps({"rows": 3}))
        with self.assertRaisesRegex(RuntimeError, "does not verify"):
            execution._require_smoke(self.cfg, self.run)

    def test_smoke_is_bound_to_the_corresponding_model_configuration(self):
        (self.run / "smoke").mkdir()
        path = self.run / "smoke/result.json"
        path.write_text(json.dumps({"status": "source_smoke_passed", "families": {"rf": {"config": self.cfg}}}))
        execution._require_smoke(self.cfg, self.run)
        cfg = copy.deepcopy(self.cfg)
        cfg["model"]["trees"] = 999
        with self.assertRaisesRegex(RuntimeError, "does not verify"):
            execution._require_smoke(cfg, self.run)


class ShellExecutionTests(unittest.TestCase):
    """Stub executables check orchestration without reading data or using a GPU."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        shutil.copyfile(ROOT / "run_all.sh", self.root / "run_all.sh")
        (self.root / "AGENTS.md").write_text("Test fixture\n")
        entry = self.root / "src/S03_Scripts/run_jev_study.py"
        entry.parent.mkdir(parents=True)
        entry.write_text("# Never executed: test uses an explicit interpreter stub.\n")
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.executable("git", '#!/bin/sh\n[ "$1" != rev-parse ] || echo fixture-revision\nexit 0\n')
        self.executable("nvidia-smi", '#!/bin/sh\nprintf queried > "$TEST_GPU_QUERIED"\nprintf "%s, GPU-fixture\\n" "$TEST_GPU_INDEX"\n')
        self.python = self.executable("python-stub", '#!/bin/sh\nif [ "$1" = -c ]; then echo GPU-fixture; exit 0; fi\nprintf "%s|%s\\n" "$CUDA_VISIBLE_DEVICES" "$*" >> "$TEST_CALLS"\nexit 0\n')
        self.environment = {**os.environ, "PATH": str(self.bin) + os.pathsep + os.environ["PATH"],
            "JEV_PYTHON": str(self.python), "JEV_RUN": str(self.root / "run"),
            "TEST_GPU_INDEX": "0", "TEST_GPU_QUERIED": str(self.root / "gpu-queried"),
            "TEST_CALLS": str(self.root / "calls"), "CUDA_VISIBLE_DEVICES": "GPU-inherited"}

    def executable(self, name, content):
        path = self.bin / name
        path.write_text(content)
        path.chmod(0o755)
        return path

    def invoke(self, *args):
        return subprocess.run(["bash", str(self.root / "run_all.sh"), *args],
            env=self.environment, text=True, capture_output=True, timeout=10)

    def test_check_is_cpu_only_and_never_queries_gpu(self):
        result = self.invoke("--check")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.root / "gpu-queried").exists())
        calls = (self.root / "calls").read_text().splitlines()
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0].startswith("|"))
        self.assertIn("semantic check", calls[0])

    def test_physical_gpu_two_blocks_before_model_command(self):
        self.environment["TEST_GPU_INDEX"] = "2"
        result = self.invoke()
        self.assertEqual(result.returncode, 2)
        self.assertIn("physical GPU 2", result.stderr)
        self.assertEqual(len((self.root / "calls").read_text().splitlines()), 1)

    def test_new_run_root_cannot_reset_approved_budget(self):
        budget = self.root / "src/outputs/jev-phm/p1-approved-20261017/execution"
        budget.mkdir(parents=True)
        (budget / "run-root.txt").write_text(str(self.root / "previous-run") + "\n")
        (budget / "usage.tsv").write_text("gpu\tfit_semantic\t800\t0\tprevious.log\n")
        result = self.invoke()
        self.assertEqual(result.returncode, 2)
        self.assertIn("already bound to another run root", result.stderr)
        self.assertIn("800", (budget / "usage.tsv").read_text())
        self.assertEqual(len((self.root / "calls").read_text().splitlines()), 1)

    def completed_artifacts(self):
        run = Path(self.environment["JEV_RUN"])
        for name in ("source_scale.json", "smoke/result.json", "summary.csv", "paired_unit_effects.csv"):
            path = run / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture artifact, not a real completed study\n")
        for name in ("paired_diagnostic_effects", "error_processing_cost", "selective_diagnosis_and_alarms"):
            for suffix in ("svg", "pdf"):
                path = run / "figures" / f"{name}.{suffix}"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture artifact\n")
        for family in ("rf", "signal", "indexed", "semantic", "token"):
            path = run / "timing" / family / "summary.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n")
        return run

    def test_complete_figure_outputs_do_not_charge_another_summary_on_resume(self):
        self.completed_artifacts()
        result = self.invoke("--resume")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("semantic summarize", (self.root / "calls").read_text())
        usage = self.root / "src/outputs/jev-phm/p1-approved-20261017/execution/usage.tsv"
        self.assertNotIn("\tsummarize\t", usage.read_text())

    def test_missing_required_figure_pdf_runs_summary_on_resume(self):
        run = self.completed_artifacts()
        (run / "figures/selective_diagnosis_and_alarms.pdf").unlink()
        result = self.invoke("--resume")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((self.root / "calls").read_text().count("semantic summarize"), 1)


if __name__ == "__main__":
    unittest.main()
