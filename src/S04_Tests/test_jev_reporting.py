"""Numerical reporting fixtures; no PHM data, pretrained model or CUDA is used."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/S01_Package"))
from jev_reporting import (alarm_and_acceptance, paired_nll_effect, prediction_distribution,
    render_saved, risk_coverage, score_saved, summarize_saved_predictions, time_selected, _measure_operations)


def saved_fixture(*, singleton: bool = False) -> dict:
    # Unit a has two records, one with two windows. Unit b has one record.
    y = np.array([0, 0, 0, 0, 1, 1, 2, 2])
    units = np.array(["a", "a", "a", "b", "c", "d", "e", "f"])
    if singleton:
        units[units == "b"] = "a"
    record = np.array(["a1", "a2", "a2", "b1", "c1", "d1", "e1", "f1"])
    return dict(y=y, unit=units, record=record, logits=np.zeros((len(y), 3)),
                temperature=np.array(2.), resampled_start=np.arange(len(y)) * 8192,
                resampled_stop=(np.arange(len(y)) + 1) * 8192)


class ReportingTests(unittest.TestCase):
    def test_record_then_unit_weights_and_confusion_f1(self):
        saved = saved_fixture()
        result = score_saved(saved)
        self.assertAlmostEqual(result["nll"], np.log(3))
        self.assertAlmostEqual(result["error"], 2 / 3)
        self.assertAlmostEqual(result["macro_f1"], 1 / 6)
        self.assertAlmostEqual(result["brier"], 2 / 3)
        self.assertAlmostEqual(result["ece"], 0.)
        # A single deliberately wrong window in a two-window record carries
        # 1/6 unit * 1/2 records * 1/2 windows, rather than 1/8 pooled windows.
        saved["logits"][:] = -30
        saved["logits"][np.arange(len(saved["y"])), saved["y"]] = 30
        saved["logits"][1] = [0, 30, 0]
        self.assertAlmostEqual(score_saved(saved)["error"], 1 / 24)

    def test_rf_floor_does_not_change_brier_or_argmax(self):
        saved = saved_fixture()
        p = np.eye(3)[saved["y"]]
        p[0] = [0, 1, 0]
        saved["raw_probabilities"] = p
        result = score_saved(saved, floor=.1)
        self.assertAlmostEqual(result["nll"], -np.log(.1) / 12)
        self.assertAlmostEqual(result["brier"], 2 / 12)
        self.assertAlmostEqual(result["error"], 1 / 12)
        self.assertAlmostEqual(result["log_floor_weight"], 1 / 12)
        np.testing.assert_array_equal(prediction_distribution(saved, floor=.1)[0], p)

    def test_stable_logits_and_calibration_do_not_change_argmax(self):
        saved = saved_fixture()
        saved["logits"][:, 1] = 1000
        raw = score_saved(saved)
        calibrated = score_saved(saved, calibrated=True)
        self.assertTrue(np.isfinite(raw["nll"]))
        self.assertEqual(raw["error"], calibrated["error"])
        self.assertNotEqual(raw["nll"], calibrated["nll"])

    def test_saved_weight_conflict_rejected(self):
        saved = saved_fixture()
        saved["weights"] = np.ones(len(saved["y"])) / len(saved["y"])
        with self.assertRaisesRegex(ValueError, "Saved weights"):
            score_saved(saved)

    def test_ece_equal_width_upper_endpoint(self):
        saved = saved_fixture()
        saved["raw_probabilities"] = np.eye(3)[saved["y"]]
        self.assertEqual(score_saved(saved)["ece"], 0.)
        saved["raw_probabilities"][:] = [1, 0, 0]
        self.assertAlmostEqual(score_saved(saved)["ece"], 2 / 3)

    def test_risk_coverage_stable_ties_and_zero_acceptance(self):
        p = np.array([[.6, .4, 0], [.6, 0, .4], [0, 0, 1.]])
        curve = risk_coverage(p, np.array([0, 2, 2]), np.array([.2, .3, .5]))
        np.testing.assert_array_equal(curve["order"], [2, 0, 1])
        np.testing.assert_allclose(curve["coverage"], [0, .5, .7, 1])
        self.assertTrue(np.isnan(curve["accepted_error"][0]))
        np.testing.assert_allclose(curve["accepted_error"][1:], [0, 0, .3])
        none = alarm_and_acceptance(p[:2], np.array([0, 2]), np.array([.4, .6]), acceptance_threshold=.99)
        self.assertEqual(none["coverage"], 0)
        self.assertIsNone(none["accepted_error"])

    def test_alarm_denominators_do_not_delete_rejections(self):
        p = np.array([[.2, .4, .4], [.8, .1, .1], [.1, .8, .1]])
        y, weights = np.array([0, 1, 2]), np.array([.5, .25, .25])
        result = alarm_and_acceptance(p, y, weights, acceptance_threshold=.99)
        self.assertEqual(result["coverage"], 0.)
        self.assertEqual(result["false_alarm_rate"], 1.)
        self.assertEqual(result["missed_fault_rate"], .5)
        self.assertIsNone(result["scenario_cost"])
        no_health = alarm_and_acceptance(p[1:], y[1:], np.array([.5, .5]))
        self.assertIsNone(no_health["false_alarm_rate"])

    def test_paired_cluster_bootstrap_is_seed_mean_and_reproducible(self):
        baseline = saved_fixture()
        better = saved_fixture()
        better["logits"][np.arange(len(better["y"])), better["y"]] = 1.
        rows, effect = paired_nll_effect({101: better, 202: better}, {101: baseline, 202: baseline})
        expected = np.log(np.e + 2) - 1 - np.log(3)
        self.assertEqual(len(rows), 12)
        self.assertAlmostEqual(effect["mean_difference"], expected)
        self.assertAlmostEqual(effect["fit_sd"], 0)
        np.testing.assert_allclose(effect["paired_unit_95_interval"], [expected, expected])
        self.assertEqual(effect["bootstrap_draws"], 10000)
        self.assertEqual(effect["bootstrap_seed"], 20261017)
        _, repeated = paired_nll_effect({101: better, 202: better}, {101: baseline, 202: baseline})
        self.assertEqual(effect, repeated)

    def test_singleton_and_unknown_units_do_not_get_intervals(self):
        saved = saved_fixture(singleton=True)
        _, result = paired_nll_effect({101: saved}, {101: saved})
        self.assertIsNone(result["paired_unit_95_interval"])
        self.assertIsNone(result["fit_sd"])
        self.assertIn("fewer than two", result["interval_unavailable_reason"])
        saved = saved_fixture()
        saved["unit"] = saved["unit"].astype("U40")
        saved["unit"][saved["unit"] == "b"] = "MFPT:UNKNOWN"
        _, result = paired_nll_effect({101: saved}, {101: saved})
        self.assertIn("UNKNOWN", result["interval_unavailable_reason"])
        self.assertEqual(result["bootstrap_draws"], 0)

    def test_pair_mismatch_and_label_conflict_fail(self):
        saved, other = saved_fixture(), saved_fixture()
        other["resampled_start"][0] += 1
        with self.assertRaisesRegex(ValueError, "different observations"):
            paired_nll_effect({101: saved}, {101: other})
        saved["y"][0] = 1
        with self.assertRaisesRegex(ValueError, "inconsistent diagnostic states"):
            paired_nll_effect({101: saved}, {101: saved})

    def test_timing_measurement_contract_and_interleaving(self):
        counts = {"a": 0, "b": 0}
        def operation(name):
            def call(index):
                counts[name] += 1
                return index
            return call
        rows = _measure_operations({name: operation(name) for name in counts}, np.array([0, 4, 8]), sync=lambda: None)
        self.assertEqual(counts, {"a": 230, "b": 230})
        self.assertEqual(len(rows), 400)
        self.assertEqual([r["readout"] for r in rows[:2]], ["a", "b"])
        self.assertEqual([r["readout"] for r in rows[80:82]], ["b", "a"])
        self.assertEqual({r["block"] for r in rows}, set(range(5)))
        with self.assertRaises(ValueError):
            _measure_operations({"a": operation("a")}, np.array([0]), sync=lambda: None, samples=10)

    def test_rf_timing_saves_actual_outputs_and_separate_source_preprocessing(self):
        class FakeRF:
            def predict_proba(self, features):
                return np.array([[.6, .3, .1]])
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            fit = run / "fits" / "rf"
            fit.mkdir(parents=True)
            (fit / "selection.json").write_text(json.dumps({"models": [{"seed": 101, "path": "fixture.joblib"}]}))
            (run / "source_scale.json").write_text(json.dumps({"mean": 0, "std": 1}))
            folder = run / "exports" / "train"
            folder.mkdir(parents=True)
            (folder / "record_lengths.csv").write_text(
                "Id,Name,resampling_seconds,resampling_scope\n1,RM_027_PU,.01,fixture-only\n2,RM_027_PU,.03,fixture-only\n")
            cfg = {"name": "rf", "model": {"kind": "rf"},
                   "evaluation": {"timing_warmup": 30, "timing_samples": 200}}
            exported = (np.zeros((3, 8, 1)), {"y": np.array([0, 1, 2])}, np.ones(3) / 3)
            with patch("joblib.load", return_value=FakeRF()), patch("jev_learning.exported_role", return_value=exported), patch("jev_learning.signal_statistics", return_value=np.zeros((1, 12))):
                time_selected(cfg, run, None)
            result = json.loads((run / "timing" / "rf" / "summary.json").read_text())
            self.assertEqual(result["source_preprocessing_records"], 2)
            self.assertAlmostEqual(result["source_preprocessing_total_seconds"], .04)
            self.assertIsNone(result["cold_candidate_encoding_ms"])
            outputs = json.loads((run / "timing" / "rf" / "timing_predictions.json").read_text())
            self.assertEqual([r["true_class"] for r in outputs], [0, 1, 2])
            self.assertEqual([r["predicted_class"] for r in outputs], [0, 0, 0])
            with (run / "timing" / "rf" / "observations.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 200)

    def test_saved_pipeline_recomputes_without_model_and_generation_accuracy_is_actual(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            for name in ("semantic", "indexed", "token"):
                for seed in (101, 202):
                    folder = run / "evaluations" / name / str(seed)
                    folder.mkdir(parents=True)
                    for role in ("test", "external"):
                        values = saved_fixture()
                        if name == "token":
                            # Perfect token logits but ALL generated labels wrong.
                            values["logits"][np.arange(len(values["y"])), values["y"]] = 30
                            values["generated_class"] = (values["y"] + 1) % 3
                            values["generated_class"][0] = -1
                        np.savez_compressed(folder / f"{role}.npz", **values)
                        if name == "semantic":
                            variant = dict(values)
                            variant["temperature"] = np.array(1.)
                            variant["logits"] = np.ones_like(values["logits"]) * 2
                            np.savez_compressed(folder / f"{role}.variant_race_wording.npz", **variant)
            cfg = {"evaluation": {"probability_floor": 1e-12, "semantic_variants": {"race_wording": ["h", "i", "o"]}}, "family_configs": ["semantic", "indexed", "token"]}
            with patch("jev_execution.read_config", side_effect=lambda path: {"name": path.name, "seeds": [101, 202]}):
                summarize_saved_predictions(cfg, run, run)
                # Same saved outputs are safe to recompute; a changed output is not overwritten.
                summarize_saved_predictions(cfg, run, run)
            with (run / "summary.csv").open() as handle:
                metrics = list(csv.DictReader(handle))
            generation = [r for r in metrics if r["view"] == "generated"]
            self.assertTrue(all(np.isclose(float(r["error"]), 1) for r in generation))
            self.assertTrue(all(r["nll"] == "" for r in generation))
            self.assertEqual(len(metrics), 28)
            with (run / "reporting" / "semantic_variant_sensitivity.csv").open() as handle:
                sensitivity = list(csv.DictReader(handle))
            self.assertEqual(len(sensitivity), 4)
            self.assertTrue(all(np.isclose(float(r["delta_nll_vs_canonical"]), 0) for r in sensitivity))
            effect = json.loads((run / "reporting" / "paired_effects.json").read_text())
            self.assertEqual(effect["external_raw"]["fits"], 2)
            timing = run / "timing" / "token"
            timing.mkdir(parents=True)
            (timing / "summary.json").write_text(json.dumps({"seed": 101, "measurements": {
                "token": {"p50_ms": 1, "p95_ms": 2}, "shortest_legal_generation": {"p50_ms": 2, "p95_ms": 3}}}))
            render_saved(cfg, run, run)
            for name in ("paired_diagnostic_effects", "error_processing_cost", "selective_diagnosis_and_alarms"):
                self.assertIn("<svg", (run / "figures" / f"{name}.svg").read_text())
                self.assertTrue((run / "figures" / f"{name}.pdf").read_bytes().startswith(b"%PDF"))
            self.assertIn("shortest_legal_generation", (run / "figures" / "error_processing_cost.svg").read_text())


if __name__ == "__main__":
    unittest.main()
