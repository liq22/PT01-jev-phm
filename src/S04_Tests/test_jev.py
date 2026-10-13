"""Numerical/interface fixtures only: these are not PHM or Qwen results."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.special import logsumexp, softmax

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"S01_Package"))
import jev_phm as jev
from jev_phm import check_data, signal_features, temperature, typed_decisions, unit_weights, run


class JevContractTests(unittest.TestCase):
    def test_unit_weights(self):
        np.testing.assert_allclose(unit_weights(np.array(["a", "a", "b"])), [.25, .25, .5])

    def test_equal_sizes_recover_pooled(self):
        z = np.array([[3., 0.], [0., 1.], [1., 0.], [2., 0.]])
        y = np.array([0, 1, 1, 0])
        self.assertAlmostEqual(temperature(z, y), temperature(z, y, np.array(["a", "a", "b", "b"])), places=6)

    def test_duplicating_one_unit_does_not_change_unit_objective(self):
        z = np.array([[3., 0.], [0., 1.], [1., 0.], [2., 0.]])
        y, g = np.array([0, 1, 1, 0]), np.array(["a", "a", "b", "b"])
        take = np.array([0, 1, 0, 1, 2, 3])
        self.assertAlmostEqual(temperature(z, y, g), temperature(z[take], y[take], g[take]), places=5)

    def test_temperature_preserves_argmax(self):
        z = np.array([[2., -2.], [-1., 4.]])
        np.testing.assert_array_equal(softmax(z/3, axis=1).argmax(1), z.argmax(1))

    def test_typed_reject_and_accept(self):
        out = typed_decisions(np.array([[.9, .1], [.5, .5], [.75, .25]]), ["healthy", "fault"], .25)
        self.assertEqual([r["kind"] for r in out], ["diagnosis", "defer", "defer"])
        self.assertIsNone(out[1]["label"])

    def test_invalid_probability_is_not_repaired(self):
        with self.assertRaises(ValueError):
            typed_decisions(np.array([[.6, .6]]), ["a", "b"], .25)

    def test_signal_scope(self):
        np.testing.assert_allclose(signal_features(np.zeros(32)), np.zeros(6))
        with self.assertRaises(ValueError):
            signal_features(np.ones((4, 4)))

    @staticmethod
    def fixture():
        rng = np.random.default_rng(0)
        y = np.tile(np.repeat([0, 1], 6), 4)
        role = np.repeat(["train", "tune", "cal", "test"], 12)
        # Two independent units per class per role, three windows per unit.
        unit = np.array([f"{r}_{i//3}" for r in ["train", "tune", "cal", "test"] for i in range(12)])
        x = rng.normal(size=(48, 6)) + y[:, None]
        return dict(features=x, y=y, role=role, unit=unit, record=np.array([str(i) for i in range(48)]), labels=np.array(["a", "b"]), provenance=np.asarray("synthetic unit-test fixture"))

    def test_group_leakage_rejected(self):
        data = self.fixture()
        data["unit"][12] = data["unit"][0]
        with self.assertRaises(ValueError):
            check_data(data)

    def test_missing_test_class_is_not_repaired_by_windows(self):
        data = self.fixture()
        data["y"][data["role"] == "test"] = 0
        with self.assertRaises(ValueError):
            check_data(data)

    def test_binary_temperature_preserves_confidence_ranking(self):
        # Keep even the lowest tested temperature below floating-point saturation.
        z = np.array([[.2, -.2], [-.1, .4], [0., 0.], [.1, 0.], [0., .1]])
        original_order = np.argsort(-softmax(z, axis=1).max(1), kind="stable")
        for t in (.05, .7, 3., 20.):
            scaled = softmax(z/t, axis=1)
            np.testing.assert_array_equal(scaled.argmax(1), z.argmax(1))
            np.testing.assert_array_equal(
                np.argsort(-scaled.max(1), kind="stable"), original_order)

    def test_full_numeric_contract(self):
        data = self.fixture()
        # Fake embeddings exercise orchestration, not a real language model.
        e = dict(hidden=data["features"], record=data["record"], provenance=np.asarray("synthetic, NOT Qwen"))
        with tempfile.TemporaryDirectory() as tmp:
            result = run(data, e, {"C_grid": [.1, 1.], "reject_cost": .25, "bootstrap_seed": 0, "bootstrap_repetitions": 40}, Path(tmp)/"run")
            self.assertEqual(result["numeric"]["raw"]["accuracy"], result["numeric"]["unit"]["accuracy"])
            saved = json.loads((Path(tmp)/"run/metrics.json").read_text())
            self.assertIn("conditional_ci95", saved["qwen"]["unit_minus_pooled_nll"])


class JevFactorizationTests(unittest.TestCase):
    """Conditional-probability properties, not evidence of diagnosis efficacy."""

    def test_composition_is_normalized_and_recovers_parent_probability(self):
        presence = np.array([[1., -2.], [-3., 4.], [2., 2.]])
        identity = np.array([[2., -1.], [1., 3.], [-4., 2.]])
        parent = softmax(presence, axis=1)
        conditional = softmax(identity, axis=1)
        for healthy_index in (0, 1, 2):
            composed = jev.compose_log_probs(presence, identity, healthy_index)
            probabilities = np.exp(composed)
            fault_columns = [k for k in range(3) if k != healthy_index]
            np.testing.assert_allclose(logsumexp(composed, axis=1), 0., atol=1e-14)
            np.testing.assert_allclose(probabilities[:, healthy_index], parent[:, 0])
            np.testing.assert_allclose(probabilities[:, fault_columns].sum(1), parent[:, 1])
            np.testing.assert_allclose(
                probabilities[:, fault_columns], parent[:, 1, None]*conditional)

    def test_weighted_leaf_nll_equals_branch_sums_with_original_fault_weights(self):
        presence = np.array([[2., -1.], [-2., 1.], [1., 2.], [0., 1.], [1., -1.]])
        identity = np.array([[0., 2.], [2., 0.], [1., 0.], [-1., 2.], [0., 1.]])
        y = np.array([1, 0, 2, 1, 2])  # Healthy is deliberately not column zero.
        weights = np.array([8., 2., 6., 3., 1.])
        fault = y != 1
        fault_y = np.array([0, 1, 1])  # Original leaf columns 0 and 2.
        leaf = jev.nll(jev.compose_log_probs(presence, identity, 1), y, weights)
        branches = (jev.nll(presence, fault.astype(int), weights)
                    + jev.nll(identity[fault], fault_y, weights[fault]))
        self.assertAlmostEqual(leaf, branches, places=12)
        separately_averaged = (
            jev.nll(presence, fault.astype(int), weights/weights.sum())
            + jev.nll(identity[fault], fault_y, weights[fault]/weights[fault].sum()))
        self.assertNotAlmostEqual(leaf/weights.sum(), separately_averaged, places=5)

    def test_conditional_head_uses_only_fault_rows_and_preserves_weights(self):
        x_presence = np.arange(30., dtype=float).reshape(5, 6)
        x_identity = x_presence + 100.
        y = np.array([1, 0, 2, 1, 2])
        weights = np.array([8., 2., 6., 3., 1.])
        captured = []

        def capture_fit(x, target, C, sample_weight=None):
            captured.append((x.copy(), target.copy(), C, sample_weight.copy()))
            return {"capture": len(captured)}

        with patch.object(jev, "fit_head", side_effect=capture_fit):
            factor = jev.fit_factorized(
                x_presence, x_identity, y, .3, sample_weight=weights, healthy_index=1)
        self.assertEqual(len(captured), 2)
        np.testing.assert_array_equal(captured[0][0], x_presence)
        np.testing.assert_array_equal(captured[0][1], [0, 1, 1, 0, 1])
        np.testing.assert_array_equal(captured[0][3], weights)
        fault = y != 1
        np.testing.assert_array_equal(captured[1][0], x_identity[fault])
        np.testing.assert_array_equal(captured[1][1], [0, 1, 1])
        np.testing.assert_array_equal(captured[1][3], weights[fault])
        self.assertEqual(captured[0][2], captured[1][2])
        np.testing.assert_array_equal(factor["fault_classes"], [0, 2])

    def test_both_prediction_branches_execute_for_every_row(self):
        x_presence = np.arange(24., dtype=float).reshape(4, 6)
        x_identity = x_presence + 100.
        presence = np.array([[9., -9.], [-9., 9.], [1., 0.], [0., 1.]])
        identity = np.array([[1., 2.], [3., 1.], [-2., 2.], [0., 1.]])
        factor = {"presence": {"name": "presence"}, "identity": {"name": "identity"},
                  "healthy_index": 0, "fault_classes": np.array([1, 2])}
        with patch.object(jev, "predict_head", side_effect=[presence, identity]) as predictor:
            result = jev.predict_factorized(factor, x_presence, x_identity)
        self.assertEqual(predictor.call_count, 2)
        np.testing.assert_array_equal(predictor.call_args_list[0].args[1], x_presence)
        np.testing.assert_array_equal(predictor.call_args_list[1].args[1], x_identity)
        self.assertEqual(result.shape, (4, 3))
        np.testing.assert_allclose(result, jev.compose_log_probs(presence, identity))

    def test_direct_questions_use_fixed_arithmetic_probability_fusion(self):
        first = np.array([[6., 0., -2.], [-1., 3., 1.]])
        second = np.array([[-3., 2., 1.], [4., -1., 1.]])
        fused = jev.direct_fusion_logits(first, second)
        expected = .5*(softmax(first, axis=1)+softmax(second, axis=1))
        np.testing.assert_allclose(np.exp(fused), expected)
        np.testing.assert_allclose(np.exp(fused).sum(1), 1.)
        self.assertFalse(np.allclose(expected, softmax((first+second)/2., axis=1)))

    def test_record_training_mass_is_independent_of_window_count(self):
        files = np.array(["a", "b", "a", "c", "b", "b", "b", "c", "b"])
        weights = jev.window_training_weights(files)
        for file_id in np.unique(files):
            self.assertAlmostEqual(weights[files == file_id].sum(), 64.)
        np.testing.assert_allclose(weights[files == "a"], [32., 32.])
        np.testing.assert_allclose(weights[files == "b"], np.full(5, 64./5))

    def test_duplicate_windows_with_redistributed_weights_preserve_fitted_head(self):
        x = np.array([[0., 1.], [1., 2.], [1., 0.], [2., 1.], [3., 2.], [4., 0.]])
        y = np.array([0, 0, 1, 1, 2, 2])
        weights = np.array([4., 1., 3., 7., 2., 5.])
        take = np.array([0, 1, 1, 2, 3, 3, 3, 4, 5])
        repeats = np.bincount(take, minlength=len(x))
        original = jev.fit_head(x, y, .3, sample_weight=weights)
        duplicated = jev.fit_head(
            x[take], y[take], .3, sample_weight=weights[take]/repeats[take])
        np.testing.assert_allclose(original["scaler"].mean_, duplicated["scaler"].mean_)
        np.testing.assert_allclose(original["scaler"].scale_, duplicated["scaler"].scale_)
        np.testing.assert_allclose(
            jev.predict_head(original, x), jev.predict_head(duplicated, x), atol=1e-6)

    def test_extreme_head_scores_keep_finite_log_probabilities(self):
        x = np.zeros((2, 6))
        for score in (np.array([-1000., 1000.]),
                      np.array([[1000., -1000., 0.], [-1000., 1000., 0.]])):
            head = {"scaler": SimpleNamespace(transform=lambda rows: rows),
                    "classifier": SimpleNamespace(decision_function=lambda rows: score)}
            logits = jev.predict_head(head, x)
            self.assertTrue(np.isfinite(logits).all())
            np.testing.assert_allclose(logsumexp(logits, axis=1), 0., atol=1e-14)
            wrong_y = np.array([1, 0])
            self.assertGreater(jev.nll(logits, wrong_y, np.full(2, .5)), 900.)

    def test_two_fault_classes_are_required(self):
        x = np.zeros((4, 6))
        with self.assertRaises(ValueError):
            jev.fit_factorized(x, x, np.array([0, 1, 0, 1]), 1.)
        with self.assertRaises(ValueError):
            jev.compose_log_probs(np.zeros((4, 2)), np.zeros((4, 1)))

    def test_head_selection_uses_train_and_tune_only(self):
        x = np.random.default_rng(5).normal(size=(24, 3))
        y = np.tile(np.repeat([0, 1, 2], 2), 4)
        roles = np.repeat(jev.ROLES, 6)
        units = np.array([f"{role}_{i//2}" for role in jev.ROLES for i in range(6)])
        weights = np.arange(1., 25.)
        head, C, tuning = jev.select_head(x, y, roles, units, [.1, 1.], weights)
        changed_x, changed_y = x.copy(), y.copy()
        held_out = np.isin(roles, ["cal", "test"])
        changed_x[held_out] += 10000.
        changed_y[held_out] = (changed_y[held_out]+1) % 3
        changed, changed_C, changed_tuning = jev.select_head(
            changed_x, changed_y, roles, units, [.1, 1.], weights)
        self.assertEqual(C, changed_C)
        self.assertEqual(tuning, changed_tuning)
        np.testing.assert_allclose(
            head["scaler"].mean_, np.average(x[roles == "train"], axis=0,
                                          weights=weights[roles == "train"]))
        np.testing.assert_allclose(jev.predict_head(head, x), jev.predict_head(changed, x))

    def test_all_question_inputs_share_observation_and_complete_class_definitions(self):
        features = np.array([[.123, .456, .789, 1.234, 2.345, 3.456],
                             [.123, .456, .789, 1.234, 2.345, 3.456]])
        labels = ["healthy", "inner_dominant", "outer_dominant"]
        queries = jev.build_queries(features, labels)
        self.assertTrue({"flat_a", "flat_b", "presence", "identity"}.issubset(queries))
        for kind in ("flat_a", "flat_b", "presence", "identity"):
            texts = queries[kind]
            self.assertEqual(len(texts), 2)
            self.assertEqual(texts[0], texts[1])  # No sample ID, role, or truth routing.
            for label in labels:
                self.assertIn(label, texts[0], msg=kind)
            for name, value in zip(jev.FEATURES, features[0]):
                self.assertIn(f"{name}={value:.8g}", texts[0], msg=kind)
        self.assertIn("condition", queries["identity"][0].lower())

    def test_e2_saved_predictions_match_fusion_and_shared_representation_control(self):
        rng = np.random.default_rng(11)
        y = np.tile(np.repeat([0, 1, 2], 9), 4)
        roles = np.repeat(jev.ROLES, 27)
        units = np.array([f"{role}_{k}_{i//3}"
                          for role in jev.ROLES for k in range(3) for i in range(9)])
        x = rng.normal(size=(108, 6))+y[:, None]
        records = np.array([f"{unit}:{i%3}" for i, unit in enumerate(units)])
        data = {"features": x, "y": y, "role": roles, "unit": units,
                "file_id": units, "record": records,
                "labels": np.array(["healthy", "inner", "outer"]),
                "provenance": np.asarray("synthetic unit-test fixture")}
        embeddings = {"hidden_flat_a": x, "hidden_flat_b": x*x,
                      "hidden_presence": x, "hidden_identity": x,
                      "record": records, "provenance": np.asarray("synthetic, NOT Qwen")}
        config = {"C_grid": [.1, 1.], "healthy_index": 0, "reject_cost": .25,
                  "bootstrap_seed": 11, "bootstrap_repetitions": 40}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = jev.run_e2(data, embeddings, config, root/"first")
            changed = dict(data, y=y.copy())
            test = roles == "test"
            changed["y"][test] = (y[test]+1) % 3
            changed_report = jev.run_e2(changed, embeddings, config, root/"changed_test_truth")
            with np.load(root/"first/predictions.npz", allow_pickle=False) as saved:
                np.testing.assert_allclose(
                    saved["Qwen_shared_factorized_logits"], saved["Numerical_factorized_logits"])
                np.testing.assert_allclose(saved["Qwen_two_question_direct_logits"],
                    jev.direct_fusion_logits(saved["Qwen_two_question_direct_branch_a_logits"],
                                             saved["Qwen_two_question_direct_branch_b_logits"]))
                for name in ("Numerical-flat", "Numerical-factorized", "Qwen-flat",
                             "Qwen-factorized", "Qwen-two-question-direct", "Qwen-shared-factorized"):
                    self.assertEqual(report[name]["C"], changed_report[name]["C"])
                    self.assertEqual(report[name]["temperatures"], changed_report[name]["temperatures"])
                    np.testing.assert_allclose(
                        np.exp(saved[name.replace("-", "_")+"_logits"]).sum(1), 1.)
                    self.assertAlmostEqual(report[name]["raw"]["accuracy"],
                                           report[name]["pooled"]["accuracy"])
            saved_report = json.loads((root/"first/metrics.json").read_text())
            self.assertIn("Qwen-factorized_minus_Qwen-flat", saved_report["contrasts"])


class JevExportTests(unittest.TestCase):
    def test_multichannel_export_uses_declared_channel_and_upstream_intervals(self):
        signal = np.column_stack([np.full(32, 100.), np.full(32, -20.), np.arange(32.)])

        class FixtureDataset:
            def __init__(self, roles):
                self.items, self.file_windows_list, self.dataset_dict = [], [], {}
                for role in roles:
                    for label in range(3):
                        file_id = f"{role}_{label}"
                        self.items.append({"file_id": file_id, "signal": signal, "label": float(label)})
                        self.file_windows_list.append({"file_id": file_id, "window_id": 0})
                        self.dataset_dict[file_id] = SimpleNamespace(window_intervals=[(10, 42)])

            def __len__(self):
                return len(self.items)

            def __getitem__(self, i):
                return self.items[i]

        factory = SimpleNamespace(train_dataset=FixtureDataset(["train"]),
                                  val_dataset=FixtureDataset(["tune", "cal"]),
                                  test_dataset=FixtureDataset(["test"]))
        with tempfile.TemporaryDirectory() as tmp:
            assignments = Path(tmp)/"units.csv"
            assignments.write_text("file_id,unit_id,role\n"+"".join(
                f"{role}_{label},{role}_{label},{role}\n" for role in jev.ROLES for label in range(3)))
            data = jev.export_factory(factory, assignments,
                                      {"0.0": "healthy", "1.0": "inner", "2.0": "outer"},
                                      "signal", "label", channel=2)
            np.testing.assert_allclose(data["features"],
                                       np.tile(jev.signal_features(signal[:, 2]), (12, 1)))
            np.testing.assert_array_equal(data["window_start"], np.full(12, 10))
            np.testing.assert_array_equal(data["window_stop"], np.full(12, 42))
            np.testing.assert_array_equal(data["sample_weight"], np.full(12, 64.))
            with self.assertRaises(ValueError):
                jev.export_factory(factory, assignments,
                                   {"0.0": "healthy", "1.0": "inner", "2.0": "outer"},
                                   "signal", "label")


class JevDeviceTests(unittest.TestCase):
    def test_uuid_mapping_excludes_forbidden_physical_gpu_two(self):
        # Mock inventory only: this test never initializes CUDA or accesses a GPU.
        inventory = "0, GPU-allowed, Fixture GPU\n2, GPU-forbidden, Fixture GPU\n"
        with patch.object(jev.subprocess, "check_output", return_value=inventory), \
                patch.dict(jev.sys.modules, {"torch": None}), \
                patch.dict(jev.os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
            mapping = jev.restrict_cuda("cuda:0", "GPU-allowed")
            self.assertEqual(mapping["physical_index"], 0)
            self.assertEqual(mapping["device"], "cuda:0")
            self.assertEqual(jev.os.environ["CUDA_VISIBLE_DEVICES"], "GPU-allowed")
            with self.assertRaises(ValueError):
                jev.restrict_cuda("cuda:0", "GPU-forbidden")
            self.assertEqual(jev.os.environ["CUDA_VISIBLE_DEVICES"], "GPU-allowed")
            with self.assertRaises(ValueError):
                jev.restrict_cuda("cuda:1", "GPU-allowed")


if __name__ == "__main__":
    unittest.main()
