"""Numerical/interface fixtures only: these are not PHM or Qwen results."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.special import softmax

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"S01_Package"))
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

    def test_full_numeric_contract(self):
        data = self.fixture()
        # Fake embeddings exercise orchestration, not a real language model.
        e = dict(hidden=data["features"], record=data["record"], provenance=np.asarray("synthetic, NOT Qwen"))
        with tempfile.TemporaryDirectory() as tmp:
            result = run(data, e, {"C_grid": [.1, 1.], "reject_cost": .25, "bootstrap_seed": 0, "bootstrap_repetitions": 40}, Path(tmp)/"run")
            self.assertEqual(result["numeric"]["raw"]["accuracy"], result["numeric"]["unit"]["accuracy"])
            saved = json.loads((Path(tmp)/"run/metrics.json").read_text())
            self.assertIn("conditional_ci95", saved["qwen"]["unit_minus_pooled_nll"])


if __name__ == "__main__":
    unittest.main()
