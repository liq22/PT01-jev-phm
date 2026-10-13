"""Synthetic window/split fixtures; no PHM measurements or model results."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"S03_Scripts"))
from run_jev_study import assemble_split_roles, resolve_window_plan, validate_export_coverage


class JevCoverageTests(unittest.TestCase):
    def test_common_window_count_covers_longest_record_and_overlap_is_parameterized(self):
        lengths = {"a": 13, "b": 11}
        default = resolve_window_plan(lengths, 4)
        self.assertEqual(default["num_window"], 4)
        self.assertEqual(default["max_stride"], 4)
        self.assertEqual(default["min_length"], 11)
        self.assertEqual(default["max_length"], 13)
        overlap = resolve_window_plan(lengths, 4, 2)
        self.assertEqual(overlap["num_window"], 6)
        self.assertEqual(overlap["max_stride"], 2)
        starts = np.array([0, 1, 3, 5, 7, 9, 0, 1, 2, 4, 5, 7])
        coverage = validate_export_coverage(
            {"file_id": np.array(["a"]*6+["b"]*6), "window_start": starts,
             "window_stop": starts+4}, lengths, overlap)
        self.assertLessEqual(coverage["maximum_actual_stride"], 2)
        self.assertGreaterEqual(coverage["minimum_overlap_samples"], 2)

    def test_divisible_tail_and_single_window_counts(self):
        self.assertEqual(resolve_window_plan({"a": 12}, 4)["num_window"], 3)
        self.assertEqual(resolve_window_plan({"a": 13}, 4)["num_window"], 4)
        plan = resolve_window_plan({"a": 4}, 4)
        self.assertEqual(plan["num_window"], 1)
        coverage = validate_export_coverage(
            {"file_id": np.array(["a"]), "window_start": np.array([0]),
             "window_stop": np.array([4])}, {"a": 4}, plan)
        self.assertEqual(coverage["coverage_fraction"], 1.)

    def test_invalid_stride_or_insufficient_record_is_not_silently_repaired(self):
        for stride in (0, 5, 1.5):
            with self.subTest(stride=stride), self.assertRaises(ValueError):
                resolve_window_plan({"a": 13}, 4, stride)
        with self.assertRaises(ValueError):
            resolve_window_plan({"a": 3}, 4)
        with self.assertRaises(ValueError):
            resolve_window_plan({"a": 4, "b": 13}, 4)

    def test_real_export_contract_covers_all_recordings_with_unique_starts(self):
        lengths = {"a": 13, "b": 11}
        plan = resolve_window_plan(lengths, 4)
        starts = np.array([0, 3, 6, 9, 0, 2, 4, 7])
        data = {"file_id": np.array(["a"]*4+["b"]*4),
                "window_start": starts, "window_stop": starts+4}
        coverage = validate_export_coverage(data, lengths, plan)
        self.assertEqual(coverage["files"], 2)
        self.assertEqual(coverage["windows"], 8)
        self.assertEqual(coverage["coverage_fraction"], 1.)
        self.assertEqual(coverage["maximum_actual_stride"], 3)
        self.assertGreaterEqual(coverage["minimum_overlap_samples"], 0)

    def test_coverage_rejects_duplicate_start_gap_wrong_width_and_missing_population(self):
        lengths = {"a": 13}
        plan = resolve_window_plan(lengths, 4)
        for starts in ([0, 3, 3, 9], [0, 1, 2, 9], [1, 3, 6, 9]):
            with self.subTest(starts=starts), self.assertRaises(ValueError):
                begin = np.array(starts)
                validate_export_coverage(
                    {"file_id": np.array(["a"]*4), "window_start": begin,
                     "window_stop": begin+4}, lengths, plan)
        good_starts = np.array([0, 3, 6, 9])
        wrong_stops = good_starts+4
        wrong_stops[1] += 1
        with self.assertRaises(ValueError):
            validate_export_coverage(
                {"file_id": np.array(["a"]*4), "window_start": good_starts,
                 "window_stop": wrong_stops}, lengths, plan)
        with self.assertRaises(ValueError):
            validate_export_coverage(
                {"file_id": np.array(["a"]*4), "window_start": good_starts,
                 "window_stop": good_starts+4}, {"a": 13, "b": 11}, plan)


class JevFrozenSplitTests(unittest.TestCase):
    @staticmethod
    def unit_labels():
        return {f"class_{label}_bearing_{i:02d}": label
                for label, count in ((0, 6), (1, 13), (2, 13)) for i in range(count)}

    def test_split_is_reproducible_and_primary_preserves_all_roles_and_classes(self):
        labels = self.unit_labels()
        protocols = assemble_split_roles(labels)
        self.assertEqual(protocols, assemble_split_roles(dict(reversed(list(labels.items())))))
        self.assertEqual(set(protocols), {"primary", *(f"rotation_{i}" for i in range(1, 7))})
        primary = protocols["primary"]
        self.assertEqual(set(primary), set(labels))
        for label, expected in ((0, (2, 1, 1, 2)), (1, (7, 1, 2, 3)), (2, (7, 1, 2, 3))):
            actual = tuple(sum(labels[u] == label and r == role for u, r in primary.items())
                           for role in ("train", "tune", "cal", "test"))
            self.assertEqual(actual, expected)

    def test_six_rotations_test_each_physical_bearing_exactly_once(self):
        labels = self.unit_labels()
        protocols = assemble_split_roles(labels)
        test_uses = {unit: 0 for unit in labels}
        for fold in range(1, 7):
            roles = protocols[f"rotation_{fold}"]
            self.assertEqual(set(roles), set(labels))
            for unit, role in roles.items():
                test_uses[unit] += int(role == "test")
            for label in (0, 1, 2):
                counts = {role: sum(labels[u] == label and r == role for u, r in roles.items())
                          for role in ("train", "tune", "cal", "test")}
                self.assertGreater(counts["train"], 0)
                self.assertEqual(counts["tune"], 1)
                self.assertEqual(counts["cal"], 1 if label == 0 else 2)
                self.assertEqual(counts["test"], 1 if label == 0 else (3 if fold == 1 else 2))
        self.assertEqual(set(test_uses.values()), {1})

    def test_missing_independent_bearing_is_not_replaced_by_windows(self):
        labels = self.unit_labels()
        del labels["class_0_bearing_00"]
        with self.assertRaises(ValueError):
            assemble_split_roles(labels)


if __name__ == "__main__":
    unittest.main()
