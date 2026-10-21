"""
Unit Test Suite for AneRBC Leakage-Safe Dataset Split Pipeline
--------------------------------------------------------------
Tests:
1. test_parent_grouping: All 12 crops + original smear co-located in same component & fold.
2. test_duplicate_hash_grouping: Duplicate image hashes never cross component or fold boundaries.
3. test_conflicting_labels: 22 mixed/conflicted components detected, reported, and kept intact.
4. test_deterministic_assignments: Identical seeds produce byte-for-byte identical split assignments.
5. test_no_leakage_and_coverage: 100% record coverage and 0 parent/hash overlap across folds.
"""

import sys
import unittest
import csv
from pathlib import Path
from collections import defaultdict

# Add parent directory to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from create_dataset_splits import (
    build_connected_components,
    allocate_stratified_group_folds,
    validate_splits,
    UnionFind
)


class TestDatasetSplits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        manifest_path = PROJECT_ROOT / "reports" / "dataset_manifest.csv"
        hash_path = PROJECT_ROOT / "reports" / "image_hash_audit.csv"

        if not manifest_path.exists() or not hash_path.exists():
            raise unittest.SkipTest("Manifest or hash audit file not found. Skipping full-data unit tests.")

        with open(manifest_path, "r", encoding="utf-8-sig", newline="") as f:
            cls.manifest_rows = list(csv.DictReader(f))
        with open(hash_path, "r", encoding="utf-8-sig", newline="") as f:
            cls.hash_rows = list(csv.DictReader(f))

        cls.comp_list, cls.records = build_connected_components(cls.manifest_rows, cls.hash_rows)
        cls.folds = allocate_stratified_group_folds(cls.comp_list, n_splits=5, seed=42)

    def test_parent_grouping(self):
        """Confirm all 13 files belonging to each parent (smear + 12 crops) are in the same component and fold."""
        # Check components
        parent_to_comps = defaultdict(set)
        for c in self.comp_list:
            for p in c["parent_ids"]:
                parent_to_comps[p].add(c["component_id"])

        for p, comps in parent_to_comps.items():
            self.assertEqual(len(comps), 1, f"Parent {p} is split across multiple connected components: {comps}")

        # Check folds
        parent_to_folds = defaultdict(set)
        for f in self.folds:
            for c in f["components"]:
                for p in c["parent_ids"]:
                    parent_to_folds[p].add(f["fold"])

        for p, fold_set in parent_to_folds.items():
            self.assertEqual(len(fold_set), 1, f"Parent {p} is split across multiple folds: {fold_set}")

        # Confirm 1,000 total parents evaluated
        self.assertEqual(len(parent_to_folds), 1000)

    def test_duplicate_hash_grouping(self):
        """Confirm identical SHA-256 images never cross component or fold boundaries."""
        hash_to_folds = defaultdict(set)
        for f in self.folds:
            for c in f["components"]:
                for h in c["sha256_hashes"]:
                    hash_to_folds[h].add(f["fold"])

        for h, fold_set in hash_to_folds.items():
            self.assertEqual(len(fold_set), 1, f"Hash {h[:12]} occurs across multiple folds: {fold_set}")

        # Total unique hashes should be 10,192
        self.assertEqual(len(hash_to_folds), 10192)

    def test_conflicting_labels(self):
        """Confirm conflicting-label components are detected, reported, and kept intact within single folds."""
        mixed_comps = [c for c in self.comp_list if c["is_mixed"]]
        # There should be exactly 22 mixed components encompassing 71 parents
        self.assertEqual(len(mixed_comps), 22, f"Expected 22 mixed components; got {len(mixed_comps)}")

        parents_in_mixed = set(p for c in mixed_comps for p in c["parent_ids"])
        self.assertEqual(len(parents_in_mixed), 71, f"Expected 71 parents in mixed components; got {len(parents_in_mixed)}")

        anemic_parents = [p for p in parents_in_mixed if p.endswith("_a")]
        healthy_parents = [p for p in parents_in_mixed if p.endswith("_h")]
        self.assertEqual(len(anemic_parents), 49)
        self.assertEqual(len(healthy_parents), 22)

        # Confirm none of the 22 mixed components is split across folds
        for c in mixed_comps:
            comp_folds = set()
            for f in self.folds:
                if any(comp["component_id"] == c["component_id"] for comp in f["components"]):
                    comp_folds.add(f["fold"])
            self.assertEqual(len(comp_folds), 1, f"Mixed component {c['component_id']} is split across folds: {comp_folds}")

    def test_deterministic_assignments(self):
        """Confirm identical seeds produce identical fold allocations."""
        run_a = allocate_stratified_group_folds(self.comp_list, n_splits=5, seed=42)
        run_b = allocate_stratified_group_folds(self.comp_list, n_splits=5, seed=42)

        for fa, fb in zip(run_a, run_b):
            self.assertEqual(fa["fold"], fb["fold"])
            self.assertEqual(fa["total_records"], fb["total_records"])
            self.assertEqual(fa["anemic_records"], fb["anemic_records"])
            self.assertEqual(fa["healthy_records"], fb["healthy_records"])
            self.assertEqual(sorted(fa["parent_ids"]), sorted(fb["parent_ids"]))

    def test_no_leakage_and_coverage(self):
        """Confirm full 13,000 coverage and 0 cross-fold leakage via validate_splits engine."""
        val_res = validate_splits(self.records, self.folds, n_splits=5)
        self.assertEqual(val_res["status"], "PASSED")
        self.assertEqual(val_res["total_records"], 13000)
        self.assertEqual(val_res["total_parents"], 1000)


if __name__ == "__main__":
    unittest.main()
