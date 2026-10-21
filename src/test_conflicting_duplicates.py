"""
Automated Test Suite for Conflicting-Label Duplicate Group Investigation
-----------------------------------------------------------------------
Validates all independent checks for the 286 conflicting duplicate groups
in the AneRBC dataset and leakage-safe 5-fold split allocations.

Usage:
    python src/test_conflicting_duplicates.py
"""

import unittest
import csv
from pathlib import Path
from collections import defaultdict, Counter


class TestConflictingDuplicates(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parent.parent
        cls.splits_csv = cls.project_root / "reports" / "dataset_splits.csv"
        cls.hash_csv = cls.project_root / "reports" / "image_hash_audit.csv"
        cls.investigation_csv = cls.project_root / "reports" / "conflicting_duplicate_investigation.csv"
        cls.investigation_txt = cls.project_root / "reports" / "conflicting_duplicate_investigation.txt"
        cls.split_audit_txt = cls.project_root / "reports" / "dataset_split_audit.txt"

        # Verify prerequisite files exist
        for p in [cls.splits_csv, cls.hash_csv, cls.investigation_csv, cls.investigation_txt, cls.split_audit_txt]:
            assert p.exists(), f"Required file not found: {p}"

        # Load splits CSV
        with open(cls.splits_csv, 'r', encoding='utf-8') as f:
            cls.splits_rows = list(csv.DictReader(f))

        # Load hash CSV
        with open(cls.hash_csv, 'r', encoding='utf-8-sig') as f:
            cls.hash_rows = list(csv.DictReader(f))

        # Build lookup maps
        cls.split_by_path = {r['image_path']: r for r in cls.splits_rows}

        # Parent smear hash map
        cls.parent_smear_hash = {}
        for h in cls.hash_rows:
            p = h['image_path']
            s = cls.split_by_path[p]
            if s['dataset_tier'] == 'AneRBC-I':
                cls.parent_smear_hash[s['original_image_id']] = h['sha256']

        # Group by SHA-256
        cls.records_by_hash = defaultdict(list)
        for h in cls.hash_rows:
            p = h['image_path']
            s = cls.split_by_path[p]
            cls.records_by_hash[h['sha256']].append({
                'image_path': p,
                'dataset_tier': s['dataset_tier'],
                'original_image_id': s['original_image_id'],
                'crop_id': s['crop_id'],
                'directory_label': s['directory_label'],
                'fold': int(s['fold']),
                'sha256': h['sha256']
            })

        # Conflicting groups
        cls.conflicting_groups = {}
        for h, recs in cls.records_by_hash.items():
            labels = set(r['directory_label'] for r in recs)
            if 'Anemic' in labels and 'Healthy' in labels:
                cls.conflicting_groups[h] = recs

    def test_total_conflicting_groups_count(self):
        """Verify exactly 286 conflicting SHA-256 hash groups exist."""
        self.assertEqual(len(self.conflicting_groups), 286,
                         f"Expected exactly 286 conflicting groups, found {len(self.conflicting_groups)}")

    def test_tier_breakdown_of_conflicts(self):
        """Verify 22 smear conflict groups, 264 crop conflict groups, 0 cross-tier."""
        smear_groups = [h for h, recs in self.conflicting_groups.items() if recs[0]['dataset_tier'] == 'AneRBC-I']
        crop_groups = [h for h, recs in self.conflicting_groups.items() if recs[0]['dataset_tier'] == 'AneRBC-II']
        cross_tier_groups = [h for h, recs in self.conflicting_groups.items()
                             if len(set(r['dataset_tier'] for r in recs)) > 1]

        self.assertEqual(len(smear_groups), 22, f"Expected 22 smear groups, got {len(smear_groups)}")
        self.assertEqual(len(crop_groups), 264, f"Expected 264 crop groups, got {len(crop_groups)}")
        self.assertEqual(len(cross_tier_groups), 0, f"Expected 0 cross-tier groups, got {len(cross_tier_groups)}")

    def test_crop_conflicts_trace_to_smears(self):
        """Verify 100% of the 264 crop conflict groups trace to the 22 smear conflict groups."""
        smear_conflict_hashes = {h for h, recs in self.conflicting_groups.items() if recs[0]['dataset_tier'] == 'AneRBC-I'}
        crop_groups = {h: recs for h, recs in self.conflicting_groups.items() if recs[0]['dataset_tier'] == 'AneRBC-II'}

        traced_count = 0
        for h, recs in crop_groups.items():
            pids = [r['original_image_id'] for r in recs]
            smear_hashes = set(self.parent_smear_hash.get(pid) for pid in pids)
            self.assertEqual(len(smear_hashes), 1, f"Crop group {h} has multiple parent smear hashes: {smear_hashes}")
            s_hash = list(smear_hashes)[0]
            self.assertIn(s_hash, smear_conflict_hashes,
                          f"Crop group {h} parent smear hash {s_hash} not in smear conflict groups")
            traced_count += 1

        self.assertEqual(traced_count, 264, f"Expected 264 traced crop groups, got {traced_count}")

    def test_conflicting_groups_zero_cross_fold_leakage(self):
        """Verify each of the 286 conflicting groups is allocated to exactly one fold."""
        leaked_groups = []
        for h, recs in self.conflicting_groups.items():
            folds = set(r['fold'] for r in recs)
            if len(folds) > 1:
                leaked_groups.append((h, folds))

        self.assertEqual(len(leaked_groups), 0,
                         f"Found {len(leaked_groups)} conflicting groups leaking across folds: {leaked_groups}")

    def test_global_zero_hash_leakage(self):
        """Verify that across ALL 13,000 records, no SHA-256 hash appears in multiple folds."""
        hash_to_folds = defaultdict(set)
        for h, recs in self.records_by_hash.items():
            for r in recs:
                hash_to_folds[h].add(r['fold'])

        leaked_hashes = {h: folds for h, folds in hash_to_folds.items() if len(folds) > 1}
        self.assertEqual(len(leaked_hashes), 0,
                         f"Found {len(leaked_hashes)} SHA-256 hashes leaking across folds!")

    def test_global_zero_parent_leakage(self):
        """Verify that across ALL 1,000 parents, no parent ID appears in multiple folds."""
        parent_to_folds = defaultdict(set)
        for r in self.splits_rows:
            parent_to_folds[r['original_image_id']].add(int(r['fold']))

        leaked_parents = {pid: folds for pid, folds in parent_to_folds.items() if len(folds) > 1}
        self.assertEqual(len(leaked_parents), 0,
                         f"Found {len(leaked_parents)} parent IDs leaking across folds!")

    def test_twelve_crops_colocation_per_parent(self):
        """Verify every parent has 1 smear and 12 crops, all in the same fold."""
        parent_tier_counts = defaultdict(Counter)
        parent_folds = defaultdict(set)
        for r in self.splits_rows:
            pid = r['original_image_id']
            parent_tier_counts[pid][r['dataset_tier']] += 1
            parent_folds[pid].add(int(r['fold']))

        self.assertEqual(len(parent_tier_counts), 1000, "Expected exactly 1,000 distinct parents")
        for pid, counts in parent_tier_counts.items():
            self.assertEqual(counts['AneRBC-I'], 1, f"Parent {pid} does not have exactly 1 AneRBC-I smear")
            self.assertEqual(counts['AneRBC-II'], 12, f"Parent {pid} does not have exactly 12 AneRBC-II crops")
            self.assertEqual(len(parent_folds[pid]), 1, f"Parent {pid} files span multiple folds: {parent_folds[pid]}")

    def test_independent_fold_recomputation(self):
        """Recompute fold counts directly from CSV and assert exact balance."""
        fold_records = Counter()
        fold_parents = defaultdict(set)
        fold_anemic = Counter()
        fold_healthy = Counter()

        for r in self.splits_rows:
            f = int(r['fold'])
            fold_records[f] += 1
            fold_parents[f].add(r['original_image_id'])
            if r['directory_label'] == 'Anemic':
                fold_anemic[f] += 1
            elif r['directory_label'] == 'Healthy':
                fold_healthy[f] += 1

        for f in range(5):
            self.assertEqual(fold_records[f], 2600, f"Fold {f} record count is not 2,600")
            self.assertEqual(len(fold_parents[f]), 200, f"Fold {f} parent count is not 200")
            self.assertEqual(fold_anemic[f], 1300, f"Fold {f} anemic count is not 1,300")
            self.assertEqual(fold_healthy[f], 1300, f"Fold {f} healthy count is not 1,300")

    def test_conflicting_duplicate_investigation_csv_integrity(self):
        """Verify the generated investigation CSV has 286 rows, 923 files, and 71 distinct parents."""
        with open(self.investigation_csv, 'r', encoding='utf-8') as f:
            inv_rows = list(csv.DictReader(f))

        self.assertEqual(len(inv_rows), 286, f"Expected 286 rows in investigation CSV, got {len(inv_rows)}")

        total_files = sum(int(r['record_count']) for r in inv_rows)
        self.assertEqual(total_files, 923, f"Expected 923 files across conflicting groups, got {total_files}")

        all_parents = set()
        for r in inv_rows:
            pids = r['parent_ids'].split('; ')
            all_parents.update(pids)
            # Record count must equal anemic + healthy
            self.assertEqual(int(r['record_count']), int(r['anemic_count']) + int(r['healthy_count']))
            # Fold must be valid
            self.assertIn(int(r['assigned_fold']), range(5))

        self.assertEqual(len(all_parents), 71, f"Expected 71 distinct parents involved, got {len(all_parents)}")
        anemic_parents = {p for p in all_parents if p.endswith('_a')}
        healthy_parents = {p for p in all_parents if p.endswith('_h')}
        self.assertEqual(len(anemic_parents), 49, f"Expected 49 anemic parents, got {len(anemic_parents)}")
        self.assertEqual(len(healthy_parents), 22, f"Expected 22 healthy parents, got {len(healthy_parents)}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
