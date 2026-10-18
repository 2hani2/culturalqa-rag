"""Checks for honest retrieval accounting and immutable experimental inputs."""
import json
import unittest
import torch
from run_experiment import HERE, ROOT, metrics as conditional_metrics, zscore
from global_development import metrics as global_metrics
from reranker_models import sha256_file


class ExperimentTests(unittest.TestCase):
    def test_missing_positive_counts_as_failure(self):
        result = global_metrics([torch.tensor([2., 1.])], [
            {'refs': ['DOSA#1', 'DOSA#2'], 'positive_ref': 'DOSA#9'}])
        self.assertEqual(result['hit_at_12'], 0)
        self.assertEqual(result['mrr_at_12'], 0)
        self.assertEqual(result['ranks'], [None])

    def test_residual_zero_preserves_baseline(self):
        ce = torch.tensor([[4., 2., -1.]])
        fusion = torch.tensor([[-1., 10., 8.]])
        self.assertEqual(ce.argsort(descending=True).tolist(),
                         (zscore(ce) + 0*zscore(fusion)).argsort(descending=True).tolist())

    def test_constant_scores_do_not_make_injected_positive_perfect(self):
        result = conditional_metrics(torch.zeros(60, 12))
        self.assertLess(result['hit_at_1'], 0.5)
        self.assertLess(result['mrr_at_12'], 0.5)

    def test_zscore_is_finite_for_constant_scores(self):
        self.assertTrue(torch.isfinite(zscore(torch.zeros(2, 12))).all())

    def test_original_sources_and_models_unchanged(self):
        manifest = json.loads((HERE/'input_fingerprints.json').read_text())
        for name, expected in manifest.items():
            if name == 'experiment_script_sha256':
                actual = sha256_file(HERE/'run_experiment.py')
            else:
                actual = sha256_file(ROOT/name)
            self.assertEqual(actual, expected, name)

    def test_train_and_dev_groups_disjoint_and_no_test_examples(self):
        manifest = json.loads((HERE/'candidate_manifest.json').read_text())
        split = json.loads((ROOT/'research/split_manifest.json').read_text())['assignments']
        groups = {}
        for fold in ('train','dev'):
            groups[fold] = set()
            for row in manifest[fold]:
                self.assertEqual(split[row['positive_ref']]['fold'], fold)
                groups[fold].add(split[row['positive_ref']]['group'])
                self.assertEqual(len(row['candidate_refs']),12)
                self.assertEqual(len(set(row['candidate_refs'])),12)
                self.assertTrue(all(split[ref]['fold']==fold for ref in row['candidate_refs']))
        self.assertFalse(groups['train'] & groups['dev'])


if __name__ == '__main__':
    unittest.main()
