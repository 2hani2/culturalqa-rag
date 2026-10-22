"""Unrestricted development-selected cross-encoder + MoE reranking."""
import json
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from langchain_core.documents import Document
from project_paths import ROOT
from reranker_models import create_model, sha256_file

SELECTION = ROOT / 'experiments/moe_improvement/global_selection.json'


def standardize(values):
    values = torch.as_tensor(values).float()
    if values.ndim != 1 or not torch.isfinite(values).all():
        raise ValueError('Expected one finite score per candidate.')
    if not len(values):
        return values
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)


def combine_scores(baseline, fusion, alpha):
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError('Residual weight must be finite and nonnegative.')
    baseline = torch.as_tensor(baseline).float()
    fusion = torch.as_tensor(fusion).float()
    if baseline.shape != fusion.shape:
        raise ValueError('Baseline and fusion scores must describe the same candidates.')
    return standardize(baseline) + alpha * standardize(fusion)


class ResidualReranker:
    def __init__(self, kind='moe', selection_path=SELECTION):
        if kind != 'moe':
            raise ValueError('Only the MoE reranker is supported.')
        self.kind = kind
        self.selection = json.loads(Path(selection_path).read_text())[kind]
        if self.selection['selection_fold'] != 'dev' or self.selection['test_used']:
            raise ValueError('Reranker must have been selected on development data only.')
        self.alpha = float(self.selection['alpha'])
        path = ROOT / self.selection['checkpoint']
        if sha256_file(path) != self.selection['checkpoint_sha256']:
            raise ValueError('Selected reranker checkpoint changed.')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if 'input_fingerprints' in payload:
            if payload.get('test_fold_used', True):
                raise ValueError('Selected checkpoint must declare test_fold_used=False.')
            for name in ('moe_fusion.py', 'checkpoints/moe_fusion.pt',
                         'research/training_pairs.json', 'research/split_manifest.json'):
                if sha256_file(ROOT / name) != payload['input_fingerprints'][name]:
                    raise ValueError(f'Selected training input changed: {name}')
            initial = torch.load(ROOT / 'checkpoints/moe_fusion.pt', map_location='cpu', weights_only=True)
            self.provenance = initial['provenance']
        else:
            # A future development search can select the initial MoE checkpoint.
            # Its inference still uses the same unrestricted combined-score rule.
            self.provenance = payload.get('provenance', {})
            if self.provenance.get('test_fold_used', True) or self.provenance.get('training_fold') != 'train':
                raise ValueError('MoE checkpoint needs documented train-only supervision.')
            expected = {
                'moe_fusion.py': self.provenance['code_sha256']['moe_fusion.py'],
                'research/split_manifest.json': self.provenance['split_sha256'],
                'research/training_pairs.json': self.provenance['training_pairs_sha256'],
            }
            for name, fingerprint in expected.items():
                if sha256_file(ROOT / name) != fingerprint:
                    raise ValueError(f'Selected training input changed: {name}')
        if payload['model_kind'] != kind:
            raise ValueError('Checkpoint kind does not match selection.')
        self.model = create_model(kind)
        self.model.load_state_dict(payload['state_dict'])
        self.model.eval()

    @torch.no_grad()
    def fusion_scores(self, features):
        qt, qi, dt, di = features
        if len(dt) == 0:
            return torch.empty(0)
        query = F.normalize(self.model(qt, qi), dim=-1)
        documents = F.normalize(self.model(dt, di), dim=-1)
        return (query @ documents.T)[0]

    def rank(self, candidates, features):
        if not candidates:
            return []
        ce = torch.tensor([d.metadata['cross_encoder_score'] for d in candidates])
        fusion = self.fusion_scores(features) if self.alpha else torch.zeros(len(candidates))
        combined = combine_scores(ce, fusion, self.alpha)
        order = self.order(ce, fusion, combined)
        result = []
        for j in order:
            d = candidates[j]
            result.append(Document(page_content=d.page_content, metadata={
                **d.metadata, 'ranking_score': float(combined[j]),
                'fusion_score': float(fusion[j]) if self.alpha else None,
                'baseline_rank': j + 1, 'residual_alpha': self.alpha,
                'ranking_policy': self.policy_name if self.alpha else 'cross_encoder_fallback',
            }))
        return result

    policy_name = 'ce_plus_moe_residual'

    def order(self, baseline, fusion, combined):
        return combined.argsort(descending=True, stable=True).tolist()
