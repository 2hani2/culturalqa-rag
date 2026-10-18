"""Development-selected MoE correction to the existing relevance score.

This module never uses reference answers or relevance labels. The historical
fusion-only models and pipeline remain available for reproducibility.
"""
import json
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from langchain_core.documents import Document
from project_paths import ROOT
from reranker_models import create_model, sha256_file

SELECTION = ROOT / 'experiments/moe_improvement/global_selection.json'
CONSERVATIVE_SELECTION = ROOT / 'experiments/moe_release/conservative_selection.json'


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


def conservative_order(baseline, fusion, alpha, max_span, protected_cutoffs=(3,6)):
    """Reorder near ties while preserving the baseline evidence at each cutoff.

    Bands have bounded total score range, rather than chaining adjacent ties
    into a large band. This is an inference rule, not a relevance-label filter.
    """
    if not math.isfinite(max_span) or max_span < 0:
        raise ValueError('Near-tie span must be finite and nonnegative.')
    base = standardize(baseline)
    scores = combine_scores(baseline, fusion, alpha)
    order = base.argsort(descending=True, stable=True).tolist()
    result = []
    boundaries = sorted(set([0,len(order)] + [k for k in protected_cutoffs if 0 < k < len(order)]))
    for left,right in zip(boundaries,boundaries[1:]):
        start = left
        while start < right:
            end = start + 1
            while end < right and float(base[order[start]]-base[order[end]]) <= max_span:
                end += 1
            band = order[start:end]
            result.extend(sorted(band, key=lambda j: -float(scores[j])))
            start = end
    return result


class ResidualReranker:
    def __init__(self, kind='moe', selection_path=SELECTION):
        self.kind = kind
        self.selection = json.loads(Path(selection_path).read_text())[kind]
        if self.selection['selection_fold'] != 'dev' or self.selection['test_used']:
            raise ValueError('Reranker must have been selected on development data only.')
        self.alpha = float(self.selection['alpha'])
        path = ROOT / self.selection['checkpoint']
        if sha256_file(path) != self.selection['checkpoint_sha256']:
            raise ValueError('Selected reranker checkpoint changed.')
        payload = torch.load(path, map_location='cpu', weights_only=True)
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


class ConservativeReranker(ResidualReranker):
    policy_name = 'baseline_evidence_moe_near_ties'

    def __init__(self, kind='moe', selection_path=CONSERVATIVE_SELECTION):
        super().__init__(kind, selection_path)
        self.max_span = float(self.selection['max_span'])
        self.protected_cutoffs = tuple(self.selection['protected_cutoffs'])
        if self.protected_cutoffs != (3,6):
            raise ValueError('This release protects the declared top-3 and top-6 evidence budgets.')

    def order(self, baseline, fusion, combined):
        return conservative_order(baseline, fusion, self.alpha, self.max_span, self.protected_cutoffs)
