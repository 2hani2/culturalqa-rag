"""Development-only listwise fusion experiment; never reads the 120-question suite.

Run from the project root with ./venv/bin/python experiments/moe_improvement/run_experiment.py.
All new artifacts stay beside this script. The initial MoE checkpoint is an immutable input.
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
from reranker_models import create_model, load_model, sha256_file, source_hash
from research_protocol import canonical, digest, load_records, record_ref
from structured_store import record_to_text

PLAN = {
    'purpose': 'Development-only diagnosis and candidate model selection. No test claims.',
    'training_data': 'Existing 300 weak source-derived train anchors; 60 development anchors.',
    'negative_policy': 'Every different-group, different-question positive record in the same fold is a weak negative. Training uses all eligible negatives in a listwise loss.',
    'diagnostic_candidate_policy': 'Positive injected plus top 11 eligible frozen-BERT-cosine negatives, separately within train/dev. This evaluates conditional reranking, NOT end-to-end retrieval.',
    'epochs': 20, 'learning_rate': 0.0001, 'seed': 42,
    'temperatures': [0.05, 0.1], 'initializations': ['existing_checkpoint', 'fresh'],
    'model_kinds': ['moe'], 'load_balance_weight': 0.01,
    'residual_alphas': [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
    'residual_formula': 'z(cross_encoder_score) + alpha * z(fusion_cosine), per query over identical candidates.',
    'selection': 'Highest development MRR@12, then Hit@3; ties prefer alpha=0 and earlier epochs. Separate best configuration reported per model kind.',
    'baseline': 'Unchanged pretrained cross-encoder/ms-marco-MiniLM-L-6-v2 on exactly the same candidates.',
    'test_data_accessed': False, 'query_expansion': False,
    'cpu_threads': 2,
}


def dump(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def metrics(scores):
    # The sole known positive is position 0. Break exact ties with a seeded,
    # label-independent permutation so a constant model cannot exploit position.
    tie_order = torch.rand(scores.shape, generator=torch.Generator().manual_seed(31415)).argsort(dim=-1)
    order = tie_order.gather(1, torch.argsort(scores.gather(1, tie_order), dim=-1, descending=True, stable=True))
    ranks = (order == 0).nonzero()[:, 1] + 1
    return {'hit_at_1': float((ranks <= 1).float().mean()),
            'hit_at_3': float((ranks <= 3).float().mean()),
            'hit_at_6': float((ranks <= 6).float().mean()),
            'mrr_at_12': float((1.0 / ranks).mean()),
            'ranks': ranks.tolist()}


def zscore(scores):
    return (scores - scores.mean(-1, keepdim=True)) / scores.std(-1, keepdim=True, unbiased=False).clamp_min(1e-6)


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    dump('experiment_plan.json', {**PLAN, 'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
    original_hashes = source_hash()
    original_hashes.update({name: sha256_file(ROOT / name) for name in (
        'checkpoints/moe_fusion.pt',
        'research/training_pairs.json', 'research/split_manifest.json')})
    pairs = json.loads((ROOT / 'research/training_pairs.json').read_text())
    split = json.loads((ROOT / 'research/split_manifest.json').read_text())
    provenance = json.loads((ROOT / 'checkpoints/moe_fusion.provenance.json').read_text())
    assert provenance['training_pairs_sha256'] == original_hashes['research/training_pairs.json']
    assert provenance['split_sha256'] == original_hashes['research/split_manifest.json']
    settings = {k: provenance[k] for k in ('seed', 'epochs', 'batch_size', 'learning_rate', 'margin', 'enable_images', 'train_per_dataset', 'dev_per_dataset', 'split_sha256', 'training_pairs_sha256', 'code_sha256')}
    feature_path = ROOT / 'research' / ('training_features_' + digest(json.dumps(settings, sort_keys=True))[:16] + '.pt')
    features = torch.load(feature_path, map_location='cpu', weights_only=True)
    assert feature_path.is_file()
    dataset = {}
    for fold in ('train', 'dev'):
        examples = pairs[fold]
        refs = [p['positive_ref'] for p in examples]
        groups = [split['assignments'][ref]['group'] for ref in refs]
        questions = [canonical(p['query']) for p in examples]
        assert len(set(refs)) == len(refs)
        assert all(split['assignments'][ref]['fold'] == fold for ref in refs)
        n = len(refs)
        eligible = torch.tensor([[i == j or (groups[i] != groups[j] and questions[i] != questions[j]) for j in range(n)] for i in range(n)])
        raw = F.normalize(features[fold]['qt'], dim=-1) @ F.normalize(features[fold]['pt'], dim=-1).T
        raw = raw.masked_fill(~eligible | torch.eye(n, dtype=torch.bool), -torch.inf)
        negatives = raw.topk(11, dim=-1).indices
        candidates = torch.cat([torch.arange(n)[:, None], negatives], dim=-1)
        dataset[fold] = {'refs': refs, 'groups': groups, 'eligible': eligible, 'candidates': candidates}
    assert not set(dataset['train']['groups']) & set(dataset['dev']['groups'])
    dump('candidate_manifest.json', {fold: [
        {'query': p['query'], 'positive_ref': p['positive_ref'], 'group': dataset[fold]['groups'][i],
         'candidate_refs': [dataset[fold]['refs'][j] for j in dataset[fold]['candidates'][i].tolist()]}
        for i, p in enumerate(pairs[fold])] for fold in ('train', 'dev')})
    fingerprint = {**original_hashes, str(feature_path.relative_to(ROOT)): sha256_file(feature_path),
                   'experiment_script_sha256': sha256_file(Path(__file__))}
    dump('input_fingerprints.json', fingerprint)
    print('[data] 300 train / 60 dev; source-group disjoint; test not accessed', flush=True)

    ce_cache = HERE / 'cross_encoder_scores.pt'
    candidate_hash = sha256_file(HERE / 'candidate_manifest.json')
    if ce_cache.exists():
        cached = torch.load(ce_cache, weights_only=True)
        assert cached['candidate_sha256'] == candidate_hash
        ce = cached['scores']
    else:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu', local_files_only=True)
        lookup = {record_ref(r): r for r in load_records()}
        ce = {}
        for fold in ('train', 'dev'):
            input_pairs = [(p['query'], record_to_text(lookup[dataset[fold]['refs'][j]])[:2000])
                           for i, p in enumerate(pairs[fold]) for j in dataset[fold]['candidates'][i].tolist()]
            ce[fold] = torch.as_tensor(model.predict(input_pairs, batch_size=32, show_progress_bar=True)).reshape(-1, 12)
        torch.save({'scores': ce, 'candidate_sha256': candidate_hash}, ce_cache)
        del model, lookup
    ce_z = zscore(ce['dev'])
    history = []
    baseline = metrics(ce['dev'])
    dump('cross_encoder_dev_metrics.json', baseline)
    print('[baseline dev]', {k: v for k, v in baseline.items() if k != 'ranks'}, flush=True)

    def score(model, fold):
        data = features[fold]
        q = F.normalize(model(data['qt'], data['qi']), dim=-1)
        d = F.normalize(model(data['pt'], data['pi']), dim=-1)
        return (q @ d.T).gather(1, dataset[fold]['candidates'])

    def evaluate(model, label, epoch, extra=None):
        model.eval()
        with torch.no_grad():
            dev_scores = score(model, 'dev')
            fusion = metrics(dev_scores)
            records = []
            for alpha in PLAN['residual_alphas']:
                measured = metrics(ce_z + alpha * zscore(dev_scores))
                records.append({'label': label, 'epoch': epoch, 'alpha': alpha, **measured})
            row = {'label': label, 'epoch': epoch, 'fusion_only': fusion, 'residuals': records, **(extra or {})}
            history.append(row)
        return row

    for kind in PLAN['model_kinds']:
        existing, _ = load_model(kind)
        evaluate(existing, 'existing_' + kind, 0)
        del existing
    trials = []
    for kind in PLAN['model_kinds']:
        for initialization in PLAN['initializations']:
            for temperature in PLAN['temperatures']:
                label = f'{kind}_{initialization}_tau{temperature}'
                model = load_model(kind)[0] if initialization == 'existing_checkpoint' else create_model(kind, 42)
                optimizer = torch.optim.AdamW(model.parameters(), lr=PLAN['learning_rate'], weight_decay=0.01)
                best_key = None
                best_record = None
                best_fusion_key = None
                for epoch in range(1, PLAN['epochs'] + 1):
                    model.train()
                    optimizer.zero_grad()
                    data = features['train']
                    q = F.normalize(model(data['qt'], data['qi']), dim=-1)
                    q_aux = model.load_balancing_loss() if kind == 'moe' else torch.tensor(0.)
                    d = F.normalize(model(data['pt'], data['pi']), dim=-1)
                    d_aux = model.load_balancing_loss() if kind == 'moe' else torch.tensor(0.)
                    logits = (q @ d.T / temperature).masked_fill(~dataset['train']['eligible'], -torch.inf)
                    loss = F.cross_entropy(logits, torch.arange(len(pairs['train']))) + PLAN['load_balance_weight'] * (q_aux + d_aux) / 2
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    row = evaluate(model, label, epoch, {'train_loss': float(loss.detach())})
                    fusion_key = (row['fusion_only']['mrr_at_12'], row['fusion_only']['hit_at_3'])
                    if best_fusion_key is None or fusion_key > best_fusion_key:
                        best_fusion_key = fusion_key
                        torch.save({'state_dict': model.state_dict(), 'model_kind': kind,
                                    'selection': {'label': label, 'epoch': epoch, **row['fusion_only']},
                                    'selection_metric': 'development fusion-only MRR@12',
                                    'initialization': initialization, 'temperature': temperature,
                                    'input_fingerprints': fingerprint, 'test_fold_used': False,
                                    'experiment_plan': PLAN}, HERE / (label + '_fusion.pt'))
                    selected = max(row['residuals'], key=lambda r: (r['mrr_at_12'], r['hit_at_3'], -r['alpha']))
                    key = (selected['mrr_at_12'], selected['hit_at_3'], -selected['alpha'])
                    if best_key is None or key > best_key:
                        best_key, best_record = key, selected
                        torch.save({'state_dict': model.state_dict(), 'model_kind': kind,
                                    'selection': selected, 'initialization': initialization,
                                    'temperature': temperature, 'input_fingerprints': fingerprint,
                                    'test_fold_used': False, 'experiment_plan': PLAN}, HERE / (label + '.pt'))
                    if epoch in (1, 5, 10, 15, 20):
                        print(f'[{label} epoch {epoch}] loss={float(loss):.4f} fusion_mrr={row["fusion_only"]["mrr_at_12"]:.4f} best_residual_mrr={selected["mrr_at_12"]:.4f} alpha={selected["alpha"]}', flush=True)
                trials.append({'kind': kind, 'label': label, 'selected': best_record})
                dump('all_trials.json', {'cross_encoder_baseline': baseline, 'trials': trials, 'history': history})
    after = {name: sha256_file(ROOT / name) for name in original_hashes}
    assert after == original_hashes, 'Immutable input changed during experiment'
    selected = {kind: max((t for t in trials if t['kind'] == kind), key=lambda t: (
        t['selected']['mrr_at_12'], t['selected']['hit_at_3'], -t['selected']['alpha'])) for kind in PLAN['model_kinds']}
    summary = {'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'cross_encoder_baseline': baseline, 'selected_development_trials': selected,
        'original_files_unchanged': True, 'test_fold_used': False, 'trial_count': len(trials),
        'epochs_evaluated': len(trials) * PLAN['epochs'], 'elapsed_seconds': time.perf_counter() - started,
        'limitations': ['Development-selected, no independent test evaluated.',
            'Only 300 weakly labelled training anchors; false negatives possible.',
            'Positive is injected into the 12-item diagnostic shortlist: these are conditional reranking metrics, not global retrieval metrics.',
            'No API expansion, reader evaluation, or image-only question task in this development experiment.',
            'Any alpha=0 selected configuration equals the cross-encoder baseline and is not evidence of an MoE gain.']}
    dump('experiment_summary.json', summary)
    print('[completed]', json.dumps({k: {m: v for m, v in t['selected'].items() if m != 'ranks'} for k, t in selected.items()}), flush=True)


if __name__ == '__main__':
    main()
