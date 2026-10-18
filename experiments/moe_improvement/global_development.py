"""Evaluate residual rerankers on global retrieval for development queries only.

No gold records are injected. The original 120-question benchmark is never read.
Use --ask TEXT for the selected experimental model; historical CLI stays intact.
"""
from run_experiment import HERE, ROOT, PLAN, dump, zscore
import argparse
import json
import time
import torch
from torch.nn import functional as F
from reranker_models import create_model, load_model, sha256_file, source_hash
from research_protocol import record_ref
from retrieval import load_faiss_index, GlobalRetriever
from encoders import MultimodalEncoder
from validation_metrics import evidence_ref


def initialize():
    store = load_faiss_index()
    chunks = [store.docstore.search(key) for key in store.index_to_docstore_id.values()]
    retriever = GlobalRetriever(store, chunks, candidate_k=12, branch_k=8, use_query_expansion=False)
    encoder = MultimodalEncoder(enable_live_fetch=True)
    # New calculations may be cached in memory but never modify the main cache.
    encoder.image_encoder.save_cache = lambda: None
    return retriever, encoder


def encode(encoder, query, documents):
    image_vectors = []
    for doc in documents:
        ref = doc.metadata.get('image_url') if doc.metadata.get('has_image') else None
        if ref and ref.startswith('http') and encoder.image_encoder._cache_key(ref) not in encoder.image_encoder._cache:
            raise ValueError('Offline experiment cannot fetch an uncached remote image.')
        image_vectors.append(encoder.image_encoder.encode(ref)[0])
    return {'qt': encoder.text_encoder.encode(query), 'qi': torch.zeros(1, 512),
            'dt': encoder.text_encoder.encode([d.page_content for d in documents]),
            'di': torch.stack(image_vectors)}


def score(model, features):
    with torch.no_grad():
        query = F.normalize(model(features['qt'], features['qi']), dim=-1)
        docs = F.normalize(model(features['dt'], features['di']), dim=-1)
        return (query @ docs.T)[0]


def metrics(scores, records):
    ranks = []
    for row, values in zip(records, scores):
        order = values.argsort(descending=True, stable=True).tolist()
        ordered = [row['refs'][j] for j in order]
        ranks.append(ordered.index(row['positive_ref']) + 1 if row['positive_ref'] in ordered else None)
    n = len(ranks)
    return {'count': n,
            'hit_at_1': sum(r is not None and r <= 1 for r in ranks) / n,
            'hit_at_3': sum(r is not None and r <= 3 for r in ranks) / n,
            'hit_at_6': sum(r is not None and r <= 6 for r in ranks) / n,
            'hit_at_12': sum(r is not None and r <= 12 for r in ranks) / n,
            'mrr_at_12': sum(1/r if r else 0 for r in ranks) / n,
            'ranks': ranks}


def load_experiment(path):
    payload = torch.load(path, weights_only=True, map_location='cpu')
    model = create_model(payload['model_kind'])
    model.load_state_dict(payload['state_dict'])
    model.eval()
    return model, payload


def evaluate():
    plan = {'selection_fold': 'dev', 'test_used': False, 'query_count': 60,
            'global_candidate_k': 12, 'branch_k': 8, 'query_expansion': False,
            'image_features_enabled': True, 'positive_injection': False,
            'candidate_pool_shared': True, 'all_five_datasets_searched': True,
            'models': 'Existing MoE/simple plus each of 8 listwise trials selected by fusion-only and residual diagnostic development metrics.',
            'residual_alphas': PLAN['residual_alphas'],
            'selection': 'Maximum dev MRR@12, then Hit@3, then prefer smaller alpha. Dev was used for candidate training selection too; no unbiased test claim.'}
    dump('global_development_plan.json', plan)
    pairs = json.loads((ROOT / 'research/training_pairs.json').read_text())
    split = json.loads((ROOT / 'research/split_manifest.json').read_text())
    assert all(split['assignments'][r['positive_ref']]['fold'] == 'dev' for r in pairs['dev'])
    input_hashes = {name: sha256_file(ROOT/name) for name in (
        'research/training_pairs.json', 'research/split_manifest.json',
        'faiss_store/index.faiss', 'faiss_store/index.pkl')}
    cache_path = HERE / 'global_development_features.pt'
    if cache_path.exists():
        cache = torch.load(cache_path, weights_only=True)
        assert cache['input_hashes'] == input_hashes
        assert cache['source_hashes'] == source_hash()
    else:
        cache = {'input_hashes': input_hashes, 'source_hashes': source_hash(), 'records': []}
    records = cache['records']
    if len(records) < len(pairs['dev']):
        retriever, encoder = initialize()
        for i, example in enumerate(pairs['dev']):
            if i < len(records):
                assert records[i]['query'] == example['query']
                continue
            started = time.perf_counter()
            docs = retriever.invoke(example['query'])
            features = encode(encoder, example['query'], docs)
            refs = [evidence_ref(d) for d in docs]
            records.append({'query': example['query'], 'positive_ref': example['positive_ref'],
                'refs': refs, 'features': features,
                'candidate_texts': [d.page_content for d in docs],
                'cross_encoder_scores': torch.tensor([d.metadata['cross_encoder_score'] for d in docs]),
                'trace': retriever.last_trace, 'retrieval_and_encoding_seconds': time.perf_counter()-started})
            torch.save(cache, cache_path)
            print(f'[global dev] {i+1}/{len(pairs["dev"])} known positive in pool={example["positive_ref"] in refs}', flush=True)
        del retriever, encoder
    baseline_scores = [r['cross_encoder_scores'] for r in records]
    baseline = metrics(baseline_scores, records)
    ce_z = [zscore(s[None])[0] for s in baseline_scores]
    all_models = [('existing_'+kind, ROOT/'checkpoints'/f'{kind}_fusion.pt') for kind in ('simple', 'moe')]
    all_models += [(p.stem, p) for p in sorted(HERE.glob('*_tau*.pt'))]
    trials = []
    for label, path in all_models:
        model, payload = load_experiment(path)
        cosine = [score(model, r['features']) for r in records]
        fusion = metrics(cosine, records)
        normalized = [zscore(s[None])[0] for s in cosine]
        settings = []
        for alpha in PLAN['residual_alphas']:
            measured = metrics([ce+alpha*f for ce, f in zip(ce_z, normalized)], records)
            settings.append({'alpha': alpha, **measured})
        best = max(settings, key=lambda r: (r['mrr_at_12'], r['hit_at_3'], -r['alpha']))
        trials.append({'label': label, 'kind': payload['model_kind'],
            'checkpoint': str(path.relative_to(ROOT)), 'checkpoint_sha256': sha256_file(path),
            'fusion_only': fusion, 'residuals': settings, 'best': best})
        print('[global model]', label, {k:v for k,v in best.items() if k!='ranks'}, flush=True)
    selected = {kind: max((t for t in trials if t['kind'] == kind),
                         key=lambda t: (t['best']['mrr_at_12'], t['best']['hit_at_3'], -t['best']['alpha']))
                for kind in ('simple', 'moe')}
    summary = {'plan': plan, 'cross_encoder_baseline': baseline, 'trials': trials,
               'selected': selected, 'features_cache_sha256': sha256_file(cache_path),
               'input_hashes': input_hashes,
               'no_test_claim': 'These are development-selected results on weak source-derived queries. An independent verified test is still needed.',
               'reader_evaluated': False, 'image_only_retrieval_evaluated': False}
    dump('global_development_results.json', summary)
    dump('global_selection.json', {kind: {'label': t['label'], 'checkpoint': t['checkpoint'],
         'checkpoint_sha256': t['checkpoint_sha256'], 'alpha': t['best']['alpha'],
         'selection_fold': 'dev', 'test_used': False} for kind, t in selected.items()})
    dump('global_development_predictions.json', [
        {k:v for k,v in r.items() if k not in ('features','cross_encoder_scores')} |
        {'cross_encoder_scores': r['cross_encoder_scores'].tolist()} for r in records])


def ask(question, kind='moe', top_k=3):
    selection = json.loads((HERE/'global_selection.json').read_text())[kind]
    path = ROOT/selection['checkpoint']
    assert sha256_file(path) == selection['checkpoint_sha256']
    retriever, encoder = initialize()
    docs = retriever.invoke(question)
    features = encode(encoder, question, docs)
    model, _ = load_experiment(path)
    ce = torch.tensor([d.metadata['cross_encoder_score'] for d in docs])
    fused = score(model, features)
    values = zscore(ce[None])[0] + selection['alpha']*zscore(fused[None])[0]
    order = values.argsort(descending=True, stable=True).tolist()
    chosen = [docs[j] for j in order[:top_k]]
    from reader import ExtractiveReader
    answer = ExtractiveReader().answer(question, chosen)
    print('[experimental model]', selection['label'], 'residual weight', selection['alpha'])
    print('[development selection only; independent test pending]')
    print('[answer]', answer.get('answer'))
    print('[search]', ', '.join(retriever.datasets_searched))
    for rank, j in enumerate(order[:top_k], 1):
        print(f'[context] {rank}: {evidence_ref(docs[j])} | score={float(values[j]):.6f}')


if __name__ == '__main__':
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ask')
    parser.add_argument('--kind', choices=['moe', 'simple'], default='moe')
    parser.add_argument('--top_k', type=int, choices=[3, 6], default=3)
    args = parser.parse_args()
    if args.ask:
        ask(args.ask, args.kind, args.top_k)
    else:
        evaluate()
