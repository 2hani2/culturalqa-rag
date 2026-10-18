"""Evaluate the frozen development selection using identical saved candidates.

Reuses recorded retrieval and reader outputs. It does not make API calls,
retrieve a different pool, fit weights, or choose parameters from test scores.
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import copy
import argparse
import datetime
import json
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import torch
from langchain_core.documents import Document
from encoders import MultimodalEncoder
from reader import ExtractiveReader
from residual_reranker import ResidualReranker, ConservativeReranker, CONSERVATIVE_SELECTION
from reranker_models import sha256_file
from validation import evaluate_cases, write_report
from validation_metrics import evidence_rows, evidence_ref


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conservative',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2)
    plan = json.loads((HERE / 'release_plan.json').read_text())
    source_path = ROOT / 'validation_large_120/validation_report.json'
    for name in ('validation_large_120/validation_report.json', 'validation_queries_120.json',
                 'experiments/moe_improvement/global_selection.json', 'research/split_manifest.json',
                 'research/training_pairs.json'):
        assert sha256_file(ROOT / name) == plan['source_hashes'][name], name
    source = json.loads(source_path.read_text())
    model_type = ConservativeReranker if args.conservative else ResidualReranker
    rerankers = {kind: model_type(kind) for kind in ('simple', 'moe')}
    cache_path = HERE / 'evaluation_features.pt'
    key = {'source_report': sha256_file(source_path), 'encoder_code': sha256_file(ROOT / 'encoders.py'),
           'faiss_docstore': sha256_file(ROOT / 'faiss_store/index.pkl')}
    cache = torch.load(cache_path, weights_only=True) if cache_path.exists() else {'key': key, 'features': {}}
    assert cache['key'] == key
    text_cases = [x for x in source['raw_results'] if x['case']['task'] == 'text']
    needed = {(d['ref'], raw['result']['candidate_texts'][d['ref']])
              for raw in text_cases for d in raw['result']['conditions']['without_moe']['ranked']}
    # Trusted local FAISS docstore; verify its hash in the cache key.
    with (ROOT / 'faiss_store/index.pkl').open('rb') as f:
        store, _ = pickle.load(f)
    metadata = {(evidence_ref(d), d.page_content): d.metadata for d in store._dict.values()
                if (evidence_ref(d), d.page_content) in needed}
    assert set(metadata) == needed, 'Recorded evaluation context differs from current docstore.'
    del store
    encoder = None
    predictions = {}
    for number, raw in enumerate(source['raw_results'], 1):
        case, old = raw['case'], raw['result']
        result = copy.deepcopy(old)
        baseline = old['conditions']['without_moe']
        result['conditions'] = {'without_moe': copy.deepcopy(baseline)}
        if case['task'] == 'image':
            for mode in ('simple_fusion', 'with_moe'):
                result['conditions'][mode] = copy.deepcopy(baseline)
                result['conditions'][mode].update(ranking_policy='clip_fallback',
                                                  fusion_features_ms=0., reranker_ms=0.)
        else:
            docs = []
            for row in baseline['ranked']:
                body = old['candidate_texts'][row['ref']]
                docs.append(Document(page_content=body, metadata={**metadata[(row['ref'], body)],
                    'cross_encoder_score': row['cross_encoder_score']}))
            if case['id'] not in cache['features']:
                if encoder is None:
                    encoder = MultimodalEncoder(enable_live_fetch=True)
                    encoder.image_encoder.save_cache = lambda: None
                start = time.perf_counter()
                qt = encoder.text_encoder.encode(case['question'])
                dt = encoder.text_encoder.encode([d.page_content for d in docs])
                images = []
                for d in docs:
                    ref = d.metadata.get('image_url') if d.metadata.get('has_image') else None
                    if ref and ref.startswith('http') and encoder.image_encoder._cache_key(ref) not in encoder.image_encoder._cache:
                        raise ValueError('Uncached remote image is not allowed in saved-candidate replay.')
                    images.append(encoder.image_encoder.encode(ref)[0])
                # Clone CLS views so serialization does not retain full hidden-state storage.
                cache['features'][case['id']] = {'values': tuple(t.detach().clone() for t in
                                                (qt, torch.zeros(1,512), dt, torch.stack(images))),
                                                'feature_ms': (time.perf_counter()-start)*1000}
                if len(cache['features']) % 5 == 0:
                    torch.save(cache, cache_path)
            features = cache['features'][case['id']]
            reader_by_ref = {r['ref']: r for r in old['reader_context_results']}
            for kind, mode in [('simple', 'simple_fusion'), ('moe', 'with_moe')]:
                start = time.perf_counter()
                ranked = rerankers[kind].rank(docs, features['values'])
                rank_ms = (time.perf_counter()-start)*1000
                result['conditions'][mode] = {
                    'ranked': evidence_rows(ranked),
                    'answers': {str(k): ExtractiveReader.select_answer([reader_by_ref[evidence_ref(d)] for d in
                        (sorted(ranked[:k],key=lambda d:d.metadata['baseline_rank']) if args.conservative else ranked[:k])]) for k in (3,6)},
                    'fusion_features_ms': features['feature_ms'] if rerankers[kind].alpha else 0.,
                    'reranker_ms': rank_ms if rerankers[kind].alpha else 0.,
                    'ranking_policy': rerankers[kind].policy_name if rerankers[kind].alpha else 'cross_encoder_fallback',
                    'alpha': rerankers[kind].alpha,
                }
        predictions[case['question']] = result
        if number % 10 == 0:
            print(f'[frozen replay] {number}/120', flush=True)
    torch.save(cache, cache_path)
    # Freeze all predictions before reference-based metrics are calculated.
    pred_path = HERE / ('conservative_predictions.json' if args.conservative else 'frozen_predictions.json')
    pred_path.write_text(json.dumps(predictions, ensure_ascii=False))
    cases = [x['case'] for x in source['raw_results']]
    report = evaluate_cases(cases, lambda q: predictions[q], lambda q: predictions[q])
    report.update({k: copy.deepcopy(v) for k,v in source.items() if k not in report})
    report['created_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report['settings']['execution'] = 'Replay identical recorded candidates, phrasings and reader outputs; recompute text fusion.'
    report['settings']['image_ranking_policy'] = 'CLIP baseline fallback; no MoE image gain claimed.'
    report['settings']['text_ranking_policy'] = ('Development-selected near-tie reranking with baseline top-3/top-6 evidence protected.'
        if args.conservative else 'Development-selected CE plus MoE residual, alpha=0.25.')
    report['historical_checkpoints'] = report.pop('checkpoints')
    report['checkpoints'] = {k:v.selection for k,v in rerankers.items()}
    report['provenance'] = {'release_plan': plan, 'predictions_sha256': sha256_file(pred_path),
                            'feature_key': key, 'historical_report_sha256': sha256_file(source_path)}
    if args.conservative:
        report['provenance']['conservative_selection_sha256']=sha256_file(CONSERVATIVE_SELECTION)
        report['provenance']['conservative_plan']=json.loads((HERE/'conservative_plan.json').read_text())
    report['limitations'].extend([
        'Reused benchmark previously inspected for diagnosis; not a new blind test.',
        'No parameters selected from these evaluation answers; all predictions frozen before metrics.',
        'Image condition uses explicit CLIP fallback. Any selected alpha-zero text condition is also a baseline fallback.',
        'Retrieval and reader timings are historical components; this is not a new end-to-end latency run.',
    ])
    if args.conservative:
        report['limitations'].append('Top-3/top-6 evidence membership and reader tie handling are preserved by design; matching answer accuracy is not evidence of MoE answering benefit.')
    output = ROOT / ('validation_moe_conservative_120' if args.conservative else 'validation_moe_residual_120')
    write_report(report, output)
    for row in report['summary']:
        if row['modality']=='all' and row['target_dataset']=='all':
            print('[result]', {k:row[k] for k in ('task','mode','k','hit_rate','mrr','ndcg','exact_match','answer_f1')}, flush=True)


if __name__ == '__main__':
    main()
