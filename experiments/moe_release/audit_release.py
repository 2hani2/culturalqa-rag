"""Audit the repaired benchmark independently; does not change predictions."""
import argparse
from collections import Counter
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import string

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
QUALITY = ('hit_rate', 'recall', 'precision', 'mrr', 'ndcg', 'exact_match',
           'answer_f1', 'answer_evidence_hit')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def normalize(value):
    value = str(value or '').casefold().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', value).split())


def recompute(row):
    gold = set(row['relevant_refs'])
    found = set()
    flags = []
    for ref in row['ranked_refs']:
        flags.append(int(ref in gold and ref not in found))
        found.add(ref)
    k = row['k']
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(gold))))
    values = {'hit_rate': float(any(flags)), 'recall': sum(flags) / len(gold),
              'precision': sum(flags) / k,
              'mrr': next((1 / (i + 1) for i, hit in enumerate(flags) if hit), 0.),
              'ndcg': sum(hit / math.log2(i + 2) for i, hit in enumerate(flags)) / ideal}
    if row['task'] == 'text':
        pred = normalize(row['answer'])
        exact = f1 = 0.
        for answer in row['expected_answers']:
            answer = normalize(answer)
            exact = max(exact, float(pred == answer))
            p, a = pred.split(), answer.split()
            shared = sum((Counter(p) & Counter(a)).values())
            f1 = max(f1, 2 * shared / (len(p) + len(a)) if p and a else float(p == a))
        values.update(exact_match=exact, answer_f1=f1)
    return values


def audit_report(relative, historical):
    path = ROOT / relative / 'validation_report.json'
    report = json.loads(path.read_text())
    rows = report['per_question']
    assert len(rows) == 720 and len(report['raw_results']) == 120
    assert not any(row['status'] != 'ok' for row in rows)
    assert all(row['scope_coverage'] == 1. for row in rows)
    checked = 0
    for row in rows:
        for metric, expected in recompute(row).items():
            assert math.isclose(row[metric], expected, abs_tol=1e-12), (row['query_id'], metric)
            checked += 1
    summary_checked = 0
    for summary in report['summary']:
        group = [row for row in rows if all(row[key] == summary[key] for key in ('task', 'mode', 'k'))
                 and (summary['modality'] == 'all' or row['modality'] == summary['modality'])
                 and (summary['target_dataset'] == 'all' or row['target_dataset'] == summary['target_dataset'])]
        assert summary['queries'] == len(group) and summary['failed_queries'] == 0
        for metric in QUALITY + ('scope_coverage', 'retrieval_ms', 'fusion_features_ms', 'reranker_ms', 'reader_ms'):
            values = [row[metric] for row in group if row[metric] is not None]
            if values:
                assert math.isclose(summary[metric], sum(values) / len(values), abs_tol=1e-12)
            else:
                assert summary[metric] is None
            summary_checked += 1
    lookup = {(r['query_id'], r['mode'], r['k']): r for r in rows}
    old = {(r['query_id'], r['mode'], r['k']): r for r in historical['per_question']}
    mismatches = []
    for row in rows:
        base = lookup[row['query_id'], 'without_moe', row['k']]
        assert set(row['ranked_refs']) == set(base['ranked_refs'])
        assert row['answer'] == base['answer']
        for metric in QUALITY:
            assert row[metric] == base[metric], (row['query_id'], row['mode'], metric)
            if row['mode'] == 'without_moe' and row[metric] != old[row['query_id'], 'without_moe', row['k']][metric]:
                mismatches.append((row['query_id'], row['k'], metric))
    changes = {}
    for task in ('text', 'image'):
        cases = [x for x in report['raw_results'] if x['case']['task'] == task]
        counts = {'questions': len(cases), 'full_order_changes': 0, 'top3_order_changes': 0,
                  'top6_order_changes': 0, 'top3_set_changes': 0, 'top6_set_changes': 0}
        for raw in cases:
            assert not raw['error']
            result = raw['result']
            base = [x['ref'] for x in result['conditions']['without_moe']['ranked']]
            moe = [x['ref'] for x in result['conditions']['with_moe']['ranked']]
            assert set(base) == set(moe) and len(base) == len(moe) == 12
            counts['full_order_changes'] += int(base != moe)
            for k in (3, 6):
                counts[f'top{k}_order_changes'] += int(base[:k] != moe[:k])
                counts[f'top{k}_set_changes'] += int(set(base[:k]) != set(moe[:k]))
            if task == 'text':
                assert result['query_expansion'] is True
                assert result['expansion_cache_hit'] is True
            else:
                assert base == moe
                assert all(not x.get('moe_rescore_used') for x in result['conditions']['with_moe']['ranked'])
        changes[task] = counts
    summary = [row for row in report['summary'] if row['modality'] == row['target_dataset'] == 'all']
    return {'report': str(path.relative_to(ROOT)), 'report_sha256': sha(path),
            'execution': report['settings']['execution'], 'questions': 120, 'metric_rows': 720,
            'failed_rows': 0, 'independently_recomputed_values': checked,
            'independently_verified_summary_values': summary_checked,
            'all_conditions_match_baseline_quality': True, 'all_evidence_sets_preserved': True,
            'baseline_quality_changes_from_historical': mismatches, 'ranking_changes': changes,
            'summary': summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Also require the completed fresh 120-question run.')
    args = parser.parse_args()
    plan = json.loads((HERE / 'release_plan.json').read_text())
    immutable = []
    for name, expected in plan['source_hashes'].items():
        if name in ('main.py', 'main_with_moe.py'):
            assert sha(HERE / 'before' / name) == expected
        else:
            assert sha(ROOT / name) == expected, name
            immutable.append(name)
    selections = json.loads((HERE / 'conservative_selection.json').read_text())
    for value in selections.values():
        assert sha(ROOT / value['checkpoint']) == value['checkpoint_sha256']
        assert value['selection_fold'] == 'dev' and value['test_used'] is False
    historical = json.loads((ROOT / 'validation_large_120/validation_report.json').read_text())
    results = {'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'immutable_inputs_verified': immutable, 'checkpoint_fingerprints_verified': True,
               'selection': selections,
               'replay': audit_report('validation_moe_conservative_120', historical),
               'fresh_full_validation_checked': args.live}
    if args.live:
        results['live'] = audit_report('validation_conservative_live_120', historical)
    source_names = ['main.py', 'main_with_moe.py', 'main_without_moe.py', 'residual_pipeline.py',
                    'residual_reranker.py', 'validation_conservative.py', 'test_residual.py',
                    'experiments/moe_release/audit_release.py', 'experiments/moe_release/evaluate_saved.py']
    results['release_source_sha256'] = {name: sha(ROOT / name) for name in source_names}
    (HERE / 'release_verification.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in results.items() if key in
                     ('created_at', 'checkpoint_fingerprints_verified', 'fresh_full_validation_checked')}))
    for kind in ('replay', 'live'):
        if kind in results:
            print(kind, results[kind]['questions'], 'questions;', results[kind]['failed_rows'], 'errors;',
                  results[kind]['independently_recomputed_values'], 'metric values verified')


if __name__ == '__main__':
    main()
