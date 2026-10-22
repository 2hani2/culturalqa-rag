"""Recompute published metrics from saved predictions; no model downloads."""
import json
import math
from pathlib import Path
from validation_metrics import ranking_metrics, answer_metrics

ROOT = Path(__file__).resolve().parent
MODES = ('without_moe', 'with_moe')


def verify():
    report = json.loads((ROOT/'validation_moe_residual_120/validation_report.json').read_text())
    rows = {(r['query_id'],r['mode'],r['k']):r for r in report['per_question']}
    assert len(rows) == 480, 'Expected 400 text and 80 shared-image rows.'
    assert {r['mode'] for r in rows.values()} == set(MODES)
    checked = 0
    for raw in report['raw_results']:
        case, result = raw['case'], raw['result']
        assert not raw.get('error')
        assert set(result['conditions']) == set(MODES)
        for mode in MODES:
            c = result['conditions'][mode]
            labels = case.get('relevant_records') if case['task']=='text' else case.get('resolved_image_labels',case.get('relevant_images'))
            ranked = [r['ref'] for r in c['ranked']]
            metrics = ranking_metrics(ranked,labels)
            for k in (3,6):
                row = rows[(case['id'],mode,k)]
                assert row['status']=='ok'
                expected = {name:metrics[f'{key}@{k}'] for name,key in
                            [('hit_rate','hit'),('recall','recall'),('precision','precision'),('mrr','mrr'),('ndcg','ndcg')]}
                if case['task']=='text':
                    answer = c['answers'][str(k)]
                    assert not answer.get('reader_errors')
                    expected.update(answer_metrics(answer.get('answer'),case.get('expected_answers')))
                for name,value in expected.items():
                    matches = row[name] is None if value is None else math.isclose(row[name],value,abs_tol=1e-12)
                    assert matches, (case['id'],mode,k,name)
                    checked += 1
    for row in report['summary']:
        group = [r for r in rows.values() if r['task']==row['task'] and r['mode']==row['mode'] and r['k']==row['k']
                 and (row['modality']=='all' or r['modality']==row['modality'])
                 and (row['target_dataset']=='all' or r['target_dataset']==row['target_dataset'])]
        assert len(group)==row['queries']
        for metric in ('hit_rate','mrr','ndcg','exact_match','answer_f1'):
            values = [r[metric] for r in group if r.get(metric) is not None]
            if values:
                assert math.isclose(row[metric],sum(values)/len(values),abs_tol=1e-12)
                checked += 1
    paper = json.loads((ROOT/'paper/data/evaluation_extract.json').read_text())
    lookup={(r['task'],r['modality'],r['target_dataset'],r['mode'],r['k']):r for r in report['summary']}
    for r in paper['summary']:
        target=lookup[(r['task'],r['modality'],r['target_dataset'],r['mode'],r['k'])]
        for key in ('hit_rate','mrr','ndcg','exact_match','answer_f1'):
            assert r[key]==target[key]
    print(f'PASS: {checked} saved metrics verified; 100 text + 20 shared CLIP queries; exactly two conditions.')
    return checked


if __name__=='__main__':
    verify()
