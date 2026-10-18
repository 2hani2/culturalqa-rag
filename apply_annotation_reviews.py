"""Import a named human's completed review into a separate evaluation suite.

Example: ./venv/bin/python apply_annotation_reviews.py --reviews research/annotation_review.csv
Rows left human_reviewed=False do not change labels. Retain source evidence and
gold targets; use additional_relevant_refs for other verified positive evidence.
Set relevance_complete=True only after judging all relevant corpus candidates.
"""
import argparse,copy,csv,json
from pathlib import Path
from project_paths import ROOT
from dataset_config import ACTIVE_DATASETS

def boolean(value):
    value=str(value or '').strip().casefold()
    if value in ('true','yes','1'):return True
    if value in ('false','no','0',''):return False
    raise ValueError(f'Use true/false for review flags, received {value!r}')

def labels(value):
    value=str(value or '').strip()
    if not value:return []
    result=json.loads(value) if value.startswith('[') else [x.strip() for x in value.split(';') if x.strip()]
    if not isinstance(result,list) or not all(isinstance(x,str) and x.strip() for x in result):
        raise ValueError('Labels must be a JSON list of nonempty strings, or separated by semicolons.')
    return list(dict.fromkeys(result))

def apply_reviews(suite,rows,known_records):
    result=copy.deepcopy(suite);cases=result['queries'] if isinstance(result,dict) else result
    lookup={q['id']:q for q in cases};seen=set()
    for row in rows:
        key=row.get('query_id','').strip()
        if key not in lookup or key in seen:raise ValueError(f'Unknown or duplicate query_id: {key}')
        seen.add(key);q=lookup[key]
        if row.get('question',q['question']).strip()!=q['question'].strip():
            raise ValueError(f'{key}: changed question requires a new independently checked case.')
        if not boolean(row.get('human_reviewed')):continue
        reviewer=row.get('reviewer','').strip()
        if not reviewer:raise ValueError(f'{key}: human_reviewed requires a reviewer name.')
        accepted=labels(row.get('accepted_answers'))
        if q['task']=='text' and not accepted:raise ValueError(f'{key}: supply accepted answer aliases.')
        if q['task']=='image' and accepted:raise ValueError(f'{key}: image retrieval has no answer-accuracy label.')
        additions=labels(row.get('additional_relevant_refs'))
        if q['task']=='text':
            if any(ref not in known_records or ref.split('#')[0] not in ACTIVE_DATASETS for ref in additions):
                raise ValueError(f'{key}: additional evidence must be an existing active dataset#record_id.')
            field='relevant_records'
        else:
            if any(not (ROOT/ref).is_file() or not str(ref).startswith('data/raw/DRISHTIKON/') for ref in additions):
                raise ValueError(f'{key}: additional images must exist in the active local image folder.')
            field='relevant_images'
        q[field]=list(dict.fromkeys(q.get(field,[])+additions))
        q.update(expected_answers=accepted,human_reviewed=True,reviewer=reviewer,
                 relevance_complete=boolean(row.get('relevance_complete')),review_notes=row.get('notes','').strip())
        # Source hashes remain intact; the validation runner checks new aliases
        # against the original gold record before computing any quality metric.
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',default=str(ROOT/'validation_queries.json'))
    p.add_argument('--reviews',default=str(ROOT/'research/annotation_review.csv'))
    p.add_argument('--output',default=str(ROOT/'reviewed_validation_queries.json'))
    args=p.parse_args();output=Path(args.output)
    if output.resolve()==Path(args.suite).resolve():p.error('Use a separate output filename; keep the original suite.')
    if output.exists():p.error('Output already exists; choose a new filename to preserve previous reviews.')
    records=json.loads((ROOT/'faiss_store/raw_records_cache.json').read_text())
    known={f"{r['dataset']}#{r['id']}" for r in records}
    with open(args.reviews,newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    result=apply_reviews(json.loads(Path(args.suite).read_text()),rows,known)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    cases=result['queries'] if isinstance(result,dict) else result
    print(f'Saved {output}; {sum(bool(q.get("human_reviewed")) for q in cases)} named human-reviewed cases.')
if __name__=='__main__':main()
