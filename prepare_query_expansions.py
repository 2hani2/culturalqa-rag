"""Prepare API search phrasings, without sending gold labels or evidence.

Cached expansions are retained. Calls for new questions are spaced to reduce
rate-limit pressure. Evaluation later shares exactly these caches across modes.
"""
import argparse,datetime,hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
from dotenv import load_dotenv
from project_paths import ROOT
from retrieval import GlobalRetriever
from validation import load_suite

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',default=str(ROOT/'validation_queries_120.json'))
    p.add_argument('--model',default='openai/gpt-oss-20b')
    p.add_argument('--output',default=str(ROOT/'research/query_expansion_120_manifest.json'))
    args=p.parse_args();load_dotenv(ROOT/'.env')
    cases=[q for q in load_suite(args.suite) if q['task']=='text']
    client=SimpleNamespace(use_query_expansion=True,expansion_model=args.model);rows=[];last_call=0.
    for i,q in enumerate(cases,1):
        key=hashlib.sha256(json.dumps([q['question'],args.model,'expansion-v1']).encode()).hexdigest()
        path=ROOT/'query_expansion_cache'/f'{key}.json';existed=path.exists()
        for attempt in range(4):
            try:
                if not path.exists():
                    time.sleep(max(0,2.1-(time.monotonic()-last_call)));last_call=time.monotonic()
                forms,cached=GlobalRetriever.expand(client,q['question']);break
            except Exception as exc:
                if attempt==3:
                    print(f'[expansion] Failed {q["id"]}: {type(exc).__name__}; no secret/configuration values logged.',flush=True)
                    raise RuntimeError(f'Could not expand {q["id"]}') from None
                print(f'[expansion] Retry {q["id"]}: {type(exc).__name__}',flush=True);time.sleep(min(30,5*(attempt+1)))
        if forms[0]!=q['question'] or not all(isinstance(x,str) and x.strip() for x in forms):raise ValueError('Invalid cached phrasings.')
        rows.append({'query_id':q['id'],'question_sha256':hashlib.sha256(q['question'].encode()).hexdigest(),
            'queries':forms,'reused_existing_cache':existed,'cache_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        print(f'[expansion] {i}/{len(cases)} {q["id"]}: {"reused" if existed else "API-generated"}; {len(forms)} phrasings',flush=True)
    manifest={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model':args.model,
        'suite_sha256':hashlib.sha256(Path(args.suite).read_bytes()).hexdigest(),'text_queries':len(cases),
        'new_expansions':sum(not r['reused_existing_cache'] for r in rows),
        'gold_answers_sent':False,'source_evidence_sent':False,'image_query_expansion':False,
        'timing_note':'API preparation is separate from cached evaluation latency. Evaluation logs cache hits and uses shared phrasings.',
        'queries':rows}
    Path(args.output).write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print('[expansion] Prepared all phrasings; saved',args.output,flush=True)
if __name__=='__main__':main()
