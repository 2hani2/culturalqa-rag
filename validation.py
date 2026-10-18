"""Three-condition held-out validation with shared pools and error-aware metrics."""
import argparse,csv,datetime,json,random,shutil,time
from collections import defaultdict
from pathlib import Path
import torch
from project_paths import ROOT,RESEARCH
from dataset_config import ACTIVE_DATASETS
from validation_metrics import ranking_metrics,answer_metrics,normalize_answer
from research_protocol import canonical,digest,record_ref,load_records,image_digest
from reranker_models import sha256_file

MODES=('without_moe','simple_fusion','with_moe')
QUALITY=('hit_rate','recall','precision','mrr','ndcg','exact_match','answer_f1','answer_evidence_hit')

def load_suite(path):
    data=json.loads(Path(path).read_text());cases=data.get('queries') if isinstance(data,dict) else data
    if not isinstance(cases,list) or not cases:raise ValueError('Supply a nonempty list of questions.')
    ids=set()
    for q in cases:
        if not isinstance(q,dict) or not isinstance(q.get('id'),str) or not q['id'] or q['id'] in ids:raise ValueError('Query IDs must be unique strings.')
        ids.add(q['id'])
        if q.get('task') not in ('text','image') or not isinstance(q.get('question'),str) or not q['question'].strip():raise ValueError('Each item needs a text/image task and a question.')
        for key in ('relevant_records','relevant_images','expected_answers'):
            if q.get(key) is not None and (not isinstance(q[key],list) or not all(isinstance(x,str) for x in q[key])):raise ValueError(f'{key} must be a list of strings or null.')
        if q.get('human_reviewed') and not q.get('reviewer'):raise ValueError('A human-reviewed label requires its reviewer name.')
    return cases

def verify_corpus_sources(split,root=ROOT,research=RESEARCH):
    """Permit only a documented index rebuild of an unchanged source corpus.

    Checkpoint/split provenance keeps its original hashes. Re-embedded FAISS
    artifacts get separate hashes; altered raw sources still fail validation.
    """
    root=Path(root);research=Path(research)
    expected=dict(split['source_hashes'])
    index_keys=('faiss_store/index.faiss','faiss_store/index.pkl')
    reembedded=any(not (root/key).is_file() or sha256_file(root/key)!=expected[key] for key in index_keys)
    recovery=None
    if reembedded:
        path=research/'index_recovery_manifest.json'
        if not path.is_file():raise ValueError('Knowledge index changed without a verified recovery manifest.')
        recovery=json.loads(path.read_text())
        checks=(
            recovery.get('format_version')==1,
            recovery.get('original_split_sha256')==sha256_file(research/'split_manifest.json'),
            recovery.get('raw_cache_sha256')==expected['faiss_store/raw_records_cache.json'],
            recovery.get('original_index_hashes')=={key:expected[key] for key in index_keys},
            set(recovery.get('recovered_index_hashes',{}))==set(index_keys),
            recovery.get('coverage_matches_original') is True,
            recovery.get('coverage_csv_sha256')==sha256_file(research/'index_coverage.csv'),
            recovery.get('coverage_summary_sha256')==sha256_file(research/'index_coverage_summary.json'),
        )
        if not all(checks):raise ValueError('Index recovery does not match the recorded corpus and split.')
        expected.update(recovery['recovered_index_hashes'])
    for relative,fingerprint in expected.items():
        source=root/relative
        if not source.is_file() or sha256_file(source)!=fingerprint:raise ValueError(f'Source changed since split preparation: {relative}')
    return recovery

def check_protocol(cases,split,provenance,records):
    from structured_store import record_to_text
    lookup={record_ref(r):r for r in records};checks=[]
    if split.get('image_duplicate_audit',{}).get('cross_fold_duplicate_groups_after_merge')!=0:
        raise ValueError('Image duplicate audit must establish zero cross-fold aliases.')
    for r in records:
        if r.get('image_url') and Path(r['image_url']).is_file():
            expected=r['dataset']+'::image-sha256:'+image_digest(r['image_url'])
            if split['assignments'][record_ref(r)]['group']!=expected:
                raise ValueError('Image bytes changed since split preparation.')
    for kind,p in provenance.items():
        if set(p['active_datasets'])!=set(ACTIVE_DATASETS):raise ValueError('Checkpoint includes an excluded training dataset.')
        if p['split_sha256']!=sha256_file(RESEARCH/'split_manifest.json'):raise ValueError('Checkpoint split differs from evaluation split.')
        if p['training_fold']!='train' or p['test_fold_used']:raise ValueError('Test data were used for training/selection.')
        if any(split['assignments'][ref]['fold']!='train' for ref in p['training_refs']):raise ValueError('Training reference leaked from another fold.')
        train_groups={split['assignments'][ref]['group'] for ref in p['training_refs']}
        for q in cases:
            ref=q.get('gold_record_ref')
            if q.get('fold')=='test':
                if not ref or split['assignments'][ref]['fold']!='test':raise ValueError(f"Invalid held-out target: {q['id']}")
                if split['assignments'][ref]['group'] in train_groups:raise ValueError('Shared image/subject leaked into training.')
                if digest(canonical(q['question'])) in p['train_query_sha256']:raise ValueError('Evaluation question was a training query.')
        checks.append({'model':kind,'training_refs':len(p['training_refs']),'test_refs_in_training':0,'shared_test_groups_in_training':0,'query_overlap':0})
    if provenance['moe']['training_pairs_sha256']!=provenance['simple']['training_pairs_sha256']:raise ValueError('The two models used different training data.')
    for key in ('seed','epochs','batch_size','learning_rate','margin','enable_images'):
        if provenance['moe'][key]!=provenance['simple'][key]:raise ValueError(f'Training control mismatch: {key}')
    recovery=verify_corpus_sources(split)
    for check in checks:check['knowledge_index_reembedded']=recovery is not None
    for q in cases:
        if q.get('source_verified') and q['task']=='text':
            text=record_to_text(lookup[q['gold_record_ref']])
            if digest(text)!=q['source_text_sha256']:raise ValueError(f"Gold source changed: {q['id']}")
            if not any(canonical(a) in canonical(text) for a in q.get('expected_answers',[])):raise ValueError('Accepted answer absent from labelled source.')
        if q['task']=='image':
            if any(not (ROOT/ref).is_file() for ref in q.get('relevant_images',[])):raise ValueError('Labelled image is missing.')
    return checks

def compare_images(pipe,question,candidate_k):
    started=time.perf_counter();hits=pipe.image_search.pool.retrieve(question,top_k=candidate_k);retrieval_ms=(time.perf_counter()-started)*1000
    text_started=time.perf_counter();qt=pipe.encoder.text_encoder.encode(question);feature_data=[]
    for hit in hits:
        from structured_store import record_to_text
        texts=[record_to_text(r)[:2000] for r in hit['_records']]
        vectors=[]
        for start in range(0,len(texts),32):vectors.append(pipe.encoder.text_encoder.encode(texts[start:start+32]))
        dt=torch.cat(vectors);di=pipe.encoder.image_encoder.encode(hit['image_ref']).expand(len(dt),-1) if pipe.enable_images else torch.zeros(len(dt),512)
        feature_data.append((dt,di))
    feature_ms=(time.perf_counter()-text_started)*1000;conditions={}
    for mode in MODES:
        started=time.perf_counter();ranked=[]
        model=pipe.models['simple' if mode=='simple_fusion' else 'moe'] if mode!='without_moe' else None
        for hit,(dt,di) in zip(hits,feature_data):
            row={k:v for k,v in hit.items() if k!='_records'}
            row['ref']='image:'+image_digest(hit['image_ref'])
            if model is None:score=hit['clip_score']
            else:
                scores=model.score_relevance(qt.expand(len(dt),-1),torch.zeros(len(dt),512),dt,di)
                if not torch.isfinite(scores).all():raise ValueError('Nonfinite image reranking score.')
                score=float(scores.max())
            row['ranking_score']=score;ranked.append(row)
        ranked.sort(key=lambda r:r['ranking_score'],reverse=True)
        conditions[mode]={'ranked':ranked,'answers':{},'fusion_features_ms':feature_ms if model else 0.,'reranker_ms':(time.perf_counter()-started)*1000 if model else 0.}
    return {'conditions':conditions,'datasets_searched':pipe.image_search.pool.datasets_searched,
            'datasets_with_images':pipe.image_search.pool.datasets_with_images,'query_expansion':False,
            'phrasings':[question],'retrieval_ms':retrieval_ms,'image_features_enabled':pipe.enable_images,'candidate_k':candidate_k}

def evaluate_cases(cases,evaluate_text,evaluate_image):
    rows=[];raw=[]
    for i,q in enumerate(cases,1):
        print(f"[validation] {i}/{len(cases)} {q['id']}: {q['question']}",flush=True)
        result={};error=''
        try:
            result=(evaluate_text if q['task']=='text' else evaluate_image)(q['question'])
            if set(result.get('conditions',{}))!=set(MODES):raise ValueError('Missing paired condition.')
            sets=[{r['ref'] for r in result['conditions'][mode]['ranked']} for mode in MODES]
            if not all(s==sets[0] for s in sets):raise ValueError('Conditions used different candidate pools.')
        except Exception as exc:error=f'{type(exc).__name__}: {exc}';print('[validation error]',error,flush=True)
        raw.append({'case':q,'result':result,'error':error})
        relevant=q.get('relevant_records') if q['task']=='text' else q.get('resolved_image_labels',q.get('relevant_images'))
        for mode in MODES:
            condition=result.get('conditions',{}).get(mode,{});ranking=condition.get('ranked',[])
            metrics=ranking_metrics([r['ref'] for r in ranking],relevant)
            for k in (3,6):
                answer=condition.get('answers',{}).get(str(k),{});reader_errors=answer.get('reader_errors',0)
                status='error' if error or reader_errors else 'ok'
                row=dict(query_id=q['id'],evaluation_group=q.get('evaluation_group',q.get('gold_record_ref',q['id'])),task=q['task'],modality=q.get('modality','unspecified'),target_dataset=q.get('target_dataset','unspecified'),
                    mode=mode,k=k,status=status,error=error or '; '.join(answer.get('reader_error_details',[])),reader_errors=reader_errors,
                    evaluation_role=q.get('evaluation_role','unverified'),source_verified=bool(q.get('source_verified')),human_reviewed=bool(q.get('human_reviewed')),
                    relevance_complete=bool(q.get('relevance_complete')),relevance_labelled=bool(relevant),answer_labelled=bool(q.get('expected_answers')),
                    label_unit='record_id' if q['task']=='text' else 'image_file_content_sha256',question=q['question'],answer=answer.get('answer'),
                    expected_answers=q.get('expected_answers',[]),relevant_refs=relevant or [],ranked_refs=[r['ref'] for r in ranking[:k]],
                    datasets_searched=result.get('datasets_searched',[]),scope_coverage=len(set(result.get('datasets_searched',[]))&set(ACTIVE_DATASETS))/len(ACTIVE_DATASETS),
                    returned_count=min(k,len(ranking)),query_expansion=result.get('query_expansion',False),
                    retrieval_ms=result.get('retrieval_ms'),fusion_features_ms=condition.get('fusion_features_ms'),reranker_ms=condition.get('reranker_ms'),reader_ms=answer.get('reader_ms',0.))
                for field,key in [('hit_rate','hit'),('recall','recall'),('precision','precision'),('mrr','mrr'),('ndcg','ndcg')]:row[field]=metrics[f'{key}@{k}'] if status=='ok' else None
                scores=answer_metrics(answer.get('answer'),q.get('expected_answers')) if q['task']=='text' else answer_metrics(None,None)
                row.update({key:value if status=='ok' else None for key,value in scores.items()})
                row['answer_evidence_hit']=None
                if q['task']=='text' and q.get('expected_answers') and relevant and status=='ok':
                    # Full text is supplied in the reader-result evidence map.
                    supporting=set(result.get('answer_supporting_refs',[]))
                    row['answer_evidence_hit']=float(any(r['ref'] in supporting and r['ref'] in relevant for r in ranking[:k]))
                rows.append(row)
    return {'per_question':rows,'raw_results':raw,'summary':summarize(rows),'paired_differences':paired_differences(rows)}

def summarize(rows):
    groups=defaultdict(list)
    for row in rows:
        for modality in {'all',row['modality']}:
            for ds in {'all',row['target_dataset']}:groups[(row['task'],modality,ds,row['mode'],row['k'])].append(row)
    out=[]
    for (task,modality,ds,mode,k),data in sorted(groups.items()):
        failed=sum(r['status']!='ok' for r in data)
        summary=dict(task=task,modality=modality,target_dataset=ds,mode=mode,k=k,queries=len(data),failed_queries=failed,
            source_verified=sum(r['source_verified'] for r in data),human_reviewed=sum(r['human_reviewed'] for r in data),
            complete_relevance_labels=sum(r['relevance_complete'] for r in data),relevance_labelled=sum(r['relevance_labelled'] for r in data),answer_labelled=sum(r['answer_labelled'] for r in data))
        for metric in QUALITY+('scope_coverage','retrieval_ms','fusion_features_ms','reranker_ms','reader_ms'):
            vals=[r[metric] for r in data if r.get(metric) is not None]
            summary[metric]=sum(vals)/len(vals) if vals and not (failed and metric in QUALITY) else None
        out.append(summary)
    return out

def paired_differences(rows):
    lookup={(r['query_id'],r['mode'],r['k']):r for r in rows};out=[]
    for task in ('text','image'):
        for mode in ('simple_fusion','with_moe'):
            for k in (3,6):
                for metric in ('mrr','answer_f1'):
                    pairs=[];grouped=defaultdict(list)
                    for r in rows:
                        if r['task']==task and r['mode']=='without_moe' and r['k']==k:
                            other=lookup[(r['query_id'],mode,k)]
                            if r['status']=='ok' and other['status']=='ok' and r[metric] is not None and other[metric] is not None:
                                delta=other[metric]-r[metric];pairs.append(delta);grouped[r.get('evaluation_group',r['query_id'])].append(delta)
                    if not pairs:continue
                    rng=random.Random(42);groups=list(grouped.values());samples=[]
                    for _ in range(1000):
                        sample=[delta for group in rng.choices(groups,k=len(groups)) for delta in group]
                        samples.append(sum(sample)/len(sample))
                    samples.sort()
                    out.append(dict(task=task,mode=mode,k=k,metric=metric,paired_queries=len(pairs),independent_source_groups=len(groups),
                        mean_delta=sum(pairs)/len(pairs),bootstrap_low=samples[24],bootstrap_high=samples[974],bootstrap_unit='source_group',
                        note='Descriptive paired cluster bootstrap over source groups; related questions are resampled together, not proof of generalization.'))
    return out

def write_report(report,output_dir):
    directory=Path(output_dir);directory=directory if directory.is_absolute() else ROOT/directory
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'validation_report.json').exists():
        previous=directory/'previous_runs'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');previous.mkdir(parents=True)
        for name in ('validation_report.json','validation_matrix.csv','per_question.csv','paired_differences.csv'):
            if (directory/name).exists():shutil.copy2(directory/name,previous/name)
    report['created_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    for field,name in [('summary','validation_matrix.csv'),('per_question','per_question.csv'),('paired_differences','paired_differences.csv')]:
        data=report[field]
        if data:
            with (directory/name).open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader()
                w.writerows({key:json.dumps(value,ensure_ascii=False) if isinstance(value,(list,dict)) else value for key,value in row.items()} for row in data)
    (directory/'validation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print('[validation] saved',directory,flush=True);return report

def run_validation(input_path='validation_queries.json',output_dir='validation_results',candidate_k=12,use_query_expansion=True,enable_images=True,max_queries=None):
    if candidate_k<6:raise ValueError('K=6 evaluation needs a shared candidate pool of at least six.')
    cases=load_suite(input_path);cases=cases[:max_queries] if max_queries else cases
    for q in cases:
        if q['task']=='image':q['resolved_image_labels']=['image:'+image_digest(ROOT/ref) for ref in q.get('relevant_images',[])]
    split=json.loads((RESEARCH/'split_manifest.json').read_text())
    from pipeline import CulturalQAPipeline
    pipe=CulturalQAPipeline(enable_images=enable_images,load_rerankers=True,candidate_k=candidate_k,
                           image_top_k=candidate_k,image_candidate_k=candidate_k,use_query_expansion=use_query_expansion)
    if any(q['task']=='image' for q in cases) and pipe.image_search is None:
        from image_retrieval import ImageSearch
        pipe.encoder.image_encoder.enable_live_fetch=True
        pipe.image_search=ImageSearch(pipe.encoder.image_encoder,list(pipe.record_index.values()),top_k=candidate_k,candidate_k=candidate_k)
    checks=check_protocol(cases,split,pipe.provenance,list(pipe.record_index.values()))
    def text(q):
        result=pipe.compare_validation(q)
        from validation_metrics import evidence_ref
        labels=next(c.get('expected_answers',[]) for c in cases if c['question']==q)
        result['answer_supporting_refs']=[ref for ref,body in result['candidate_texts'].items() if any(normalize_answer(a) and normalize_answer(a) in normalize_answer(body) for a in labels)]
        return result
    report=evaluate_cases(cases,text,lambda q:compare_images(pipe,q,candidate_k))
    report.update(protocol_checks=checks,checkpoints=pipe.provenance,split_counts=split['counts'],image_duplicate_audit=split.get('image_duplicate_audit'),
        index_recovery=verify_corpus_sources(split),
        current_index_hashes={key:sha256_file(ROOT/key) for key in ('faiss_store/index.faiss','faiss_store/index.pkl')},
        settings=dict(candidate_k=candidate_k,cutoffs=[3,6],branch_k=8,query_expansion=use_query_expansion,
            expansion_model=pipe.retriever.expansion_model,image_features_enabled=enable_images,shared_candidate_pool=True,reader_shared=True,
            reader_timing_policy='Sum measured per-context extraction costs for each subset. Results extracted once per query, shared across conditions.',
            latency_excludes='Model loading and process startup; image-pool initialization may affect first image query.',
            bootstrap_policy='Paired source-group cluster resampling; 1000 resamples, seed 42; query-weighted mean difference.'),
        input_sha256=sha256_file(input_path),index_coverage=json.loads((RESEARCH/'index_coverage_summary.json').read_text()),
        limitations=['Relevance labels are verified positives, not exhaustive judgements of the full corpus.',
            'Source-verified labels are not independent human expert review; human_reviewed flags remain explicit.',
            'Training pairs use weak source-derived positives and sampled negatives.',
            'Partial indexing of long source records is quantified in index_coverage.',
            'A single training seed is recorded; these small-sample results do not establish a general MoE benefit.'])
    return write_report(report,output_dir)

if __name__=='__main__':
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env');torch.set_num_threads(2)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',default=str(ROOT/'validation_queries.json'))
    p.add_argument('--output_dir',default='validation_results');p.add_argument('--candidate_k',type=int,default=12)
    p.add_argument('--no_query_expansion',action='store_true');p.add_argument('--text_only',action='store_true');p.add_argument('--max_queries',type=int)
    a=p.parse_args();run_validation(a.suite,a.output_dir,a.candidate_k,not a.no_query_expansion,not a.text_only,a.max_queries)
