"""Fresh matched validation of baseline, simple fallback and conservative MoE.

This performs real retrieval and reading. evaluate_saved.py separately replays
the historical pools. Neither path selects parameters from evaluation labels.
"""
import argparse,copy,datetime,json,time
import torch
from project_paths import ROOT,RESEARCH
from residual_pipeline import ConservativePipeline,ClipFallbackSearch
from residual_reranker import CONSERVATIVE_SELECTION
from reranker_models import sha256_file
from research_protocol import image_digest
from validation_metrics import normalize_answer
from validation import load_suite,check_protocol,verify_corpus_sources,evaluate_cases,write_report,MODES


def verify_selected_inputs():
    selections=json.loads(CONSERVATIVE_SELECTION.read_text())
    checked=[]
    for kind,s in selections.items():
        path=ROOT/s['checkpoint']
        assert sha256_file(path)==s['checkpoint_sha256']
        assert s['selection_fold']=='dev' and s['test_used'] is False
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if 'input_fingerprints' in payload:
            assert payload['test_fold_used'] is False
            for name,expected in payload['input_fingerprints'].items():
                source=ROOT/name if name!='experiment_script_sha256' else ROOT/'experiments/moe_improvement/run_experiment.py'
                if sha256_file(source)!=expected:raise ValueError(f'Training input changed: {name}')
        else:
            assert payload['provenance']['training_fold']=='train'
            assert payload['provenance']['test_fold_used'] is False
        checked.append({'model':kind,'checkpoint_sha256':s['checkpoint_sha256'],
                        'selection_fold':'dev','recorded_training_inputs_verified':True})
    return checked


def compare_images(pipe,question,candidate_k):
    start=time.perf_counter();hits=pipe.image_search.pool.retrieve(question,top_k=candidate_k)
    elapsed=(time.perf_counter()-start)*1000
    ranked=[]
    for hit in hits:
        row={k:v for k,v in hit.items() if k!='_records'}
        row.update(ref='image:'+image_digest(hit['image_ref']),ranking_score=hit['clip_score'],
                   moe_rescore_used=False,ranking_policy='clip_baseline')
        ranked.append(row)
    conditions={mode:{'ranked':copy.deepcopy(ranked),'answers':{},'fusion_features_ms':0.,
                       'reranker_ms':0.,'ranking_policy':'clip_baseline' if mode=='without_moe' else 'clip_fallback'} for mode in MODES}
    return {'conditions':conditions,'datasets_searched':pipe.image_search.pool.datasets_searched,
            'datasets_with_images':pipe.image_search.pool.datasets_with_images,'query_expansion':False,
            'phrasings':[question],'retrieval_ms':elapsed,'candidate_k':candidate_k,'image_features_enabled':pipe.enable_images}


def run_validation(input_path=ROOT/'validation_queries_120.json',output_dir='validation_conservative_live',
                   candidate_k=12,use_query_expansion=True,enable_images=True,max_queries=None):
    if candidate_k<6:raise ValueError('At least six candidates are required for top-6 evaluation.')
    selected_checks=verify_selected_inputs()
    cases=load_suite(input_path)
    if max_queries:cases=cases[:max_queries]
    for q in cases:
        if q['task']=='image':q['resolved_image_labels']=['image:'+image_digest(ROOT/ref) for ref in q.get('relevant_images',[])]
    pipe=ConservativePipeline(enable_images=enable_images,load_rerankers=True,force_moe_rescore=True,
        candidate_k=candidate_k,image_candidate_k=candidate_k,image_top_k=candidate_k,use_query_expansion=use_query_expansion)
    if pipe.image_search is None and any(q['task']=='image' for q in cases):
        pipe.encoder.image_encoder.enable_live_fetch=True
        pipe.image_search=ClipFallbackSearch(pipe.encoder.image_encoder,list(pipe.record_index.values()),candidate_k=candidate_k)
    split=json.loads((RESEARCH/'split_manifest.json').read_text())
    original_checks=check_protocol(cases,split,pipe.provenance,list(pipe.record_index.values()))
    def text(question):
        result=pipe.compare_validation(question)
        labels=next(q.get('expected_answers',[]) for q in cases if q['question']==question)
        result['answer_supporting_refs']=[ref for ref,body in result['candidate_texts'].items()
            if any(normalize_answer(a) and normalize_answer(a) in normalize_answer(body) for a in labels)]
        return result
    report=evaluate_cases(cases,text,lambda q:compare_images(pipe,q,candidate_k))
    report.update(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        checkpoints={'moe':pipe.residual.selection,'simple':pipe.simple_residual.selection},
        protocol_checks=original_checks,selected_checkpoint_checks=selected_checks,
        base_training_provenance=pipe.provenance,input_sha256=sha256_file(input_path),
        conservative_selection_sha256=sha256_file(CONSERVATIVE_SELECTION),
        split_counts=split['counts'],image_duplicate_audit=split.get('image_duplicate_audit'),
        index_recovery=verify_corpus_sources(split),index_coverage=json.loads((RESEARCH/'index_coverage_summary.json').read_text()),
        settings={'execution':'fresh retrieval and reader inference','candidate_k':candidate_k,'cutoffs':[3,6],
                  'query_expansion':use_query_expansion,'image_features_enabled':enable_images,
                  'shared_candidate_pool':True,'reader_shared':True,'protected_evidence_cutoffs':[3,6],
                  'text_ranking_policy':'development-selected bounded near-tie MoE; baseline evidence membership protected',
                  'image_ranking_policy':'CLIP baseline fallback; no MoE image gain claimed',
                  'reader_timing_policy':'Shared per-context extraction; selected costs summed. Startup and fresh expansion generation are separate concerns.',
                  'bootstrap_policy':'Paired source-group bootstrap, 1000 resamples, seed 42.'},
        limitations=['Conservative selection preserves top-3/top-6 evidence; matching answer quality is not evidence of MoE answering benefit.',
                     'Image results use CLIP fallback; alpha-zero simple results are also baseline fallback.',
                     'Benchmark labels are source-verified by the assistant, not independently human-reviewed, and relevance judgements are incomplete.',
                     'The 120-question benchmark has been inspected before; this is not a new blind test.',
                     'A single training seed and 60 development queries do not establish general MoE superiority.',
                     'Long-record index coverage remains partial and is reported separately.'])
    return write_report(report,output_dir)


if __name__=='__main__':
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env');torch.set_num_threads(2)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',default=str(ROOT/'validation_queries_120.json'))
    p.add_argument('--output_dir',default='validation_conservative_live')
    p.add_argument('--candidate_k',type=int,default=12)
    p.add_argument('--no_query_expansion',action='store_true')
    p.add_argument('--text_only',action='store_true');p.add_argument('--max_queries',type=int)
    a=p.parse_args()
    run_validation(a.suite,a.output_dir,a.candidate_k,not a.no_query_expansion,not a.text_only,a.max_queries)
