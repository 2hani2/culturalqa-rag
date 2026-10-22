"""Shared global candidate retrieval, documented rerankers and extractive QA."""
import csv,datetime,json,time
from pathlib import Path
import torch
from langchain_core.documents import Document
from project_paths import ROOT,INDEX,CHECKPOINTS
from dataset_config import is_active_dataset,validate_dataset_filter,excluded_dataset_in_query
from retrieval import load_faiss_index,build_full_retriever,build_faiss_index
from encoders import MultimodalEncoder
from image_retrieval import ImageSearch
from reader import ExtractiveReader,READER_VERSION
from residual_reranker import ResidualReranker
from validation_metrics import evidence_ref,evidence_rows
from structured_store import build_record_index,format_record_answer,load_all_raw_records,records_to_documents
from query_router import is_lookup_query,parse_lookup,is_filter_query
CHUNK_CACHE=str(INDEX/'chunks_cache.json');RAW_RECORDS_CACHE=str(INDEX/'raw_records_cache.json')
PROJECT_DIR=str(ROOT)

def _load_raw_records_cache():
    return [r for r in json.loads((INDEX/'raw_records_cache.json').read_text()) if is_active_dataset(r.get('dataset'))]
def _load_chunks_cache():
    if not (INDEX/'chunks_cache.json').exists():raise FileNotFoundError('Use the FAISS docstore; obsolete duplicate chunk cache is not restored.')
    return [Document(page_content=r['content'],metadata=r['metadata']) for r in json.loads((INDEX/'chunks_cache.json').read_text()) if is_active_dataset(r['metadata'].get('dataset'))]
def build_index(data_dir=str(ROOT/'data')):
    from ingestion import load_all_documents
    from chunking import chunk_documents
    reports=[d for d in load_all_documents(data_dir) if is_active_dataset(d.metadata.get('dataset'))]
    records=[r for r in load_all_raw_records(str(Path(data_dir)/'raw')) if is_active_dataset(r.get('dataset'))]
    docs=(chunk_documents(reports) if reports else [])+records_to_documents(records)
    build_faiss_index(docs);(INDEX/'raw_records_cache.json').write_text(json.dumps(records,ensure_ascii=False))
    print('[index] Re-run research_protocol.py and refresh gold labels after changing the corpus.')

def print_answer(result):
    print('[answer] '+str(result.get('answer') or result.get('message') or 'No answer.'))
    if result.get('score') is not None:print(f"[answer confidence] {result['score']:.4f}")
    if result.get('datasets_searched'):print('[search] '+', '.join(result['datasets_searched']))
    for r in result.get('retrieved_candidates',[]):print(f"[context] {r['rank']}: {r['ref']} | score={r.get('ranking_score') if r.get('ranking_score') is not None else r.get('cross_encoder_score')}")
    if result.get('reader_errors'):print('[reader errors]',result['reader_error_details'])
    for r in result.get('retrieved_images',[]):print(f"[image] {r['rank']}: {r['image_ref']} | CLIP={r['clip_score']:.4f}")

class CulturalQAPipeline:
    def __init__(self,dataset_filter=None,enable_images=False,force_moe_rescore=None,image_top_k=3,image_candidate_k=6,
                 top_k=3,candidate_k=12,use_query_expansion=True,load_rerankers=False,branch_k=8):
        validate_dataset_filter(dataset_filter)
        if not 1<=top_k<=candidate_k:raise ValueError('Require 1 <= top_k <= candidate_k.')
        self.top_k=top_k;self.candidate_k=candidate_k;self.use_query_expansion=use_query_expansion;self.enable_images=enable_images
        self.vectordb=load_faiss_index();self.chunks=[self.vectordb.docstore.search(key) for key in self.vectordb.index_to_docstore_id.values() if is_active_dataset(self.vectordb.docstore.search(key).metadata.get('dataset'))]
        self.retriever=build_full_retriever(self.vectordb,self.chunks,dataset_filter=dataset_filter,candidate_k=candidate_k,use_query_expansion=use_query_expansion,branch_k=branch_k)
        self.datasets_searched=self.retriever.datasets_searched;self.encoder=MultimodalEncoder(enable_live_fetch=enable_images)
        self.force_moe_rescore = bool(force_moe_rescore)
        self.residual = ResidualReranker() if (self.force_moe_rescore or load_rerankers) else None
        self.provenance = {'moe': self.residual.provenance} if self.residual else {}
        self.reader=ExtractiveReader();records=_load_raw_records_cache();self.record_index=build_record_index(records)
        self.image_search=ImageSearch(self.encoder.image_encoder,records,top_k=image_top_k,candidate_k=image_candidate_k) if enable_images else None

    def _resolve_use_moe_rescore(self,explicit=None):
        return self.force_moe_rescore if explicit is None else bool(explicit)

    def _try_exact_lookup(self,question):
        if not is_lookup_query(question):return None
        ds,rid=parse_lookup(question)
        if rid is None:return None
        found=[r for (dataset,key),r in self.record_index.items() if key==rid and (ds is None or dataset==ds)]
        if len(found)==1:return {'answer':format_record_answer(found[0]),'dataset':found[0]['dataset'],'record_id':rid,'path':'exact_lookup','score':1.,'search_scope':'exact_id_lookup'}
        return {'answer':None,'message':'Exact record is missing or ambiguous; specify its dataset.','path':'exact_lookup'}

    def encode_candidates(self,question,candidates):
        started=time.perf_counter();qt=self.encoder.text_encoder.encode(question)
        dt=self.encoder.text_encoder.encode([d.page_content for d in candidates]) if candidates else torch.zeros(0,768)
        di=torch.stack([self.encoder.image_encoder.encode(d.metadata.get("image_url") if self.enable_images and d.metadata.get("has_image") else None)[0] for d in candidates]) if candidates else torch.zeros(0,512)
        return (qt,torch.zeros(1,512),dt,di),(time.perf_counter()-started)*1000

    def _moe_rescore(self,question,candidates):
        if self.residual is None:
            self.residual=ResidualReranker()
            self.provenance={'moe':self.residual.provenance}
        features,_=self.encode_candidates(question,candidates)
        return self.residual.rank(candidates,features)

    def _trace(self,result,docs):
        result.update(datasets_searched=self.datasets_searched,search_scope='all_available_datasets',
            top_k_requested=self.top_k,candidate_k=self.candidate_k,retrieved_candidates=evidence_rows(docs),
            retrieval_trace=self.retriever.last_trace)
        result['ranking_policy']='ce_plus_moe_residual' if result.get('moe_rescore_used') else 'cross_encoder_baseline'
        if result.get('moe_rescore_used'):
            result['residual_alpha']=self.residual.alpha
            result['reranker_checkpoint']=self.residual.selection['checkpoint']
        return result
    def _rag_answer(self,question,use_moe_rescore=None,top_k_for_reader=None):
        use=self._resolve_use_moe_rescore(use_moe_rescore);docs=self.retriever.invoke(question)
        if use:docs=self._moe_rescore(question,docs)
        selected=docs[:top_k_for_reader or self.top_k];result=self.reader.answer(question,selected)
        result.update(path='rag',moe_rescore_used=use);return self._trace(result,selected)
    def _filter_answer(self,question,top_k_for_list=None,use_moe_rescore=None):
        use=self._resolve_use_moe_rescore(use_moe_rescore);docs=self.retriever.invoke(question)
        if use:docs=self._moe_rescore(question,docs)
        docs=docs[:top_k_for_list or self.top_k]
        result={'answer':'\n'.join(f'- [{evidence_ref(d)}] {d.page_content[:200]}' for d in docs) or None,
                'message':'No matching records.' if not docs else '', 'path':'filter_list','moe_rescore_used':use,
                'items':evidence_rows(docs)}
        return self._trace(result,docs)
    def ask(self,question,**kwargs):
        excluded=excluded_dataset_in_query(question)
        if excluded:return {'answer':None,'message':f'{excluded} is excluded.','path':'excluded_dataset'}
        exact=self._try_exact_lookup(question)
        if exact is not None:return exact
        result=self._filter_answer(question) if is_filter_query(question) else self._rag_answer(question,**kwargs)
        result['retrieved_images']=self.image_search.search(question,verbose=False) if self.image_search else []
        result['image_retrieval_used']=self.image_search is not None;return result
    def run_batch(self,questions,show_progress=True):
        results=[]
        for i,q in enumerate(questions,1):
            if show_progress:print(f'[batch] {i}/{len(questions)}: {q}',flush=True)
            try:r=self.ask(q)
            except Exception as exc:r={'answer':None,'message':f'{type(exc).__name__}: {exc}','status':'error'}
            r['question']=q;results.append(r)
            if show_progress:print_answer(r)
        return results

    def compare_validation(self,question):
        if self._try_exact_lookup(question) is not None:raise ValueError("Exact-ID lookup is not a retrieval evaluation.")
        docs=self.retriever.invoke(question);features,feature_ms=self.encode_candidates(question,docs)
        ranking={'without_moe':docs};timing={'without_moe':0.}
        if self.residual is None:
            self.residual=ResidualReranker()
            self.provenance={'moe':self.residual.provenance}
        started=time.perf_counter();ranking['with_moe']=self.residual.rank(docs,features)
        timing['with_moe']=(time.perf_counter()-started)*1000
        # Extract once per context; all conditions use the same cached results.
        # Reader time is reported as a sum of measured per-context times for each
        # selected subset, not the cheaper wall time of replaying cached answers.
        outputs=self.reader.evaluate_contexts(question,docs);by_ref={r['ref']:r for r in outputs};conditions={}
        for mode,ranked in ranking.items():
            answers={str(k):self.reader.select_answer([by_ref[evidence_ref(d)] for d in ranked[:k]]) for k in (3,6)}
            conditions[mode]={'ranked':evidence_rows(ranked),'answers':answers,'fusion_features_ms':feature_ms if mode!='without_moe' else 0.,'reranker_ms':timing[mode],
                'ranking_policy':'cross_encoder_baseline' if mode=='without_moe' else self.residual.policy_name}
        return {'conditions':conditions,'reader_context_results':outputs,'candidate_texts':{evidence_ref(d):d.page_content for d in docs},**self.retriever.last_trace,'reader_version':READER_VERSION,'image_features_enabled':self.enable_images}

def ask(question,**settings):
    pipeline=CulturalQAPipeline(**settings);result=pipeline.ask(question);print_answer(result);return result
def save_batch_results(results,output_path):
    from pathlib import Path
    path=Path(output_path)
    if path.suffix.casefold()=='.csv':
        fields=('question','answer','score','status','message','path','moe_rescore_used','datasets_searched','retrieved_candidates','reader_errors','reader_error_details')
        with path.open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for result in results:
                row={key:result.get(key) for key in fields}
                row['status']=result.get('status') or ('error' if result.get('reader_errors') else 'ok')
                for key,value in row.items():
                    if isinstance(value,(list,dict)):row[key]=json.dumps(value,ensure_ascii=False)
                writer.writerow(row)
    else:path.write_text(json.dumps(results,ensure_ascii=False,indent=2))
    return path
def run_questions_file(input_path,output_path=None,**settings):
    questions=[s.strip() for s in open(input_path,encoding='utf-8') if s.strip() and not s.lstrip().startswith('#')]
    results=CulturalQAPipeline(**settings).run_batch(questions)
    output_path=output_path or ROOT/f"batch_results_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
    save_batch_results(results,output_path);print('[batch] saved',output_path);return results
def run_ablation_file(input_path,output_path=None,**settings):
    pipe=CulturalQAPipeline(load_rerankers=True,**settings)
    results=[{'question':q.strip(),'result':pipe.compare_validation(q.strip())} for q in open(input_path) if q.strip()]
    from pathlib import Path
    Path(output_path or ROOT/'ablation_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2));return results
