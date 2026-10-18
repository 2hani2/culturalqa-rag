"""Global hybrid retrieval with cached expansion and a shared cross-encoder."""
import hashlib,json,re,time
from collections import defaultdict
import numpy as np
import faiss
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from dataset_config import is_active_dataset,validate_dataset_filter
from project_paths import ROOT,INDEX
from validation_metrics import evidence_ref
INDEX_DIR=str(INDEX)
EMBEDDING_MODEL_NAME='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
def _bm25_preprocess(text): return re.sub(r'[^\w\s]',' ',text.casefold()).split()
def load_faiss_index(index_dir=INDEX_DIR):
    emb=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return FAISS.load_local(str(index_dir),emb,allow_dangerous_deserialization=True)
def build_faiss_index(chunks,index_dir=INDEX_DIR,batch_size=500):
    chunks=[d for d in chunks if is_active_dataset(d.metadata.get('dataset'))]
    if not chunks: raise ValueError('No active documents to index.')
    emb=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME); store=None
    for start in range(0,len(chunks),batch_size):
        batch=chunks[start:start+batch_size]
        if store is None: store=FAISS.from_documents(batch,emb)
        else: store.add_documents(batch)
        print(f'[index] {min(start+batch_size,len(chunks))}/{len(chunks)}',flush=True)
    store.save_local(str(index_dir)); return store
def _dataset_dense_store(store,dataset):
    if not hasattr(store,'_dataset_dense_stores'): store._dataset_dense_stores={}
    if dataset not in store._dataset_dense_stores:
        pairs=[(i,key) for i,key in sorted(store.index_to_docstore_id.items()) if store.docstore.search(key).metadata.get('dataset')==dataset]
        idx=faiss.IndexFlatIP(store.index.d) if store.index.metric_type==faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(store.index.d)
        if pairs: idx.add(store.index.reconstruct_batch(np.array([i for i,_ in pairs],dtype=np.int64)))
        store._dataset_dense_stores[dataset]=FAISS(store.embedding_function,idx,store.docstore,{n:key for n,(_,key) in enumerate(pairs)},normalize_L2=store._normalize_L2,distance_strategy=store.distance_strategy)
    return store._dataset_dense_stores[dataset]
def rank_fuse(lists,weights):
    scores=defaultdict(float); docs={}
    for results,weight in zip(lists,weights):
        seen=set()
        for rank,d in enumerate(results,1):
            key=(d.metadata.get('dataset'),d.page_content)
            if key in seen: continue
            seen.add(key); scores[key]+=weight/(60+rank); docs[key]=d
    return [docs[key] for key in sorted(scores,key=lambda key:scores[key],reverse=True)]
class GlobalRetriever:
    def __init__(self,store,chunks,dataset_hint=None,candidate_k=6,use_query_expansion=True,branch_k=8,expansion_model='openai/gpt-oss-20b',cross_encoder=None):
        validate_dataset_filter(dataset_hint)
        if candidate_k<1 or branch_k<1: raise ValueError('Retrieval budgets must be positive.')
        self.hint=dataset_hint; self.candidate_k=candidate_k; self.branch_k=branch_k
        self.use_query_expansion=use_query_expansion; self.expansion_model=expansion_model
        self.datasets_searched=sorted({d.metadata['dataset'] for d in chunks if is_active_dataset(d.metadata.get('dataset'))})
        if not self.datasets_searched: raise ValueError('No active datasets in index.')
        self.dense={ds:_dataset_dense_store(store,ds) for ds in self.datasets_searched}; self.bm25={}
        for ds in self.datasets_searched:
            self.bm25[ds]=BM25Retriever.from_documents([d for d in chunks if d.metadata.get('dataset')==ds],preprocess_func=_bm25_preprocess)
        self.cross_encoder=cross_encoder or HuggingFaceCrossEncoder(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'); self.last_trace={}
    def expand(self,question):
        if not self.use_query_expansion: return [question],False
        key=hashlib.sha256(json.dumps([question,self.expansion_model,'expansion-v1']).encode()).hexdigest(); cache=ROOT/'query_expansion_cache'/f'{key}.json'
        if cache.exists(): return json.loads(cache.read_text())['queries'],True
        from langchain_groq import ChatGroq
        message=ChatGroq(model=self.expansion_model,temperature=0).invoke([('system','Generate three alternative search phrasings preserving the question meaning. Return only a JSON array of three strings. Do not answer the question.'),('human',question)])
        content=message.content
        if not isinstance(content,str): raise ValueError('Unexpected expansion response.')
        queries=json.loads(content[content.find('['):content.rfind(']')+1])
        if not isinstance(queries,list) or not all(isinstance(q,str) and q.strip() for q in queries): raise ValueError('Invalid expansion JSON.')
        queries=list(dict.fromkeys([question]+queries[:3])); cache.parent.mkdir(exist_ok=True)
        cache.write_text(json.dumps({'queries':queries,'model':self.expansion_model},ensure_ascii=False,indent=2)); return queries,False
    def invoke(self,question):
        started=time.perf_counter(); forms,cached=self.expand(question); expansion_ms=(time.perf_counter()-started)*1000
        hint=self.hint
        if hint is None:
            from query_router import detect_dataset
            hint=detect_dataset(question)
        all_results=[]
        for form in forms:
            branches=[]; weights=[]
            for ds in self.datasets_searched:
                k=self.branch_k*(2 if ds==hint else 1)
                dense=self.dense[ds].similarity_search(form,k=k); self.bm25[ds].k=k
                branches.append(rank_fuse([dense,self.bm25[ds].invoke(form)],[.6,.4])); weights.append(2. if ds==hint else 1.)
            all_results.extend(rank_fuse(branches,weights))
        pool=list({(d.metadata.get('dataset'),d.page_content):d for d in all_results}.values()); selected=[]; seen=set()
        if pool:
            scores=self.cross_encoder.score([(question,d.page_content) for d in pool])
            for d,s in sorted(zip(pool,scores),key=lambda x:float(x[1]),reverse=True):
                ref=evidence_ref(d)
                if ref in seen: continue
                seen.add(ref); selected.append(Document(page_content=d.page_content,metadata={**d.metadata,'cross_encoder_score':float(s)}))
                if len(selected)>=self.candidate_k: break
        self.last_trace=dict(datasets_searched=self.datasets_searched,query_expansion=self.use_query_expansion,phrasings=forms,expansion_cache_hit=cached,expansion_ms=expansion_ms,pooled_candidates=len(pool),candidate_k=self.candidate_k,branch_k=self.branch_k,retrieval_ms=(time.perf_counter()-started)*1000)
        return selected
def build_full_retriever(vectordb,chunks,dataset_filter=None,boost_dataset=None,boost_factor=2.,candidate_k=6,use_query_expansion=True,branch_k=8):
    return GlobalRetriever(vectordb,chunks,dataset_filter or boost_dataset,candidate_k,use_query_expansion,branch_k)
