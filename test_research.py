"""Meaningful deterministic protocol, ranking and reader regression checks."""
import contextlib,csv,io,json,math,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock,patch
import torch
torch.set_num_threads(2)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from reader import best_span,ExtractiveReader
from research_protocol import make_split,merge_image_aliases,group_key,record_ref,load_records,canonical,digest
from retrieval import _dataset_dense_store,GlobalRetriever
from dataset_config import ACTIVE_DATASETS
from validation_metrics import ranking_metrics,answer_metrics
from reranker_models import create_model,load_model,save_model
from validation import evaluate_cases,check_protocol,load_suite,verify_corpus_sources,paired_differences,MODES
from project_paths import ROOT,RESEARCH

class FixedEmbedding(Embeddings):
    def embed_documents(self,texts):return [[0.,0.] if t.startswith('large') else [10.,10.] for t in texts]
    def embed_query(self,text):return [0.,0.]

class ReaderTests(unittest.TestCase):
    def test_question_tokens_cannot_be_answer(self):
        start=torch.tensor([0.,100.,1.,4.,2.]);end=torch.tensor([0.,100.,1.,4.,2.])
        valid=torch.tensor([False,False,False,True,True])
        self.assertEqual(best_span(start,end,valid)[:2],(3,3))
    def test_joint_span_is_ordered_and_length_limited(self):
        start=torch.tensor([0.,1.,10.,0.]);end=torch.tensor([10.,9.,0.,7.]);valid=torch.tensor([False,True,True,True])
        self.assertEqual(best_span(start,end,valid,2)[:2],(2,3))
    def test_no_valid_context_span(self):
        self.assertIsNone(best_span(torch.ones(3),torch.ones(3),torch.zeros(3,dtype=torch.bool)))
    def test_empty_high_confidence_does_not_hide_valid_answer(self):
        results=[{'answer':None,'score':.99,'status':'no_answer','reader_ms':1.},
                 {'answer':'Bihar','score':.2,'status':'ok','reader_ms':2.}]
        self.assertEqual(ExtractiveReader.select_answer(results)['answer'],'Bihar')
    def test_reader_errors_are_visible_even_with_an_answer(self):
        r=ExtractiveReader.select_answer([{'answer':'Bihar','score':.2,'status':'ok','reader_ms':1.},
            {'answer':None,'status':'error','error':'model failed','reader_ms':1.}])
        self.assertEqual(r['reader_errors'],1);self.assertEqual(r['evaluation_status'],'reader_error')
    def test_exception_is_not_silently_swallowed(self):
        reader=ExtractiveReader.__new__(ExtractiveReader);reader._answer_one=Mock(side_effect=RuntimeError('bad model'))
        r=reader.answer('q',[Document(page_content='evidence',metadata={'dataset':'DOSA','record_id':1})])
        self.assertEqual(r['reader_errors'],1);self.assertIn('bad model',r['reader_error_details'][0])

class MetricTests(unittest.TestCase):
    def test_duplicate_and_missing_result_slots(self):
        m=ranking_metrics(['x','a','a','b'],['a','b'])
        self.assertEqual(m['recall@3'],.5);self.assertEqual(m['recall@6'],1.)
        self.assertAlmostEqual(m['precision@3'],1/3);self.assertAlmostEqual(m['precision@6'],2/6)
        self.assertEqual(m['mrr@3'],.5)
    def test_ndcg_discount_and_binary_ideal(self):
        m=ranking_metrics(['x','a'],['a']);self.assertAlmostEqual(m['ndcg@3'],1/math.log2(3))
    def test_unlabelled_is_not_zero(self):
        self.assertIsNone(ranking_metrics([],None)['recall@3']);self.assertIsNone(answer_metrics('x',None)['exact_match'])
    def test_answer_aliases_partial_and_no_answer(self):
        self.assertEqual(answer_metrics('The Bihar.',['Bihar'])['exact_match'],1.)
        self.assertAlmostEqual(answer_metrics('warp threads',['warp and weft threads'])['answer_f1'],2/3)
        self.assertEqual(answer_metrics(None,[''])['answer_f1'],1.)

class RetrievalTests(unittest.TestCase):
    def store(self):
        docs=[Document(page_content='large'+str(i),metadata={'dataset':'DRISHTIKON','record_id':i}) for i in range(30)]
        docs.append(Document(page_content='small',metadata={'dataset':'SCALE','record_id':1}))
        return FAISS.from_documents(docs,FixedEmbedding()),docs
    def test_small_dataset_survives_global_fetch_limit(self):
        store,docs=self.store();self.assertEqual(store.similarity_search('q',k=1,filter={'dataset':'SCALE'}),[])
        small=_dataset_dense_store(store,'SCALE');self.assertEqual(small.similarity_search('q',k=1)[0].metadata['dataset'],'SCALE')
        self.assertIs(_dataset_dense_store(store,'SCALE'),small);self.assertEqual(store.index.ntotal,31)
    def test_hint_keeps_all_branches_and_cross_encoder_deduplicates(self):
        store,docs=self.store();ce=Mock(score=lambda pairs:list(range(len(pairs))))
        retriever=GlobalRetriever(store,docs,dataset_hint='SCALE',candidate_k=6,use_query_expansion=False,cross_encoder=ce)
        result=retriever.invoke('q')
        self.assertEqual(set(retriever.last_trace['datasets_searched']),{'SCALE','DRISHTIKON'})
        self.assertEqual(len({(d.metadata['dataset'],d.metadata['record_id']) for d in result}),len(result))
        self.assertTrue(all('cross_encoder_score' in d.metadata for d in result))
    def test_excluded_hint_fails_before_models(self):
        with self.assertRaises(ValueError):GlobalRetriever(None,[],dataset_hint='Durga QA')

class ProtocolTests(unittest.TestCase):
    def recovery_fixture(self,root):
        from reranker_models import sha256_file
        root=Path(root);(root/'faiss_store').mkdir();(root/'research').mkdir()
        originals={}
        for name in ('index.faiss','index.pkl','raw_records_cache.json'):
            p=root/'faiss_store'/name;p.write_bytes(b'original '+name.encode());originals['faiss_store/'+name]=sha256_file(p)
        split={'source_hashes':originals};research=root/'research'
        (research/'split_manifest.json').write_text(json.dumps(split))
        for name in ('index_coverage.csv','index_coverage_summary.json'):(research/name).write_text('unchanged coverage')
        for name in ('index.faiss','index.pkl'):(root/'faiss_store'/name).write_bytes(b'reembedded '+name.encode())
        keys=('faiss_store/index.faiss','faiss_store/index.pkl')
        recovery={'format_version':1,'original_split_sha256':sha256_file(research/'split_manifest.json'),
            'raw_cache_sha256':originals['faiss_store/raw_records_cache.json'],'original_index_hashes':{k:originals[k] for k in keys},
            'recovered_index_hashes':{k:sha256_file(root/k) for k in keys},'coverage_matches_original':True,
            'coverage_csv_sha256':sha256_file(research/'index_coverage.csv'),'coverage_summary_sha256':sha256_file(research/'index_coverage_summary.json')}
        (research/'index_recovery_manifest.json').write_text(json.dumps(recovery));return split,research
    def test_documented_reembedding_preserves_original_split_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            split,research=self.recovery_fixture(folder)
            self.assertIsNotNone(verify_corpus_sources(split,Path(folder),research))
            self.assertEqual(split,json.loads((research/'split_manifest.json').read_text()))
    def test_recovery_cannot_hide_changed_raw_data_or_index(self):
        with tempfile.TemporaryDirectory() as folder:
            split,research=self.recovery_fixture(folder);root=Path(folder)
            (root/'faiss_store/raw_records_cache.json').write_bytes(b'changed raw data')
            with self.assertRaises(ValueError):verify_corpus_sources(split,root,research)
        with tempfile.TemporaryDirectory() as folder:
            split,research=self.recovery_fixture(folder);root=Path(folder)
            (root/'faiss_store/index.faiss').write_bytes(b'changed rebuilt index')
            with self.assertRaises(ValueError):verify_corpus_sources(split,root,research)
    def test_different_image_filenames_same_bytes_preserve_test_fold(self):
        from image_retrieval import _image_key
        with tempfile.TemporaryDirectory() as folder:
            a=Path(folder)/'a.jpg';b=Path(folder)/'b.jpg';a.write_bytes(b'identical image');b.write_bytes(a.read_bytes())
            records=[{'dataset':'DRISHTIKON','id':1,'image_url':str(a)},{'dataset':'DRISHTIKON','id':2,'image_url':str(b)}]
            split={'assignments':{'DRISHTIKON#1':{'fold':'train','group':'a'},'DRISHTIKON#2':{'fold':'test','group':'b'}},'counts':{ds:{} for ds in ACTIVE_DATASETS}}
            merge_image_aliases(split,records)
            self.assertEqual(split['assignments']['DRISHTIKON#1'],split['assignments']['DRISHTIKON#2'])
            self.assertEqual(split['assignments']['DRISHTIKON#1']['fold'],'test')
            self.assertEqual(split['image_duplicate_audit']['cross_fold_duplicate_groups_before_merge'],1)
            self.assertEqual(_image_key(str(a)),_image_key(str(b)))
    def test_shared_images_are_kept_in_one_fold(self):
        records=[{'dataset':'DRISHTIKON','id':i,'image_url':f'{i//2}.jpg'} for i in range(30)]
        split=make_split(records,42)
        for i in range(0,30,2):self.assertEqual(split['assignments'][f'DRISHTIKON#{i}']['fold'],split['assignments'][f'DRISHTIKON#{i+1}']['fold'])
        self.assertEqual(split,make_split(records,42))
    def test_artifact_original_expanded_same_group(self):
        self.assertEqual(group_key({'dataset':'DOSA','artifact':'Temple','split':'original'}),group_key({'dataset':'DOSA','artifact':'TEMPLE','split':'expanded'}))
    def test_checkpoint_requires_provenance_and_roundtrips(self):
        for kind in ('moe','simple'):
            model=create_model(kind)
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'model.pt';save_model(model,kind,path,{'split_sha256':'hash'})
                loaded,p=load_model(kind,path);self.assertEqual(p['split_sha256'],'hash')
                self.assertTrue(all(torch.equal(a,b) for a,b in zip(model.parameters(),loaded.parameters())))
    def test_legacy_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'old.pt';torch.save(create_model('moe').state_dict(),path)
            with self.assertRaises(ValueError):load_model('moe',path)
    def test_single_network_parameter_count_matches_moe(self):
        counts=[sum(p.numel() for p in create_model(k).parameters()) for k in ('moe','simple')]
        self.assertLess(abs(counts[0]-counts[1])/counts[0],.02)
    @unittest.skipUnless((ROOT/'faiss_store/raw_records_cache.json').is_file(),
                         'Full-corpus integration test requires the unbundled original data and FAISS index.')
    def test_actual_split_and_provenance_checks(self):
        cases=load_suite(ROOT/'validation_queries.json');split=json.loads((RESEARCH/'split_manifest.json').read_text())
        p={k:load_model(k)[1] for k in ('moe','simple')};checks=check_protocol(cases,split,p,load_records())
        self.assertEqual(len(checks),2);self.assertTrue(all(c['test_refs_in_training']==0 for c in checks))
        q=dict(cases[0]);q['gold_record_ref']=p['moe']['training_refs'][0]
        with self.assertRaises(ValueError):check_protocol([q],split,p,load_records())

class ReportTests(unittest.TestCase):
    def test_bootstrap_keeps_related_questions_in_one_cluster(self):
        rows=[]
        for query,delta in [('q1',1.),('q2',0.)]:
            for mode in MODES:
                for k in (3,6):rows.append({'query_id':query,'evaluation_group':'same source','task':'text','mode':mode,'k':k,'status':'ok','mrr':delta if mode=='with_moe' else 0.,'answer_f1':None})
        result=next(r for r in paired_differences(rows) if r['mode']=='with_moe' and r['k']==3)
        self.assertEqual(result['paired_queries'],2);self.assertEqual(result['independent_source_groups'],1)
        self.assertEqual(result['mean_delta'],.5);self.assertEqual(result['bootstrap_low'],.5);self.assertEqual(result['bootstrap_high'],.5)
    def case(self):return {'id':'q','task':'text','question':'q','relevant_records':['SCALE#1'],'expected_answers':['answer']}
    def result(self,q):return {'datasets_searched':list(ACTIVE_DATASETS),'conditions':{mode:{'ranked':[{'ref':'SCALE#1'}],'answers':{'3':{'answer':'answer'},'6':{'answer':'answer'}}} for mode in MODES}}
    def test_missing_labels_and_conditions(self):
        case={'id':'q','task':'text','question':'q'}
        with contextlib.redirect_stdout(io.StringIO()):report=evaluate_cases([case],self.result,self.result)
        self.assertEqual(len(report['per_question']),6);self.assertIsNone(report['per_question'][0]['precision'])
        with contextlib.redirect_stdout(io.StringIO()):bad=evaluate_cases([case],lambda q:{},self.result)
        self.assertTrue(all(r['status']=='error' for r in bad['per_question']))
    def test_reader_errors_block_affected_quality_scores(self):
        def fail(q):
            r=self.result(q);r['conditions']['with_moe']['answers']['3'].update(reader_errors=1,reader_error_details=['bad'])
            return r
        with contextlib.redirect_stdout(io.StringIO()):report=evaluate_cases([self.case()],fail,self.result)
        row=next(r for r in report['per_question'] if r['mode']=='with_moe' and r['k']==3)
        self.assertEqual(row['status'],'error');self.assertIsNone(row['exact_match'])
    def test_different_candidate_pools_fail(self):
        def fail(q):
            r=self.result(q);r['conditions']['with_moe']['ranked']=[];return r
        with contextlib.redirect_stdout(io.StringIO()):report=evaluate_cases([self.case()],fail,self.result)
        self.assertTrue(all(r['status']=='error' for r in report['per_question']))

class InterfaceTests(unittest.TestCase):
    def test_factual_associated_question_reaches_reader(self):
        from query_router import is_filter_query
        self.assertFalse(is_filter_query('Which state is associated with this festival?'))
        self.assertTrue(is_filter_query('List all artifacts associated with weddings.'))
    def test_batch_keeps_no_answer_and_later_answer_in_csv(self):
        from pipeline import save_batch_results,CulturalQAPipeline
        pipe=object.__new__(CulturalQAPipeline);pipe.ask=Mock(side_effect=[{'answer':None,'message':'No answer.'},{'answer':'next answer','score':.9}])
        results=pipe.run_batch(['first','second'],show_progress=False)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'results.csv';save_batch_results(results,path)
            with path.open(newline='') as f:rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),2);self.assertEqual(rows[0]['message'],'No answer.');self.assertEqual(rows[1]['answer'],'next answer')
    def test_excluded_image_query_does_not_search(self):
        from image_retrieval import ImageSearch
        search=object.__new__(ImageSearch);search.pool=Mock()
        self.assertEqual(search.search('Find images in GIMMICK',verbose=False),[])
        search.pool.retrieve.assert_not_called()
    def test_review_requires_named_human_and_preserves_source_suite(self):
        from apply_annotation_reviews import apply_reviews
        suite={'queries':[{'id':'q','task':'text','question':'Question','expected_answers':['old'],'relevant_records':['DOSA#1'],'human_reviewed':False}]}
        row={'query_id':'q','question':'Question','accepted_answers':'["new"]','human_reviewed':'True','reviewer':'','additional_relevant_refs':'DOSA#2'}
        with self.assertRaises(ValueError):apply_reviews(suite,[row],{'DOSA#1','DOSA#2'})
        row['reviewer']='Test reviewer';result=apply_reviews(suite,[row],{'DOSA#1','DOSA#2'})
        self.assertEqual(result['queries'][0]['relevant_records'],['DOSA#1','DOSA#2'])
        self.assertEqual(suite['queries'][0]['expected_answers'],['old'])
        self.assertFalse(result['queries'][0]['relevance_complete'])
    def test_unknown_or_duplicate_review_rows_fail(self):
        from apply_annotation_reviews import apply_reviews
        suite={'queries':[{'id':'q','task':'text','question':'Q'}]}
        with self.assertRaises(ValueError):apply_reviews(suite,[{'query_id':'missing'}],set())
        row={'query_id':'q','question':'Q','human_reviewed':'False'}
        with self.assertRaises(ValueError):apply_reviews(suite,[row,row],set())

if __name__=='__main__':unittest.main()
