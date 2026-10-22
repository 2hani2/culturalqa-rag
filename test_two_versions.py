"""Regression tests for the paper's two supported inference conditions."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock
import torch
from langchain_core.documents import Document
from project_paths import ROOT
from residual_reranker import ResidualReranker, combine_scores, standardize
from reranker_models import sha256_file
from pipeline import CulturalQAPipeline
from reader import ExtractiveReader
from validation_metrics import evidence_ref
from validation import MODES, compare_images

torch.set_num_threads(2)


class TwoVersionTests(unittest.TestCase):
    def docs(self):
        return [Document(page_content=str(i),metadata={'dataset':'DOSA','record_id':i,
                   'cross_encoder_score':s}) for i,s in enumerate([4.,3.,2.,1.99,0.,-1.,-2.])]

    def reranker(self):
        ranker=ResidualReranker.__new__(ResidualReranker)
        ranker.alpha=.25
        ranker.selection={'checkpoint':'fixture'}
        ranker.fusion_scores=lambda _:torch.tensor([0.,0.,-1.,1.,0.,0.,0.])
        return ranker

    def test_actual_selected_checkpoint_loads_without_removed_models(self):
        model=ResidualReranker()
        self.assertEqual(model.alpha,.25)
        self.assertTrue(model.selection['checkpoint'].endswith('tau0.1_fusion.pt'))
        self.assertEqual(sum(p.numel() for p in model.model.parameters()),1200451)
        self.assertFalse((ROOT/'checkpoints/simple_fusion.pt').exists())
        with torch.no_grad():
            scores=model.fusion_scores((torch.zeros(1,768),torch.zeros(1,512),torch.zeros(2,768),torch.zeros(2,512)))
        self.assertEqual(tuple(scores.shape),(2,))
        self.assertTrue(torch.isfinite(scores).all())

    def test_unrestricted_model_can_change_top_three_membership(self):
        docs=self.docs();ranked=self.reranker().rank(docs,None)
        self.assertIn('3',[d.page_content for d in ranked[:3]])
        self.assertNotIn('2',[d.page_content for d in ranked[:3]])
        self.assertEqual({d.page_content for d in docs},{d.page_content for d in ranked})
        self.assertTrue(all(d.metadata['ranking_policy']=='ce_plus_moe_residual' for d in ranked))
        self.assertNotIn('ranking_score',docs[0].metadata)

    def test_checkpoint_integrity_failure_is_not_silently_ignored(self):
        selection=json.loads((ROOT/'experiments/moe_improvement/global_selection.json').read_text())
        selection['moe']['checkpoint_sha256']='0'*64
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'selection.json';path.write_text(json.dumps(selection))
            with self.assertRaisesRegex(ValueError,'checkpoint changed'):
                ResidualReranker(selection_path=path)

    def test_development_can_select_initial_checkpoint_for_same_scoring_rule(self):
        selection={'moe':{'checkpoint':'checkpoints/moe_fusion.pt',
            'checkpoint_sha256':sha256_file(ROOT/'checkpoints/moe_fusion.pt'),
            'selection_fold':'dev','test_used':False,'alpha':.25}}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'selection.json';path.write_text(json.dumps(selection))
            model=ResidualReranker(selection_path=path)
        self.assertEqual(model.policy_name,'ce_plus_moe_residual')
        self.assertEqual(model.provenance['training_fold'],'train')

    def test_reader_receives_new_order_without_baseline_protection(self):
        pipe=CulturalQAPipeline.__new__(CulturalQAPipeline)
        pipe.force_moe_rescore=True;pipe.top_k=3
        pipe.retriever=SimpleNamespace(invoke=lambda _:self.docs())
        pipe._moe_rescore=lambda q,docs:self.reranker().rank(docs,None)
        pipe.reader=SimpleNamespace(answer=lambda q,docs:{'answer':','.join(d.page_content for d in docs)})
        pipe._trace=lambda result,docs:result
        result=pipe._rag_answer('fixture')
        self.assertIn('3',result['answer'].split(','))
        self.assertNotIn('2',result['answer'].split(','))
        baseline=pipe._rag_answer('fixture',use_moe_rescore=False)
        self.assertEqual(baseline['answer'],'0,1,2')

    def test_comparison_has_only_two_conditions_and_shared_reader_calls(self):
        pipe=CulturalQAPipeline.__new__(CulturalQAPipeline)
        pipe._try_exact_lookup=lambda _:None
        pipe.retriever=SimpleNamespace(invoke=lambda _:self.docs(),last_trace={})
        pipe.encode_candidates=lambda q,docs:(None,5.)
        pipe.residual=self.reranker();pipe.enable_images=True
        evaluate=Mock(side_effect=lambda q,docs:[{'ref':evidence_ref(d),'answer':d.page_content,
            'score':.9 if d.page_content=='3' else .2,'status':'ok','reader_ms':1.} for d in docs])
        pipe.reader=SimpleNamespace(evaluate_contexts=evaluate,select_answer=ExtractiveReader.select_answer)
        result=pipe.compare_validation('fixture')
        self.assertEqual(set(result['conditions']),set(MODES))
        self.assertEqual(set(MODES),{'without_moe','with_moe'})
        evaluate.assert_called_once()
        self.assertEqual(result['conditions']['without_moe']['answers']['3']['answer'],'0')
        self.assertEqual(result['conditions']['with_moe']['answers']['3']['answer'],'3')

    def test_cli_uses_same_settings_and_pins_only_requested_mode(self):
        import main
        settings=[]
        for mode in (False,True):
            with patch('sys.argv',['main','--ask','fixture','--enable_images','--top_k','6','--no_query_expansion']), \
                 patch('pipeline.CulturalQAPipeline') as factory, contextlib.redirect_stdout(io.StringIO()):
                factory.return_value.ask.return_value={'answer':'fixture'}
                main.run_cli(force_moe_rescore=mode)
                options=factory.call_args.kwargs
                self.assertEqual(options['force_moe_rescore'],mode)
                settings.append({k:v for k,v in options.items() if k!='force_moe_rescore'})
        self.assertEqual(settings[0],settings[1])
        self.assertEqual(settings[0]['candidate_k'],12)

    def test_validation_cli_forwards_modality_and_expansion_settings(self):
        import main
        with patch('sys.argv',['main','--validation_file','validation_queries_120.json','--enable_images','--no_query_expansion']), \
             patch('validation.run_validation') as run:
            main.run_cli(force_moe_rescore=True)
        self.assertTrue(run.call_args.kwargs['enable_images'])
        self.assertFalse(run.call_args.kwargs['use_query_expansion'])

    def test_baseline_initialization_needs_no_moe_checkpoint(self):
        store=SimpleNamespace(index_to_docstore_id={})
        retriever=SimpleNamespace(datasets_searched=[])
        with patch('pipeline.load_faiss_index',return_value=store), \
             patch('pipeline.build_full_retriever',return_value=retriever), \
             patch('pipeline.MultimodalEncoder'),patch('pipeline.ExtractiveReader'), \
             patch('pipeline._load_raw_records_cache',return_value=[]), \
             patch('pipeline.ResidualReranker',side_effect=AssertionError('baseline must not load MoE')):
            pipe=CulturalQAPipeline(force_moe_rescore=False)
        self.assertIsNone(pipe.residual)

    def test_shared_image_path_never_invokes_moe(self):
        pool=SimpleNamespace(retrieve=lambda *a,**kw:[{'image_ref':'fixture.jpg','clip_score':.8,'_records':[]}],
                             datasets_searched=['DRISHTIKON'],datasets_with_images=['DRISHTIKON'])
        pipe=SimpleNamespace(image_search=SimpleNamespace(pool=pool),enable_images=True)
        with patch('validation.image_digest',return_value='digest'):
            result=compare_images(pipe,'fixture',12)
        a,b=(result['conditions'][mode] for mode in MODES)
        self.assertEqual(a,b)
        self.assertEqual(a['ranking_policy'],'clip_shared')
        self.assertFalse(a['ranked'][0]['moe_rescore_used'])

    def test_invalid_and_constant_scores(self):
        for alpha in (-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):combine_scores([1.,2.],[2.,1.],alpha)
        with self.assertRaises(ValueError):combine_scores([1.,float('nan')],[2.,1.],.25)
        with self.assertRaises(ValueError):combine_scores([1.,2.],[1.],.25)
        self.assertTrue(torch.isfinite(standardize([1.])).all())
        self.assertEqual(combine_scores([3.,2.,1.],[7.,7.,7.],.25).argsort(descending=True).tolist(),[0,1,2])


if __name__=='__main__':
    unittest.main()
