"""Regression checks for score preservation and the additive reranker."""
import json
import unittest
import torch
from langchain_core.documents import Document
from project_paths import ROOT
from reranker_models import sha256_file
from residual_reranker import ResidualReranker, combine_scores, standardize, conservative_order


class ResidualTests(unittest.TestCase):
    def test_cli_forwards_identical_text_and_image_budgets(self):
        import main
        import residual_pipeline  # Load the real subclass before patching its parent.
        from unittest.mock import patch,MagicMock
        factory=MagicMock()
        factory.return_value.ask.return_value={'answer':'example'}
        args=['main.py','--ask','example question','--candidate_k','12','--top_k','6',
              '--image_candidates','18','--image_top_k','6','--enable_images','--no_query_expansion']
        settings=[]
        with patch('sys.argv',args),patch('pipeline.CulturalQAPipeline',factory),\
             patch('residual_pipeline.ConservativePipeline',factory),patch('builtins.print'):
            for mode in (False,True):
                main.run_cli(force_moe_rescore=mode)
                settings.append(factory.call_args.kwargs)
        for setting in settings:
            self.assertEqual(setting['candidate_k'],12)
            self.assertEqual(setting['top_k'],6)
            self.assertEqual(setting['image_candidate_k'],18)
            self.assertEqual(setting['image_top_k'],6)
            self.assertTrue(setting['enable_images'])
            self.assertFalse(setting['use_query_expansion'])
        self.assertEqual({k:v for k,v in settings[0].items() if k!='force_moe_rescore'},
                         {k:v for k,v in settings[1].items() if k!='force_moe_rescore'})

    def test_conservative_mode_preserves_evidence_budgets(self):
        generator=torch.Generator().manual_seed(567)
        for n in (3,6,12):
            for _ in range(20):
                baseline=torch.randn(n,generator=generator)
                fusion=torch.randn(n,generator=generator)
                order=conservative_order(baseline,fusion,10.,10.)
                base=baseline.argsort(descending=True,stable=True).tolist()
                self.assertEqual(set(order),set(base))
                for k in (3,6): self.assertEqual(set(order[:k]),set(base[:k]))

    def test_near_ties_do_not_chain_across_large_score_range(self):
        order=conservative_order([3.,2.,1.,0.],[0.,1.,2.,3.],10.,1.)
        self.assertEqual(order,[1,0,2,3])

    def test_conservative_reader_tie_breaking_matches_baseline(self):
        from reader import ExtractiveReader
        from residual_pipeline import ConservativePipeline
        from residual_reranker import ConservativeReranker
        from types import SimpleNamespace
        docs=[Document(page_content=str(i),metadata={'record_id':i,'dataset':'DOSA',
              'cross_encoder_score':1.}) for i in range(3)]
        records={str(i):{'ref':str(i),'answer':str(i),'score':.5,'status':'ok','reader_ms':0.} for i in range(3)}
        reranker=ConservativeReranker.__new__(ConservativeReranker)
        reranker.alpha=.25;reranker.max_span=.05;reranker.protected_cutoffs=(3,6)
        reranker.fusion_scores=lambda _:torch.tensor([0.,1.,2.])
        self.assertNotEqual([d.page_content for d in reranker.rank(docs,None)],['0','1','2'])
        pipe=ConservativePipeline.__new__(ConservativePipeline)
        pipe.top_k=3;pipe.force_moe_rescore=True
        pipe.retriever=SimpleNamespace(invoke=lambda _:docs)
        pipe.reader=SimpleNamespace(answer=lambda q,items:ExtractiveReader.select_answer([records[d.page_content] for d in items]))
        pipe._moe_rescore=lambda q,items:reranker.rank(items,None)
        pipe._trace=lambda result,items:result
        result=pipe._rag_answer('tied reader question')
        self.assertEqual(result['answer'],'0')
        self.assertEqual(result['contexts_checked'],3)

    def test_image_fallback_is_explicit_and_does_not_run_moe(self):
        from residual_pipeline import ClipFallbackSearch
        from unittest.mock import patch
        search=ClipFallbackSearch.__new__(ClipFallbackSearch)
        with patch('image_retrieval.ImageSearch.search',return_value=[{'moe_rescore_used':False}]) as parent:
            result=search.search('image query',use_moe=True,verbose=False)
        parent.assert_called_once_with('image query',use_moe=False,verbose=False)
        self.assertFalse(result[0]['moe_rescore_used'])
        self.assertEqual(result[0]['ranking_policy'],'clip_fallback')

    def test_zero_weight_preserves_baseline_and_ties(self):
        base = torch.tensor([5., 5., 0., -2.])
        scores = combine_scores(base, [-10., 10., 20., 30.], 0.)
        self.assertEqual(scores.argsort(descending=True, stable=True).tolist(), [0,1,2,3])

    def test_constant_fusion_does_not_change_ranking(self):
        scores = combine_scores([3., 2., 1.], [7.,7.,7.], .25)
        self.assertEqual(scores.argsort(descending=True).tolist(), [0,1,2])

    def test_close_baseline_scores_can_be_corrected(self):
        scores = combine_scores([2., 1.99, -4.], [-1., 1., 0.], .25)
        self.assertEqual(scores.argmax().item(), 1)

    def test_strong_baseline_gap_survives_opposing_fusion(self):
        scores = combine_scores([9., 0., -1.], [-1.,1.,0.], .25)
        self.assertEqual(scores.argmax().item(), 0)

    def test_invalid_scores_fail_instead_of_silent_misranking(self):
        for alpha in [-1., float('nan'), float('inf')]:
            with self.assertRaises(ValueError): combine_scores([1.,2.], [2.,1.], alpha)
        with self.assertRaises(ValueError): combine_scores([1.,float('nan')], [1.,2.], .25)
        with self.assertRaises(ValueError): combine_scores([1.,2.], [1.], .25)
        self.assertTrue(torch.isfinite(standardize(torch.ones(1))).all())

    def test_rank_preserves_records_and_records_both_scores(self):
        model = ResidualReranker.__new__(ResidualReranker)
        model.alpha = .25
        model.fusion_scores = lambda _: torch.tensor([-1.,1.,0.])
        docs = [Document(page_content=f'body {i}', metadata={'record_id':i,'cross_encoder_score':s})
                for i,s in enumerate([2.,1.99,-4.])]
        ranked = model.rank(docs, None)
        self.assertEqual([d.metadata['record_id'] for d in ranked], [1,0,2])
        self.assertEqual({d.page_content for d in ranked},{d.page_content for d in docs})
        self.assertEqual(ranked[0].metadata['cross_encoder_score'], 1.99)
        self.assertEqual(ranked[0].metadata['ranking_policy'], 'ce_plus_moe_residual')
        self.assertNotIn('ranking_score',docs[0].metadata)

    def test_original_training_models_and_benchmark_preserved(self):
        plan=json.loads((ROOT/'experiments/moe_release/release_plan.json').read_text())
        for name in ('moe_fusion.py','reader.py','retrieval.py','research_train.py','reranker_models.py',
                     'pipeline.py','validation.py','checkpoints/moe_fusion.pt','checkpoints/simple_fusion.pt',
                     'validation_large_120/validation_report.json','validation_queries_120.json',
                     'research/training_pairs.json','research/split_manifest.json'):
            self.assertEqual(sha256_file(ROOT/name),plan['source_hashes'][name],name)


if __name__=='__main__': unittest.main()
