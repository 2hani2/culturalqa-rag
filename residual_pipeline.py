"""CulturalQA with development-selected residual MoE and explicit image fallback."""
import time
from pipeline import CulturalQAPipeline
from image_retrieval import ImageSearch
from residual_reranker import ConservativeReranker
from validation_metrics import evidence_rows, evidence_ref
from reader import READER_VERSION


class ClipFallbackSearch(ImageSearch):
    def search(self, question, use_moe=False, verbose=True):
        if verbose and use_moe:
            print('[image policy] CLIP baseline fallback; image MoE has not been validated.')
        rows = super().search(question, use_moe=False, verbose=verbose)
        for row in rows:
            row['ranking_policy'] = 'clip_fallback' if use_moe else 'clip_baseline'
            row['moe_requested'] = bool(use_moe)
        return rows


class ConservativePipeline(CulturalQAPipeline):
    def __init__(self, **settings):
        settings.setdefault('candidate_k', 12)
        settings.setdefault('image_candidate_k', settings['candidate_k'])
        super().__init__(**settings)
        self.residual = ConservativeReranker('moe')
        self.simple_residual = ConservativeReranker('simple') if settings.get('load_rerankers') else None
        self.moe = self.residual.model
        self.models['moe'] = self.moe
        if self.image_search is not None:
            self.image_search = ClipFallbackSearch(self.encoder.image_encoder,
                list(self.record_index.values()), top_k=settings.get('image_top_k',self.top_k),
                candidate_k=settings.get('image_candidate_k',self.candidate_k))

    def _moe_rescore(self, question, candidates):
        features, _ = self.encode_candidates(question, candidates)
        return self.residual.rank(candidates, features)

    def _rag_answer(self, question, use_moe_rescore=None, top_k_for_reader=None):
        use=self._resolve_use_moe_rescore(use_moe_rescore)
        docs=self.retriever.invoke(question)
        if use: docs=self._moe_rescore(question,docs)
        k=top_k_for_reader or self.top_k
        if k not in (3,6): raise ValueError('Conservative mode supports evidence budgets 3 and 6.')
        selected=docs[:k]
        # Preserve the baseline's tie breaking between equal-confidence answers.
        reader_docs=sorted(selected,key=lambda d:d.metadata['baseline_rank']) if use else selected
        result=self.reader.answer(question,reader_docs)
        result.update(path='rag',moe_rescore_used=use)
        return self._trace(result,selected)

    def _trace(self, result, docs):
        result = super()._trace(result, docs)
        if result.get('moe_rescore_used'):
            result.update(ranking_policy=self.residual.policy_name, residual_alpha=self.residual.alpha,
                          near_tie_span=self.residual.max_span, evidence_membership_preserved=True,
                          reranker_checkpoint=self.residual.selection['checkpoint'])
            for row, doc in zip(result['retrieved_candidates'],docs):
                for key in ('baseline_rank','fusion_score','residual_alpha','ranking_policy'):
                    row[key]=doc.metadata.get(key)
        return result

    def compare_validation(self, question):
        if self._try_exact_lookup(question) is not None:
            raise ValueError('Exact-ID lookup is not a retrieval evaluation.')
        docs=self.retriever.invoke(question)
        features,feature_ms=self.encode_candidates(question,docs)
        rankings={'without_moe':docs};timings={'without_moe':0.}
        if self.simple_residual is None:
            self.simple_residual=ConservativeReranker('simple')
        for mode,model in [('simple_fusion',self.simple_residual),('with_moe',self.residual)]:
            start=time.perf_counter();rankings[mode]=model.rank(docs,features)
            timings[mode]=(time.perf_counter()-start)*1000 if model.alpha else 0.
        outputs=self.reader.evaluate_contexts(question,docs)
        by_ref={r['ref']:r for r in outputs};conditions={}
        for mode,ranked in rankings.items():
            reranker = self.residual if mode == 'with_moe' else self.simple_residual
            conditions[mode]={'ranked':evidence_rows(ranked),
                'answers':{str(k):self.reader.select_answer([by_ref[evidence_ref(d)] for d in
                    (sorted(ranked[:k],key=lambda d:d.metadata['baseline_rank']) if mode!='without_moe' else ranked[:k])]) for k in (3,6)},
                'fusion_features_ms':feature_ms if mode!='without_moe' and (self.residual if mode=='with_moe' else self.simple_residual).alpha else 0.,
                'reranker_ms':timings[mode],
                'ranking_policy': ('cross_encoder_baseline' if mode == 'without_moe' else
                    reranker.policy_name if reranker.alpha else 'cross_encoder_fallback'),
                'alpha': reranker.alpha if mode != 'without_moe' else 0.,
            }
        return {'conditions':conditions,'reader_context_results':outputs,
                'candidate_texts':{evidence_ref(d):d.page_content for d in docs},
                **self.retriever.last_trace,'reader_version':READER_VERSION,
                'image_features_enabled':self.enable_images}
