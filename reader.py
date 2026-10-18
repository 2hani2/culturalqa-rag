"""Context-only joint span extraction, overflow windows and visible errors."""
import time
import torch
from transformers import AutoTokenizer,AutoModelForQuestionAnswering
from validation_metrics import evidence_ref
READER_MODEL='deepset/roberta-base-squad2'
MAX_SEQ_LENGTH=384
READER_VERSION='context-joint-span-v2'

def best_span(start,end,valid,max_answer_length=30):
    length=len(start); positions=torch.arange(length,device=start.device)
    allowed=(positions[None,:]>=positions[:,None]) & ((positions[None,:]-positions[:,None])<max_answer_length)
    allowed &= valid[:,None] & valid[None,:]
    scores=(start[:,None]+end[None,:]).masked_fill(~allowed,float('-inf'))
    value,index=scores.flatten().max(0)
    if not torch.isfinite(value):return None
    return int(index)//length,int(index)%length,float(value)

class ExtractiveReader:
    def __init__(self,model_name=READER_MODEL,null_score_diff_threshold=0.,max_answer_length=30):
        self.tokenizer=AutoTokenizer.from_pretrained(model_name,use_fast=True)
        if not self.tokenizer.is_fast:raise ValueError('Reader requires a fast tokenizer for context offsets.')
        self.model=AutoModelForQuestionAnswering.from_pretrained(model_name).eval()
        self.model_name=model_name;self.null_score_diff_threshold=null_score_diff_threshold
        self.max_answer_length=max_answer_length

    @torch.no_grad()
    def _answer_one(self,question,context):
        if not context.strip():return {'answer':None,'score':0.,'status':'no_answer','windows':0}
        batch=self.tokenizer(question,context,return_tensors='pt',padding=True,truncation='only_second',
            max_length=MAX_SEQ_LENGTH,stride=128,return_offsets_mapping=True,return_overflowing_tokens=True)
        sequences=[batch.sequence_ids(i) for i in range(len(batch['input_ids']))]
        offsets=batch.pop('offset_mapping');batch.pop('overflow_to_sample_mapping',None)
        output=self.model(**batch); answers=[]
        for window,sequence in enumerate(sequences):
            valid=torch.tensor([sid==1 and int(offsets[window,i,1])>int(offsets[window,i,0]) for i,sid in enumerate(sequence)])
            start=output.start_logits[window];end=output.end_logits[window]
            span=best_span(start,end,valid,self.max_answer_length)
            if span is None:continue
            s,e,logits=span
            cls=int((batch['input_ids'][window]==self.tokenizer.cls_token_id).nonzero()[0])
            null=float(start[cls]+end[cls]); diff=null-logits
            if diff>self.null_score_diff_threshold:continue
            eligible=valid.clone();eligible[cls]=True
            ps=torch.softmax(start.masked_fill(~eligible,float('-inf')),0)
            pe=torch.softmax(end.masked_fill(~eligible,float('-inf')),0)
            first=int(offsets[window,s,0]);last=int(offsets[window,e,1])
            answer=context[first:last].strip()
            if answer:answers.append({'answer':answer,'score':float(ps[s]*pe[e]),'score_diff':diff,
                                      'start_char':first,'end_char':last,'status':'ok'})
        if not answers:return {'answer':None,'score':0.,'status':'no_answer','windows':len(sequences)}
        best=max(answers,key=lambda r:r['score']);best['windows']=len(sequences);return best

    def evaluate_contexts(self,question,docs):
        results=[]
        for doc in docs:
            started=time.perf_counter()
            try:result=self._answer_one(question,doc.page_content)
            except Exception as exc:
                result={'answer':None,'score':None,'status':'error','error':f'{type(exc).__name__}: {exc}'}
            result.update(ref=evidence_ref(doc),dataset=doc.metadata.get('dataset'),
                          reader_ms=(time.perf_counter()-started)*1000)
            results.append(result)
        return results

    @staticmethod
    def select_answer(results,min_score=.01):
        errors=[r for r in results if r['status']=='error']
        valid=[r for r in results if r.get('answer') and r.get('score',0.)>=min_score and r['status']=='ok']
        if valid:answer=dict(max(valid,key=lambda r:r['score']))
        else:answer={'answer':None,'score':None,'status':'no_answer','message':'No confident answer span found in retrieved contexts.'}
        answer['reader_errors']=len(errors);answer['reader_error_details']=[r.get('error') for r in errors]
        answer['contexts_checked']=len(results);answer['reader_ms']=sum(r['reader_ms'] for r in results)
        if errors:answer['evaluation_status']='reader_error'
        return answer

    def answer(self,question,candidate_chunks,min_score=.01):
        return self.select_answer(self.evaluate_contexts(question,candidate_chunks),min_score)
