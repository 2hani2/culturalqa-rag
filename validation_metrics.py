"""Binary retrieval metrics and SQuAD-style answer metrics; no models needed."""
import hashlib, math, re, string
from collections import Counter
def evidence_ref(doc):
    m=doc.metadata
    if m.get('record_id') is not None: return f"{m['dataset']}#{m['record_id']}"
    return f"{m.get('dataset','unknown')}#report:{hashlib.sha256(doc.page_content.encode()).hexdigest()[:16]}"
def evidence_rows(docs):
    return [dict(rank=i,ref=evidence_ref(d),dataset=d.metadata.get('dataset'),record_id=d.metadata.get('record_id'),
        has_image=bool(d.metadata.get('has_image')),cross_encoder_score=d.metadata.get('cross_encoder_score'),
        ranking_score=d.metadata.get('ranking_score'),preview=d.page_content[:300]) for i,d in enumerate(docs,1)]
def ranking_metrics(predictions,relevant,cutoffs=(3,6)):
    out={f'{m}@{k}':None for k in cutoffs for m in ('precision','recall','hit','mrr','ndcg')}
    if not relevant: return out
    gold=set(relevant); seen=set(); flags=[]
    for p in predictions: flags.append(int(p in gold and p not in seen)); seen.add(p)
    for k in cutoffs:
        hits=sum(flags[:k]); dcg=sum(x/math.log2(r+1) for r,x in enumerate(flags[:k],1))
        ideal=sum(1/math.log2(r+1) for r in range(1,min(k,len(gold))+1))
        out.update({f'precision@{k}':hits/k,f'recall@{k}':hits/len(gold),f'hit@{k}':float(hits>0),
            f'mrr@{k}':next((1/r for r,x in enumerate(flags[:k],1) if x),0.),f'ndcg@{k}':dcg/ideal})
    return out
def normalize_answer(answer):
    s=str(answer or '').casefold().translate(str.maketrans('','',string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',s).split())
def answer_metrics(prediction,references):
    if not references: return {'exact_match':None,'answer_f1':None}
    p=normalize_answer(prediction); em=f1=0.
    for g in map(normalize_answer,references):
        em=max(em,float(p==g)); pt=p.split(); gt=g.split(); common=sum((Counter(pt)&Counter(gt)).values())
        value=2*common/(len(pt)+len(gt)) if pt and gt else float(pt==gt)
        f1=max(f1,value)
    return {'exact_match':em,'answer_f1':f1}
