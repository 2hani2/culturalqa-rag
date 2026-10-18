"""Select bounded near-tie policies on the existing 60 development queries only."""
import sys,json,datetime
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F
from residual_reranker import conservative_order
from reranker_models import create_model,sha256_file


def measures(orders, records):
    ranks=[]
    for order,row in zip(orders,records):
        refs=[row['refs'][j] for j in order]
        ranks.append(refs.index(row['positive_ref'])+1 if row['positive_ref'] in refs else None)
    return {'mrr':sum(1/r if r else 0 for r in ranks)/len(ranks),
            'hit_at_1':sum(r==1 for r in ranks)/len(ranks),'ranks':ranks}


def main():
    torch.set_num_threads(2)
    experiment=ROOT/'experiments/moe_improvement'
    cache=torch.load(experiment/'global_development_features.pt',map_location='cpu',weights_only=True)
    records=cache['records'];assert len(records)==60
    split=json.loads((ROOT/'research/split_manifest.json').read_text())['assignments']
    assert all(split[r['positive_ref']]['fold']=='dev' for r in records)
    plan={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'source':'60 existing development queries with actual global retrieval; no positive injection',
          'scope':'Only baseline near ties may change order, with top-3 and top-6 evidence membership protected.',
          'alphas':[0.,.05,.1,.25,.5], 'max_spans':[0.,.01,.025,.05,.1,.2,.5],
          'protected_cutoffs':[3,6],
          'selection':'Require positive development MRR gain and no individual labelled-rank regression, then prefer minimum near-tie span, alpha, changed query count. Otherwise use the explicit alpha-zero fallback. Also retain the unconstrained maximum-MRR choice.',
          'test_labels_used_for_selection':False,
          'development_features_sha256':sha256_file(experiment/'global_development_features.pt')}
    (HERE/'conservative_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    baseline=[r['cross_encoder_scores'].argsort(descending=True,stable=True).tolist() for r in records]
    base=measures(baseline,records)
    models=[ROOT/'checkpoints'/f'{k}_fusion.pt' for k in ('simple','moe')]+sorted(experiment.glob('*_tau*.pt'))
    trials=[]
    for path in models:
        payload=torch.load(path,map_location='cpu',weights_only=True)
        model=create_model(payload['model_kind']);model.load_state_dict(payload['state_dict']);model.eval()
        with torch.no_grad():
            scores=[]
            for row in records:
                f=row['features'];q=F.normalize(model(f['qt'],f['qi']),dim=-1)
                d=F.normalize(model(f['dt'],f['di']),dim=-1);scores.append((q@d.T)[0])
        for span in plan['max_spans']:
            for alpha in plan['alphas']:
                orders=[conservative_order(r['cross_encoder_scores'],s,alpha,span) for r,s in zip(records,scores)]
                assert all(set(a[:k])==set(b[:k]) for a,b in zip(baseline,orders) for k in (3,6))
                trials.append({'kind':payload['model_kind'],'checkpoint':str(path.relative_to(ROOT)),
                    'checkpoint_sha256':sha256_file(path),'alpha':alpha,'max_span':span,
                    'changed_queries':sum(a!=b for a,b in zip(baseline,orders)), **measures(orders,records)})
        print('[dev policy]',path.name,flush=True)
    selected={};max_mrr={}
    for kind in ('simple','moe'):
        options=[t for t in trials if t['kind']==kind]
        max_mrr[kind]=max(options,key=lambda t:(t['mrr'],t['hit_at_1'],-t['max_span'],-t['alpha'],-t['changed_queries']))
        safe=[t for t in options if t['mrr']>base['mrr']+1e-10 and
              all((1/r if r else 0)>=(1/b if b else 0) for r,b in zip(t['ranks'],base['ranks']))]
        best=min(safe,key=lambda t:(t['max_span'],t['alpha'],t['changed_queries'],-t['mrr'])) if safe else next(t for t in options if t['alpha']==0 and t['max_span']==0)
        selected[kind]={**best,'selection_fold':'dev','test_used':False,'protected_cutoffs':[3,6]}
    (HERE/'conservative_development_results.json').write_text(json.dumps({'plan':plan,'baseline':base,'trials':trials,'selected':selected,'maximum_mrr_candidates':max_mrr},indent=2)+'\n')
    (HERE/'conservative_selection.json').write_text(json.dumps(selected,indent=2)+'\n')
    print('BASELINE', {k:v for k,v in base.items() if k!='ranks'},flush=True)
    for k,v in selected.items():print('SELECTED',k,{a:b for a,b in v.items() if a!='ranks'},flush=True)


if __name__=='__main__':main()
