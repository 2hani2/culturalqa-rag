"""Matched training for MoE and single-network fusion using train groups only."""
import argparse,datetime,importlib.metadata,json,random,time
from collections import Counter
from pathlib import Path
import torch
from torch.nn import functional as F
from project_paths import ROOT,RESEARCH,CHECKPOINTS
from dataset_config import ACTIVE_DATASETS
from structured_store import record_to_text
from research_protocol import load_records,record_ref,canonical,group_key,digest
from reranker_models import create_model,save_model,sha256_file,source_hash

def anchor(record):
    ds=record['dataset']
    if ds=='DRISHTIKON':return record.get('question','')
    if ds=='DOSA':return 'Which cultural artifact matches these clues? '+record.get('clues','').replace('\n',' ')
    if ds=='SCALE':return f"What is the translation of {record.get('Item','')}?"
    if ds=='DIWALI':return f"Describe {record.get('concept','')} from {record.get('state','')}."
    lines=[x.strip() for x in record.get('text','').splitlines() if x.strip()]
    title=next((x.removeprefix('Content:').strip() for x in lines if x.startswith('Content:')), lines[0] if lines else '')[:150]
    return f'What does the article titled {title} explain?'

def select_records(records,split,fold,cap,seed):
    counts=Counter(canonical(r.get('question')) for r in records if r['dataset']=='DRISHTIKON')
    selected=[]
    for ds in ACTIVE_DATASETS:
        pool=[r for r in records if r['dataset']==ds and split['assignments'][record_ref(r)]['fold']==fold]
        if ds=='DRISHTIKON':pool=[r for r in pool if r.get('language')=='English' and counts[canonical(r.get('question'))]==1]
        random.Random(seed+int(digest(ds+fold)[:8],16)).shuffle(pool)
        seen=set();unique=[]
        for r in pool:
            group=split["assignments"][record_ref(r)]["group"]
            if group in seen or not anchor(r).strip():continue
            seen.add(group);unique.append(r)
            if len(unique)>=cap:break
        if len(unique)<2:raise ValueError(f'Insufficient unique {fold} groups for {ds}')
        selected.extend(unique)
    return selected

def make_triplets(selected,seed,assignments=None):
    rng=random.Random(seed);out=[]
    groups={record_ref(r):assignments[record_ref(r)]["group"] if assignments else group_key(r) for r in selected}
    queries={record_ref(r):canonical(anchor(r)) for r in selected}
    for i,r in enumerate(selected):
        eligible=[n for n in selected if groups[record_ref(n)]!=groups[record_ref(r)] and queries[record_ref(n)]!=queries[record_ref(r)]]
        same=[n for n in eligible if n['dataset']==r['dataset']]
        negative=rng.choice(same if i%2==0 and same else eligible)
        out.append({'query':anchor(r),'positive_ref':record_ref(r),'negative_ref':record_ref(negative)})
    return out

def train(train_per_dataset=60,dev_per_dataset=12,epochs=8,seed=42,enable_images=True):
    torch.set_num_threads(2);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    split_path=RESEARCH/'split_manifest.json'
    split=json.loads(split_path.read_text());records=load_records();lookup={record_ref(r):r for r in records}
    print('[train] loaded split and active records',flush=True)
    train_records=select_records(records,split,'train',train_per_dataset,seed)
    dev_records=select_records(records,split,'dev',dev_per_dataset,seed)
    triplets=make_triplets(train_records,seed,split["assignments"]);dev=make_triplets(dev_records,seed+1,split["assignments"])
    for fold,examples in [('train',triplets),('dev',dev)]:
        assert all(split['assignments'][t[key]]['fold']==fold for t in examples for key in ('positive_ref','negative_ref'))
    training_data={'train':triplets,'dev':dev,'label_type':'weak source-derived pairs; sampled negatives are not human relevance judgements'}
    data_path=RESEARCH/'training_pairs.json';data_path.write_text(json.dumps(training_data,ensure_ascii=False,indent=2))
    settings={'seed':seed,'epochs':epochs,'batch_size':32,'learning_rate':.001,'margin':.2,'enable_images':enable_images,
              'train_per_dataset':train_per_dataset,'dev_per_dataset':dev_per_dataset,'split_sha256':sha256_file(split_path),
              'training_pairs_sha256':sha256_file(data_path),'code_sha256':source_hash()}
    feature_key=digest(json.dumps(settings,sort_keys=True));cache=RESEARCH/f'training_features_{feature_key[:16]}.pt'
    from encoders import MultimodalEncoder,TEXT_MODEL_NAME,CLIP_MODEL_NAME
    encoder=MultimodalEncoder(enable_live_fetch=enable_images)
    if cache.exists():features=torch.load(cache,weights_only=True)
    else:
        text_set=sorted({t['query'] for t in triplets+dev} | {record_to_text(lookup[t[k]])[:2000] for t in triplets+dev for k in ('positive_ref','negative_ref')})
        vectors={}
        for start in range(0,len(text_set),32):
            batch=text_set[start:start+32];encoded=encoder.text_encoder.encode(batch)
            vectors.update(zip(batch,encoded));print(f'[train features] {min(start+32,len(text_set))}/{len(text_set)} texts',flush=True)
        images={}
        for ref in {t[k] for t in triplets+dev for k in ('positive_ref','negative_ref')}:
            # Both positive and negative records use their own available image.
            images[ref]=encoder.image_encoder.encode(lookup[ref].get('image_url'))[0]
        features={}
        for fold,examples in [('train',triplets),('dev',dev)]:
            features[fold]={
                'qt':torch.stack([vectors[t['query']] for t in examples]),
                'qi':torch.zeros(len(examples),512),
                'pt':torch.stack([vectors[record_to_text(lookup[t['positive_ref']])[:2000]] for t in examples]),
                'pi':torch.stack([images[t['positive_ref']] for t in examples]),
                'nt':torch.stack([vectors[record_to_text(lookup[t['negative_ref']])[:2000]] for t in examples]),
                'ni':torch.stack([images[t['negative_ref']] for t in examples])}
        torch.save(features,cache)
    versions={name:importlib.metadata.version(name) for name in ('torch','transformers','faiss-cpu','sentence-transformers')}
    results={}
    for kind in ('simple','moe'):
        model=create_model(kind,seed);optimizer=torch.optim.Adam(model.parameters(),lr=settings['learning_rate']);history=[]
        best=float('inf');started=time.perf_counter()
        def loss_on(data,indices,balance):
            fused=[];aux=[]
            for prefix in ('q','p','n'):
                fused.append(model(data[prefix+'t'][indices],data[prefix+'i'][indices]))
                if kind=='moe':aux.append(model.load_balancing_loss())
            positive=F.cosine_similarity(fused[0],fused[1],dim=-1);negative=F.cosine_similarity(fused[0],fused[2],dim=-1)
            hinge=F.relu(.2-positive+negative).mean()
            regularizer=sum(aux)/len(aux) if aux else torch.tensor(0.)
            return hinge+(0.01*regularizer if balance else 0.),float((positive>negative).float().mean())
        for epoch in range(1,epochs+1):
            model.train();generator=torch.Generator().manual_seed(seed+epoch)
            order=torch.randperm(len(triplets),generator=generator);losses=[]
            for start in range(0,len(order),32):
                optimizer.zero_grad();loss,_=loss_on(features['train'],order[start:start+32],True)
                loss.backward();optimizer.step();losses.append(float(loss.detach()))
            model.eval()
            with torch.no_grad():dev_loss,accuracy=loss_on(features['dev'],torch.arange(len(dev)),False)
            row={'epoch':epoch,'train_loss':sum(losses)/len(losses),'dev_hinge':float(dev_loss),'dev_pairwise_accuracy':accuracy}
            history.append(row);print(f'[train {kind}] {row}',flush=True)
            if float(dev_loss)<best:
                best=float(dev_loss)
                provenance={**settings,'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'active_datasets':list(ACTIVE_DATASETS),'training_fold':'train','selection_fold':'dev','test_fold_used':False,
                    'selected_epoch':epoch,'selection_metric':'minimum development triplet hinge',
                    'train_counts':dict(Counter(r['dataset'] for r in train_records)),'dev_counts':dict(Counter(r['dataset'] for r in dev_records)),
                    'training_refs':sorted({t[k] for t in triplets for k in ('positive_ref','negative_ref')}),
                    'train_query_sha256':[digest(canonical(t['query'])) for t in triplets],
                    'trainable_parameters':sum(p.numel() for p in model.parameters()),
                    'frozen_encoders':{'text':TEXT_MODEL_NAME,'image':CLIP_MODEL_NAME},
                    'encoder_revisions':{'text':getattr(encoder.text_encoder.model.config,'_commit_hash',None),'image':getattr(encoder.image_encoder.model.config,'_commit_hash',None)},
                    'versions':versions,'negative_image_policy':'own image if present; zero only if absent or disabled',
                    'training_label_type':training_data['label_type']}
                save_model(model,kind,CHECKPOINTS/f'{kind}_fusion.pt',provenance)
        results[kind]={'history':history,'training_seconds':time.perf_counter()-started,'checkpoint':str(CHECKPOINTS/f'{kind}_fusion.pt')}
    (RESEARCH/'training_history.json').write_text(json.dumps(results,indent=2))
    return results

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train_per_dataset',type=int,default=60);parser.add_argument('--dev_per_dataset',type=int,default=12)
    parser.add_argument('--epochs',type=int,default=8);parser.add_argument('--seed',type=int,default=42);parser.add_argument('--text_only',action='store_true')
    a=parser.parse_args();train(a.train_per_dataset,a.dev_per_dataset,a.epochs,a.seed,not a.text_only)
