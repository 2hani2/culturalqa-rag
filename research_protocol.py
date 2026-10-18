"""Seeded group splits, source fingerprints and indexed-content coverage."""
import argparse,csv,hashlib,json,random,re,unicodedata
from collections import Counter,defaultdict
from pathlib import Path
from functools import lru_cache
from project_paths import ROOT,INDEX,RESEARCH
from dataset_config import ACTIVE_DATASETS,is_active_dataset
from structured_store import record_to_text

def canonical(text):return ' '.join(unicodedata.normalize('NFKC',str(text or '')).casefold().split())
def digest(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()
def record_ref(record):return f"{record['dataset']}#{record['id']}"
def group_key(record):
    ds=record['dataset']
    if record.get('image_url'):subject=Path(record['image_url']).name.casefold()
    elif ds=='DOSA':subject=canonical(record['artifact'])
    elif ds=='SCALE':subject=canonical(record.get('Item'))
    elif ds=='DIWALI':subject=canonical(record.get('concept'))
    else:subject=digest(canonical(record_to_text(record)))
    return ds+'::'+subject
def load_records():return [r for r in json.loads((INDEX/'raw_records_cache.json').read_text()) if is_active_dataset(r.get('dataset'))]

@lru_cache(maxsize=4096)
def _image_digest(path,size,mtime_ns):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def image_digest(path):
    path=Path(path).resolve();stat=path.stat()
    return _image_digest(str(path),stat.st_size,stat.st_mtime_ns)

def merge_image_aliases(split,records):
    aliases=defaultdict(list)
    for r in records:
        path=r.get('image_url')
        if path and Path(path).is_file():aliases[image_digest(path)].append(r)
    cross_before=0
    for key,group in aliases.items():
        folds={split['assignments'][record_ref(r)]['fold'] for r in group}
        cross_before+=int(len(folds)>1)
        # Preserve every existing held-out target while removing alias leakage.
        fold='test' if 'test' in folds else 'dev' if 'dev' in folds else 'train'
        for r in group:split['assignments'][record_ref(r)]={'fold':fold,'group':r['dataset']+'::image-sha256:'+key}
    split['image_duplicate_audit']={'distinct_file_contents':len(aliases),'cross_fold_duplicate_groups_before_merge':cross_before,'cross_fold_duplicate_groups_after_merge':0,'rule':'Exact image bytes share one group. Aliases are promoted to the furthest-held-out fold; no test target is moved into training.'}
    for ds in ACTIVE_DATASETS:
        for fold in ('train','dev','test'):
            selected=[a for ref,a in split['assignments'].items() if ref.split('#')[0]==ds and a['fold']==fold]
            split['counts'][ds][fold]={'groups':len({a['group'] for a in selected}),'records':len(selected)}
    return split

def make_split(records,seed=42):
    grouped=defaultdict(lambda:defaultdict(list))
    for r in records:grouped[r['dataset']][group_key(r)].append(record_ref(r))
    assignments={};counts={}
    for ds in sorted(grouped):
        groups=sorted(grouped[ds]);random.Random(seed+int(digest(ds)[:8],16)).shuffle(groups)
        a=int(len(groups)*.7);b=int(len(groups)*.85)
        if len(groups)<3:raise ValueError(f'Need at least three groups in {ds}')
        counts[ds]={}
        for fold,selected in [('train',groups[:a]),('dev',groups[a:b]),('test',groups[b:])]:
            refs=[ref for key in selected for ref in grouped[ds][key]]
            counts[ds][fold]={'groups':len(selected),'records':len(refs)}
            for key in selected:
                for ref in grouped[ds][key]:assignments[ref]={'fold':fold,'group':key}
    return {'seed':seed,'ratios':{'train':.7,'dev':.15,'test':.15},'unit':'image/subject/exact-content group',
            'assignments':assignments,'counts':counts,'knowledge_corpus_policy':'All active knowledge documents remain searchable. Training uses train-fold records only. Development/test query groups never enter either training model.'}

def audit_coverage(records,docs):
    by_ref=defaultdict(list);report_counts=Counter()
    for d in docs:
        if d.metadata.get('record_id') is None:report_counts[d.metadata.get('dataset')]+=1
        else:by_ref[f"{d.metadata['dataset']}#{d.metadata['record_id']}"].append(d.page_content)
    rows=[]
    for r in records:
        text=record_to_text(r);intervals=[];unmatched=0
        for chunk in by_ref.get(record_ref(r),[]):
            position=text.find(chunk)
            if position>=0:intervals.append((position,position+len(chunk)))
            else:unmatched+=1
        intervals.sort();covered=0;last=0
        for first,end in intervals:covered+=max(0,end-max(first,last));last=max(last,end)
        rows.append({'ref':record_ref(r),'dataset':r['dataset'],'raw_characters':len(text),
                     'indexed_characters':covered,'coverage_fraction':covered/len(text) if text else 1.,
                     'indexed_chunks':len(by_ref.get(record_ref(r),[])),'unmatched_chunks':unmatched,
                     'status':'fully_indexed' if covered==len(text) else 'partial' if covered else 'missing'})
    summary={}
    for ds in ACTIVE_DATASETS:
        data=[r for r in rows if r['dataset']==ds];total=sum(r['raw_characters'] for r in data)
        summary[ds]={'raw_records':len(data),'records_with_indexed_content':sum(r['indexed_chunks']>0 for r in data),
            'fully_indexed_records':sum(r['status']=='fully_indexed' for r in data),
            'missing_records':sum(r['status']=='missing' for r in data),'partial_records':sum(r['status']=='partial' for r in data),
            'character_coverage':sum(r['indexed_characters'] for r in data)/total if total else None,
            'report_chunks':report_counts[ds]}
    return rows,summary

def prepare(seed=42):
    RESEARCH.mkdir(exist_ok=True);records=load_records();split=merge_image_aliases(make_split(records,seed),records)
    # These hashes identify the exact cache/index and original downloaded sources.
    from reranker_models import sha256_file
    files=[INDEX/'raw_records_cache.json',INDEX/'index.faiss',INDEX/'index.pkl']
    for ds in ACTIVE_DATASETS:
        files.extend(p for p in (ROOT/'data/raw'/ds).rglob('*') if p.is_file() and p.suffix.casefold() in ('.csv','.json','.jsonl','.xlsx','.tsv'))
    split['source_hashes']={str(p.relative_to(ROOT)):sha256_file(p) for p in sorted(files)}
    (RESEARCH/'split_manifest.json').write_text(json.dumps(split,indent=2));print('[protocol] group split saved',flush=True)
    (RESEARCH/'image_duplicate_audit.json').write_text(json.dumps(split['image_duplicate_audit'],indent=2))
    import pickle
    with (INDEX/'index.pkl').open('rb') as f:store,mapping=pickle.load(f)
    docs=[store.search(key) for key in mapping.values()]
    rows,summary=audit_coverage(records,docs)
    with (RESEARCH/'index_coverage.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (RESEARCH/'index_coverage_summary.json').write_text(json.dumps(summary,indent=2))
    with (RESEARCH/'record_splits.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['ref','dataset','fold','group'])
        for ref,a in sorted(split['assignments'].items()):writer.writerow([ref,ref.split('#')[0],a['fold'],a['group']])
    return split

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=42);a=p.parse_args();prepare(a.seed)
