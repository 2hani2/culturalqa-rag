"""Commands run within this separate copy; original project is never written."""
import argparse,json,os
from project_paths import ROOT
from dataset_config import ACTIVE_DATASETS
VALID_TAGS=list(ACTIVE_DATASETS)
def run_cli(force_moe_rescore=None):
    os.chdir(ROOT)
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group()
    m.add_argument('--ask');m.add_argument('--questions_file');m.add_argument('--image_query');m.add_argument('--image_questions_file')
    m.add_argument('--validation_file');m.add_argument('--ablation_file');m.add_argument('--build',action='store_true')
    p.add_argument('--dataset',choices=VALID_TAGS,help='Hint; all available datasets remain searchable.')
    p.add_argument('--top_k',type=int,choices=(3,6),default=3);p.add_argument('--candidate_k',type=int,default=12)
    p.add_argument('--no_query_expansion',action='store_true');p.add_argument('--enable_images',action='store_true')
    p.add_argument('--image_top_k',type=int);p.add_argument('--image_candidates',type=int);p.add_argument('--output')
    p.add_argument('--validation_output_dir',default='validation_results');p.add_argument('--data_dir',default=str(ROOT/'data'))
    a=p.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env')
    import torch
    torch.set_num_threads(2)
    if a.candidate_k<a.top_k:p.error('candidate_k must be >= top_k')
    image_top_k=a.top_k if a.image_top_k is None else a.image_top_k
    image_candidates=a.candidate_k if a.image_candidates is None else a.image_candidates
    if not 1<=image_top_k<=image_candidates:p.error('Require 1 <= image_top_k <= image_candidates')
    if a.validation_file:
        from validation import run_validation
        run_validation(a.validation_file,a.validation_output_dir,candidate_k=max(6,a.candidate_k),
            use_query_expansion=not a.no_query_expansion,enable_images=a.enable_images);return
    if a.image_query or a.image_questions_file:
        from image_retrieval import load_image_search,run_image_questions
        print('[image policy] Shared CLIP retrieval; the MoE comparison applies to text evidence ranking.')
        search=load_image_search(dataset_filter=a.dataset,top_k=image_top_k,candidate_k=image_candidates)
        questions=[a.image_query] if a.image_query else [x.strip() for x in open(a.image_questions_file) if x.strip()]
        run_image_questions(search,questions,a.output);return
    from pipeline import CulturalQAPipeline,print_answer,build_index,save_batch_results
    if a.build:build_index(a.data_dir);return
    settings=dict(dataset_filter=a.dataset,enable_images=a.enable_images,force_moe_rescore=force_moe_rescore,
                  top_k=a.top_k,candidate_k=a.candidate_k,use_query_expansion=not a.no_query_expansion,
                  image_top_k=image_top_k,image_candidate_k=image_candidates)
    Pipeline=CulturalQAPipeline
    print('[ranking policy] '+('Unrestricted cross-encoder + MoE.' if force_moe_rescore else 'Cross-encoder baseline (without MoE).'))
    if a.ablation_file:settings['load_rerankers']=True
    pipe=Pipeline(**settings)
    if a.questions_file:
        from pathlib import Path
        import datetime
        questions=[q.strip() for q in Path(a.questions_file).read_text().splitlines() if q.strip() and not q.lstrip().startswith('#')]
        results=pipe.run_batch(questions)
        output=a.output or ROOT/f'batch_results_{datetime.datetime.now():%Y%m%d_%H%M%S}.json'
        save_batch_results(results,output);print('[batch] saved',output)
    elif a.ablation_file:
        from pathlib import Path
        results=[{'question':q.strip(),'result':pipe.compare_validation(q.strip())} for q in Path(a.ablation_file).read_text().splitlines() if q.strip()]
        Path(a.output or ROOT/'ablation_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    elif a.ask:print_answer(pipe.ask(a.ask))
    else:
        while True:
            q=input('Ask (exit to stop): ').strip()
            if q.casefold() in ('exit','quit'):break
            print_answer(pipe.ask(q))
if __name__=='__main__':run_cli()
