"""Commands run within this separate copy; original project is never written."""
import argparse,json,os
from project_paths import ROOT
from dataset_config import ACTIVE_DATASETS
VALID_TAGS=list(ACTIVE_DATASETS)
def run_cli(force_moe_rescore=None):
    os.chdir(ROOT)
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env')
    import torch
    torch.set_num_threads(2)
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group()
    m.add_argument('--ask');m.add_argument('--questions_file');m.add_argument('--image_query');m.add_argument('--image_questions_file')
    m.add_argument('--validation_file');m.add_argument('--ablation_file');m.add_argument('--build',action='store_true')
    p.add_argument('--dataset',choices=VALID_TAGS,help='Hint; all available datasets remain searchable.')
    p.add_argument('--top_k',type=int,choices=(3,6),default=3);p.add_argument('--candidate_k',type=int,default=6)
    p.add_argument('--no_query_expansion',action='store_true');p.add_argument('--enable_images',action='store_true')
    p.add_argument('--image_top_k',type=int);p.add_argument('--image_candidates',type=int);p.add_argument('--output')
    p.add_argument('--validation_output_dir',default='validation_results');p.add_argument('--data_dir',default=str(ROOT/'data'))
    a=p.parse_args()
    if a.candidate_k<a.top_k:p.error('candidate_k must be >= top_k')
    image_top_k=a.top_k if a.image_top_k is None else a.image_top_k
    image_candidates=a.candidate_k if a.image_candidates is None else a.image_candidates
    if not 1<=image_top_k<=image_candidates:p.error('Require 1 <= image_top_k <= image_candidates')
    if a.validation_file:
        from validation import run_validation
        run_validation(a.validation_file,a.validation_output_dir,candidate_k=max(6,a.candidate_k),use_query_expansion=not a.no_query_expansion);return
    if a.image_query or a.image_questions_file:
        from image_retrieval import load_image_search,run_image_questions
        search=load_image_search(dataset_filter=a.dataset,use_moe=bool(force_moe_rescore),top_k=image_top_k,candidate_k=image_candidates)
        questions=[a.image_query] if a.image_query else [x.strip() for x in open(a.image_questions_file) if x.strip()]
        run_image_questions(search,questions,bool(force_moe_rescore),a.output);return
    from pipeline import ask,run_questions_file,run_ablation_file,CulturalQAPipeline,print_answer,build_index
    if a.build:build_index(a.data_dir);return
    settings=dict(dataset_filter=a.dataset,enable_images=a.enable_images,force_moe_rescore=force_moe_rescore,
                  top_k=a.top_k,candidate_k=a.candidate_k,use_query_expansion=not a.no_query_expansion)
    if a.questions_file:run_questions_file(a.questions_file,a.output,**settings)
    elif a.ablation_file:run_ablation_file(a.ablation_file,a.output,**settings)
    elif a.ask:ask(a.ask,**settings)
    else:
        pipe=CulturalQAPipeline(**settings)
        while True:
            q=input('Ask (exit to stop): ').strip()
            if q.casefold() in ('exit','quit'):break
            print_answer(pipe.ask(q))
if __name__=='__main__':run_cli()
