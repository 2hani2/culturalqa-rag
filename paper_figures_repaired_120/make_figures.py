"""Verified, descriptive CulturalQA figures. No model runs or data edits.

Run with Python 3.12 plus matplotlib/numpy. Source reports remain unchanged.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import textwrap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOCAL_LIBS = ROOT.parents[1] / 'work/paper_plot_dependencies'
if LOCAL_LIBS.is_dir() and sys.version_info[:2] == (3, 12):
    sys.path.insert(0, str(LOCAL_LIBS))
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.plot_cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch, FancyBboxPatch
from matplotlib.ticker import PercentFormatter

INK, MUTED = '#17344A', '#536878'
COLORS = ['#0072B2', '#D55E00', '#009E73']
HATCHES = ['', '//', '..']
NAMES = ['Without MoE (fresh)', 'Original MoE (historical)', 'Conservative MoE (fresh)']
DATASETS = ['DOSA', 'SCALE', 'DRISHTIKON', 'DIWALI', 'Indian Cultural Dataset']
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.titlesize':12,
    'axes.titleweight':'bold', 'axes.labelsize':10, 'xtick.labelsize':10, 'ytick.labelsize':10,
    'axes.spines.top':False, 'axes.spines.right':False, 'axes.edgecolor':'#B3C0CB',
    'axes.labelcolor':INK, 'text.color':INK, 'xtick.color':INK, 'ytick.color':INK,
    'axes.axisbelow':True, 'grid.color':'#DCE4EA', 'grid.linewidth':.7,
    'figure.facecolor':'white', 'axes.facecolor':'white', 'savefig.facecolor':'white',
    'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none'})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_sources():
    paths = {'fresh':ROOT/'validation_conservative_live_120/validation_report.json',
             'historical':ROOT/'validation_large_120/validation_report.json'}
    reports = {key:json.loads(path.read_text()) for key,path in paths.items()}
    verified = 0
    for report in reports.values():
        rows = report['per_question']
        assert len(rows) == 720 and all(r['status'] == 'ok' for r in rows)
        for summary in report['summary']:
            group = [r for r in rows if all(r[k] == summary[k] for k in ('task','mode','k'))
                     and (summary['modality'] == 'all' or r['modality'] == summary['modality'])
                     and (summary['target_dataset'] == 'all' or r['target_dataset'] == summary['target_dataset'])]
            assert len(group) == summary['queries']
            for metric in ('hit_rate','mrr','ndcg','exact_match','answer_f1','retrieval_ms',
                           'fusion_features_ms','reranker_ms','reader_ms'):
                vals = [r[metric] for r in group if r[metric] is not None]
                expected = sum(vals)/len(vals) if vals else None
                assert expected == summary[metric], (summary, metric)
                verified += 1
    old = {(r['query_id'],r['k']):r for r in reports['historical']['per_question'] if r['mode']=='without_moe'}
    for r in reports['fresh']['per_question']:
        if r['mode']=='without_moe':
            for metric in ('hit_rate','mrr','ndcg','exact_match','answer_f1'):
                assert r[metric] == old[r['query_id'],r['k']][metric]
    metadata = {key:{'path':str(path.relative_to(ROOT)),'sha256':sha(path),
                     'created_at':reports[key]['created_at']} for key,path in paths.items()}
    return reports, metadata, verified


REPORTS, SOURCES, VERIFIED_MEANS = load_sources()
CONDITIONS = [('fresh','without_moe'),('historical','with_moe'),('fresh','with_moe')]


def summary(i, task='text', k=3, ds='all', modality='all'):
    source, mode = CONDITIONS[i]
    matches = [r for r in REPORTS[source]['summary'] if (r['task'],r['mode'],r['k'],r['target_dataset'],r['modality'])
               == (task,mode,k,ds,modality)]
    assert len(matches)==1
    return matches[0]


def heading(fig, title, subtitle):
    fig.text(.065,.955,title,fontsize=18,weight='bold',va='top')
    fig.text(.065,.900,subtitle,fontsize=10.5,color=MUTED,va='top')


def footer(fig, first, second='Descriptive results on a previously inspected benchmark; no claim of MoE superiority.'):
    fig.text(.065,.066,first,fontsize=9,color=MUTED,va='bottom')
    fig.text(.065,.035,second,fontsize=9,color=MUTED,va='bottom')


def legend(fig, names=NAMES, y=.835):
    fig.legend(handles=[Patch(facecolor=c,edgecolor=INK,lw=.5,hatch=h,label=n)
                       for c,h,n in zip(COLORS,HATCHES,names)],
               loc='upper center',bbox_to_anchor=(.53,y),ncol=3,frameon=False,
               fontsize=9.5,handlelength=2,columnspacing=1.5)


def bars(ax, groups, values, percent=False):
    x = np.arange(len(groups))
    scale = 100 if percent else 1
    for i in range(3):
        yy = np.array(values[i])*scale
        patches = ax.bar(x+(i-1)*.25,yy,.23,color=COLORS[i],edgecolor=INK,
                         linewidth=.45,hatch=HATCHES[i])
        ax.bar_label(patches,labels=[f'{v:.1f}' if percent else f'{v:.3f}' for v in yy],
                     padding=4,fontsize=9 if len(groups)<4 else 8.5)
    ax.set_xticks(x,groups);ax.tick_params(axis='x',length=0,pad=9)
    ax.set_ylim(0,1.13*scale);ax.set_yticks(np.linspace(0,scale,6));ax.grid(axis='y')
    if percent:ax.yaxis.set_major_formatter(PercentFormatter(100,decimals=0))


def overview():
    fig,ax=plt.subplots(figsize=(11,6.2))
    heading(fig,'Text results: the repaired mode matches the baseline','100 text questions | final top 3 | 12 candidates | query expansion ON')
    legend(fig)
    fields=['hit_rate','exact_match','answer_f1']
    bars(ax,['Evidence Hit@3','Answer exact match','Answer token F1'],
         [[summary(i)[f] for f in fields] for i in range(3)],percent=True)
    ax.set_ylabel('Score (%)')
    footer(fig,'Conservative MoE preserves baseline top-3/top-6 evidence; equal answer quality is expected by design.')
    fig.subplots_adjust(left=.09,right=.97,bottom=.23,top=.73)
    return fig


def retrieval(task='text'):
    fig,axes=plt.subplots(1,2,figsize=(11,6.2),sharey=True)
    image = task=='image'
    heading(fig,'Image retrieval: repaired mode uses CLIP fallback' if image else 'Text evidence retrieval at top 3 and top 6',
            '20 image queries | 12 CLIP candidates | no query expansion or answer reader' if image else
            '100 text questions / 95 source groups | identical candidate budgets and cached query expansions')
    names = ['CLIP baseline (fresh)',NAMES[1],'Repaired mode: CLIP fallback'] if image else NAMES
    legend(fig,names)
    for ax,k in zip(axes,(3,6)):
        bars(ax,['Hit rate','MRR','nDCG'],[[summary(i,task,k)[m] for m in ('hit_rate','mrr','ndcg')] for i in range(3)])
        ax.set_title(f'Top {k}',pad=9)
    axes[0].set_ylabel('Mean metric value (0-1)')
    footer(fig,'Equal image scores are a fallback result, not an improvement from image MoE.' if image else
           'Known positive evidence labels are incomplete; these are not exhaustive corpus relevance judgements.')
    fig.subplots_adjust(left=.09,right=.97,bottom=.23,top=.70,wspace=.16)
    return fig


def datasets():
    fig,axes=plt.subplots(1,2,figsize=(11,6.8))
    heading(fig,'Text performance by labelled source dataset','20 questions per dataset | all five datasets searched for each question | final top 3')
    labels=['Without\nMoE','Original\nMoE','Conservative\nMoE']
    for ax,metric,title in zip(axes,('hit_rate','answer_f1'),('Evidence Hit@3 (%)','Answer token F1 (%)')):
        data=np.array([[summary(i,ds=ds)[metric]*100 for i in range(3)] for ds in DATASETS])
        im=ax.imshow(data,vmin=0,vmax=100,cmap='cividis',aspect='auto')
        ax.set_title(title,pad=12);ax.set_xticks(range(3),labels)
        ax.set_yticks(range(5),['DOSA','SCALE','DRISHTIKON','DIWALI','Indian Cultural\nDataset'] if ax is axes[0] else ['']*5)
        ax.tick_params(length=0)
        for r in range(5):
            for c in range(3):ax.text(c,r,f'{data[r,c]:.1f}',ha='center',va='center',fontsize=12,color='white' if data[r,c]<52 else INK)
        ax.set_xticks(np.arange(-.5,3,1),minor=True);ax.set_yticks(np.arange(-.5,5,1),minor=True)
        ax.grid(which='minor',color='white',linewidth=2);ax.tick_params(which='minor',length=0)
    color_ax=fig.add_axes([.925,.25,.015,.49]);fig.colorbar(im,cax=color_ax).set_label('Score (%)')
    footer(fig,'Indian Cultural Dataset: 20 questions from 15 source groups. Others: 20 questions from 20 groups each.',
           'Original MoE: historical run. Baseline/conservative MoE: fresh run. Baseline quality matched across runs.')
    fig.subplots_adjust(left=.20,right=.90,bottom=.24,top=.76,wspace=.17)
    return fig


def modalities():
    fig,axes=plt.subplots(1,2,figsize=(11,6.2),sharey=True)
    heading(fig,'Text questions with and without associated images','Image-associated text QA is separate from the 20 image-only retrieval queries')
    legend(fig)
    for ax,modality in zip(axes,('text_only','image_associated')):
        n=summary(0,modality=modality)['queries']
        bars(ax,['Evidence Hit@3','Answer token F1'],[[summary(i,modality=modality)[m] for m in ('hit_rate','answer_f1')] for i in range(3)],percent=True)
        ax.set_title(('Text-only records' if modality=='text_only' else 'Image-associated records')+f' (n={n})',pad=10)
    axes[0].set_ylabel('Score (%)')
    footer(fig,'Available image features were enabled in matched validation. Each panel groups questions by labelled source.')
    fig.subplots_adjust(left=.09,right=.97,bottom=.23,top=.70,wspace=.20)
    return fig


def costs():
    fig,ax=plt.subplots(figsize=(11,6.2))
    extra=sum(summary(2)[f] for f in ('fusion_features_ms','reranker_ms'))/1000
    heading(fig,f'Equal answer quality costs an extra {extra:.2f} seconds per text question',
            'Fresh run only | 100 text questions | final top 3 | mean measured processing components')
    fields=[('retrieval_ms','Retrieval','#8DA8BD'),('reader_ms','Reader','#0072B2'),
            ('fusion_features_ms','MoE features','#009E73'),('reranker_ms','MoE reranking','#E69F00')]
    left=np.zeros(2)
    for field,label,color in fields:
        values=np.array([summary(i)[field]/1000 for i in (0,2)])
        ax.barh([0,1],values,left=left,height=.44,label=label,color=color,edgecolor='white',linewidth=.6)
        for j,v in enumerate(values):
            if v>.4:ax.text(left[j]+v/2,j,f'{v:.2f} s',ha='center',va='center',fontsize=10,color='white' if color!='#8DA8BD' else INK)
        left+=values
    for j,total in enumerate(left):ax.text(total+.13,j,f'{total:.2f} s',va='center',weight='bold',fontsize=12)
    ax.set_yticks([0,1],['Without MoE','Conservative MoE']);ax.invert_yaxis()
    ax.set_xlim(0,max(left)+1);ax.set_ylim(1.6,-.6);ax.set_xlabel('Mean component sum (seconds)');ax.grid(axis='x')
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.55,.81),ncol=4,frameon=False)
    footer(fig,'Shared retrieval/reader measurements; component sums are not independently timed end-to-end latencies.',
           'Model startup and new API paraphrase generation are excluded. Timings are specific to this machine/run.')
    fig.subplots_adjust(left=.20,right=.97,bottom=.23,top=.71)
    return fig


def effects_data():
    all_data=[]
    for source in ('historical','fresh'):
        counts={'full_order':0,'top3_order':0,'top3_set':0,'lost_hits':0,'gained_hits':0,'answer_changes':0}
        for raw in REPORTS[source]['raw_results']:
            if raw['case']['task']!='text':continue
            c=raw['result']['conditions'];base=c['without_moe'];moe=c['with_moe']
            a=[r['ref'] for r in base['ranked']];b=[r['ref'] for r in moe['ranked']]
            gold=set(raw['case']['relevant_records']);bh=bool(gold&set(a[:3]));mh=bool(gold&set(b[:3]))
            counts['full_order']+=a!=b;counts['top3_order']+=a[:3]!=b[:3];counts['top3_set']+=set(a[:3])!=set(b[:3])
            counts['lost_hits']+=bh and not mh;counts['gained_hits']+=mh and not bh
            counts['answer_changes']+=base['answers']['3']['answer']!=moe['answers']['3']['answer']
        all_data.append(counts)
    return all_data


def effects():
    fig,ax=plt.subplots(figsize=(11,6.5))
    heading(fig,'What changed when MoE reranked the evidence?','100 text questions | changes relative to each run\'s baseline | final top 3')
    data=effects_data();fields=['top3_order','top3_set','lost_hits','gained_hits','answer_changes']
    labels=['Top-3 order changed','Top-3 evidence set changed','Baseline evidence hits lost',
            'Baseline misses recovered','Answer text changed']
    y=np.arange(len(fields))
    for j,(vals,color,hatch,name) in enumerate(zip(data,COLORS[1:],HATCHES[1:],NAMES[1:])):
        counts=[vals[f] for f in fields]
        b=ax.barh(y+(j-.5)*.32,counts,.28,color=color,hatch=hatch,edgecolor=INK,linewidth=.4,label=name)
        ax.bar_label(b,padding=5,fontsize=10)
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,112);ax.set_xticks(range(0,101,20))
    ax.set_xlabel('Number of questions (out of 100)');ax.grid(axis='x')
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.59,.83),ncol=2,frameon=False)
    footer(fig,'Conservative MoE changes near-tie order but cannot replace baseline top-3/top-6 evidence.',
           'An answer change is not necessarily an error. Hit losses/recoveries use the known positive evidence labels.')
    fig.subplots_adjust(left=.30,right=.96,bottom=.22,top=.73)
    return fig


def example_data():
    q=REPORTS['fresh']['raw_results'][0]['case']['question'];cards=[]
    for source,mode in CONDITIONS:
        raw=next(x for x in REPORTS[source]['raw_results'] if x['case']['question']==q)
        c=raw['result']['conditions'][mode]
        cards.append({'answer':c['answers']['3'],'ranked':c['ranked'][:3]})
    return {'question':q,'conditions':cards,'evidence':cards[0]['ranked'][0]['preview']}


def example():
    fig=plt.figure(figsize=(11,7.0));ax=fig.add_axes([0,0,1,1]);ax.set_axis_off()
    data=example_data()
    heading(fig,'Example answer output from the completed runs',textwrap.fill(data['question'],100))
    for i,(card,color) in enumerate(zip(data['conditions'],COLORS)):
        x=.065+i*.303;y=.29;w=.28;h=.48
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.009,rounding_size=.012',
                                   facecolor='#F5F8FA',edgecolor=color,linewidth=1.5))
        label=['WITHOUT MoE\nFresh run','ORIGINAL MoE\nHistorical run','CONSERVATIVE MoE\nFresh run'][i]
        fig.text(x+.016,y+h-.025,label,fontsize=11,weight='bold',va='top',color=color,linespacing=1.55)
        fig.text(x+.016,.61,'ANSWER',fontsize=8.5,color=MUTED)
        fig.text(x+.016,.565,card['answer']['answer'],fontsize=14,weight='bold')
        fig.text(x+.016,.515,f"Reader score: {card['answer']['score']:.4f}",fontsize=10)
        fig.text(x+.016,.45,'TOP-3 EVIDENCE',fontsize=8.5,color=MUTED)
        for j,row in enumerate(card['ranked']):
            fig.text(x+.016,.408-j*.041,f"{j+1}. {row['ref']}",fontsize=10.5)
    fig.text(.065,.225,'SUPPORTING SOURCE: DOSA#2',fontsize=9,weight='bold',color=MUTED)
    fig.text(.065,.185,'Artifact: Rushi konda beach. Description: A tourist beach in Vishakhapatnam. Navy air show takes place.',
             fontsize=10)
    footer(fig,'The original MoE kept the correct record but the reader chose a more confident distractor answer.',
           'Rendered saved outputs, not a terminal screenshot. Reader scores are not calibrated correctness probabilities.')
    return fig


SPECS=[
 ('01_text_quality',overview,'Text top-3 quality. Fresh conservative MoE matches the baseline by preserving its evidence. Original MoE values are historical on the same 100 questions.'),
 ('02_text_retrieval',lambda:retrieval('text'),'Text Hit, MRR and nDCG at top 3 and top 6. Relevance labels contain verified positives but are not exhaustive. Metrics shown on the 0-1 scale.'),
 ('03_dataset_breakdown',datasets,'Text results grouped by labelled source dataset. Every question searches all five datasets. There are 20 questions per dataset, with only 15 independent Indian Cultural Dataset source groups.'),
 ('04_modality_breakdown',modalities,'80 text-only and 20 image-associated text questions. These categories describe the labelled source record. They are distinct from the 20 image-only retrieval tasks.'),
 ('05_image_retrieval',lambda:retrieval('image'),'Image metrics on 20 queries. Repaired mode uses unchanged CLIP order, not a successful image-MoE reranker. Its equality with baseline follows from fallback.'),
 ('06_processing_time',costs,'Fresh-run mean processing components at top 3. Shared retrieval and per-context reader costs are reused across conditions. These sums exclude startup and are not independent end-to-end latency measurements.'),
 ('07_reranking_effects',effects,'Changes relative to the baseline on 100 text questions. Conservative MoE protects evidence membership while allowing bounded near-tie reordering. A changed answer is not necessarily a wrong answer.'),
 ('08_example_output',example,'Actual saved answer outputs for the beach question, rendered as a comparison card. Original MoE retrieved DOSA#2 but a distractor produced a more confident wrong answer. The repair retains baseline evidence and returns the baseline answer.')
]


def main():
    for name in ('png','svg','data'):(HERE/name).mkdir(parents=True,exist_ok=True)
    inputs={'sources':SOURCES,'conditions':dict(zip(NAMES,CONDITIONS)),
            'summary':{k:r['summary'] for k,r in REPORTS.items()},'reranking_effects':effects_data(),
            'example_output':example_data(),
            'limitations':['Reused, previously inspected benchmark; not a blind independent test.',
                           'Labels source-checked by the assistant, not independently reviewed by humans.',
                           'Conservative answer parity follows from evidence protection; no superiority claim.',
                           'Image mode is CLIP fallback. Simple-fusion alpha-zero fallback omitted from main plots because it duplicates baseline.',
                           'Original MoE is historical; baseline quality matched between historical and fresh runs.',
                           'Figures are descriptive point estimates; no statistical significance claim.']}
    (HERE/'data/plot_inputs.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2))
    exported=[]
    with PdfPages(HERE/'culturalqa_repaired_results.pdf') as pdf:
        pdf.infodict().update(Title='CulturalQA: conservative MoE validation figures',Author='CulturalQA project',
                             Subject='Fresh baseline parity, historical comparison, explicit CLIP fallback')
        for number,(stem,build,caption) in enumerate(SPECS,1):
            fig=build();fig.text(.975,.035,str(number),ha='right',fontsize=8.5,color=MUTED)
            pdf.savefig(fig)
            for fmt,dpi in [('png',300),('svg',None)]:
                path=HERE/fmt/f'{stem}.{fmt}';fig.savefig(path,dpi=dpi)
                exported.append(path)
            plt.close(fig)
    exported.append(HERE/'culturalqa_repaired_results.pdf')
    readme='CULTURALQA GRAPH OUTPUTS - verified 17 October 2026\n\n'
    readme+='Eight figures: 300-DPI PNG, editable vector SVG, and one combined vector PDF.\n\n'
    readme+='The figures compare fresh baseline and conservative MoE against historical original MoE.\n'
    readme+='Conservative MoE preserves baseline top-3/top-6 evidence. Image mode uses CLIP fallback.\n'
    readme+='Matching baseline quality is not evidence of expert-mixing superiority.\n\n'
    readme+='FIGURE CAPTIONS\n\n'+'\n\n'.join(f'{n}. {stem}\n{caption}' for n,(stem,_,caption) in enumerate(SPECS,1))
    readme+='\n\nLIMITATIONS\n'+'\n'.join('- '+x for x in inputs['limitations'])
    readme+='\n\nSOURCES\n'+json.dumps(SOURCES,indent=2)
    readme+='\n\nREPRODUCE\nPython 3.12 with matplotlib==3.11.2 and numpy==2.5.4:\npython make_figures.py\n'
    readme+='The two source reports must be present in the parent project folder.\nNo models, API calls, training or report changes are performed by this script.\n'
    (HERE/'README.txt').write_text(readme)
    data_rows=[]
    for i,name in enumerate(NAMES):
        for task in ('text','image'):
            for k in (3,6):
                r=summary(i,task,k)
                data_rows.append({'condition':name,'task':task,'k':k,'n':r['queries'],
                                  **{f:r[f] for f in ('hit_rate','mrr','ndcg','exact_match','answer_f1','retrieval_ms','fusion_features_ms','reranker_ms','reader_ms')}})
    with (HERE/'data/overall_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=data_rows[0].keys());writer.writeheader();writer.writerows(data_rows)
    manifest={'sources':SOURCES,'independently_recomputed_summary_values':VERIFIED_MEANS,
              'baseline_quality_identical_between_runs':True,'figures':len(SPECS),
              'files':{str(p.relative_to(HERE)):sha(p) for p in exported}}
    (HERE/'verification.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'figures':len(SPECS),'summary_values_verified':VERIFIED_MEANS,'output':str(HERE)},indent=2))


if __name__=='__main__':main()
