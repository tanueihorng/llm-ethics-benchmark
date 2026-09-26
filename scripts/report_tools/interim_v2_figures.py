"""Evidence-based figures for the interim v2; never edits the shared figures."""
from pathlib import Path
import io
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
PAIRS = ['qwen_2b', 'qwen_4b', 'llama_3_2_3b', 'mistral_7b', 'phi4_mini']
NAMES = ['Qwen3-1.7B', 'Qwen3-4B', 'Llama-3.2-3B', 'Mistral-7B', 'Phi-4-mini']
COLORS = ['#2563eb', '#1e3a8a', '#dc2626', '#d97706', '#059669']

def png(fig):
    data = io.BytesIO()
    fig.savefig(data, format='png', dpi=220, facecolor='white')
    plt.close(fig)
    return data.getvalue()

def joint_results():
    """Plot classifier ASR and MMLU effects; no untested equivalence regions."""
    ja = json.loads((ROOT/'results_512/analysis/judge_agreement.json').read_text())
    caps = json.loads((ROOT/'results_512/analysis/pairwise_deltas.json').read_text())
    fig, ax = plt.subplots(figsize=(7.5, 6))
    fig.subplots_adjust(left=.14, right=.97, bottom=.15, top=.89)
    ax.axhline(0, color='#64748b', lw=.8); ax.axvline(0, color='#64748b', lw=.8)
    offsets = [(8,8), (8,8), (8,-16), (8,-17), (8,8)]
    for pair, name, color, offset in zip(PAIRS, NAMES, COLORS, offsets):
        h = next(v for v in ja['per_pair'] if v['pair_id']==pair)
        c = next(v for v in caps if v['pair_id']==pair and v['benchmark']=='mmlu')
        x,y = c['absolute_delta'],h['judge_harm_delta']
        ax.errorbar(x,y,xerr=[[x-c['delta_ci_lower']],[c['delta_ci_upper']-x]],
                    yerr=[[y-h['judge_harm_delta_ci_lower']],[h['judge_harm_delta_ci_upper']-y]],
                    fmt='o',color=color,capsize=3,ms=7,lw=1.2)
        ax.annotate(name,(x,y),xytext=offset,textcoords='offset points',color=color,fontsize=9)
    ax.set(xlim=(-.15,.06),ylim=(-.097,.095),xlabel='Change in MMLU accuracy (NF4 minus fp16)',
           ylabel='Change in classifier-scored HarmBench ASR')
    ax.set_title('Joint capability and harmful-compliance estimates',fontsize=12,pad=15)
    ax.grid(alpha=.15)
    fig.text(.5,.025,'Bars show paired 95% confidence intervals. Corrected tests appear in Chapter 4.',ha='center',fontsize=8)
    return png(fig)

def precision_results():
    data = json.loads((ROOT/'results_512/analysis/precision_sweep.json').read_text())['per_pair']
    fig, axes = plt.subplots(1,3,figsize=(11.5,4.6))
    fig.subplots_adjust(left=.07,right=.98,bottom=.24,top=.79,wspace=.3)
    for ax,metric,title in zip(axes,['harmbench_asr_judge','mmlu_accuracy','arc_accuracy'],
                               ['Classifier-scored HarmBench ASR','MMLU accuracy','ARC accuracy']):
        for pair,name,color in zip(PAIRS,NAMES,COLORS):
            ax.plot(range(3),[data[pair]['metrics'][metric][p] for p in ['fp16','int8','nf4']],'-o',color=color,label=name,lw=1.5,ms=4)
        ax.set_xticks(range(3),['fp16','INT8','NF4']);ax.set_title(title,fontsize=10)
        ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Rate or accuracy')
    fig.suptitle('Observed results across the three loading conditions',fontsize=14,y=.94)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.065),ncol=5,frameon=False,fontsize=9)
    fig.text(.5,.018,'Lines connect point estimates; they do not establish a causal effect of bit-width alone.',ha='center',fontsize=8)
    return png(fig)

def study_design():
    fig, ax = plt.subplots(figsize=(10,8.06));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    fig.subplots_adjust(0,0,1,1)
    def box(x,y,w,h,title,lines):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008',facecolor='#f3f6fa',edgecolor='#cbd5e1'))
        ax.text(x+.015,y+h-.032,title,fontsize=11,fontweight='bold',va='top',color='#172033')
        for i,line in enumerate(lines):ax.text(x+.015,y+h-.07-i*.029,line,fontsize=10,va='top',color='#334155')
    box(.035,.79,.93,.17,'Five matched models from four families',[
        'Qwen3-1.7B · Qwen3-4B · Llama-3.2-3B · Mistral-7B · Phi-4-mini',
        'Same checkpoint revision, prompts, seed and decoding settings within each pair.',
        'Primary generation: greedy decoding, seed 42, maximum 512 new tokens.'])
    box(.035,.575,.93,.165,'Three loading conditions',[
        'fp16 baseline versus NF4 four-bit: the primary matched comparison.',
        'LLM.int8: an additional condition, differing in both method and bit-width.',
        'Ten primary model conditions plus five INT8 conditions.'])
    box(.035,.355,.93,.17,'Four outcomes per model condition',[
        'HarmBench: 200 behaviours; official classifier measures harmful compliance.',
        'XSTest: 250 benign prompts; pre-specified regex measures over-refusal.',
        'MMLU: 300 items across six subjects. ARC-Challenge: 1,172 items.'])
    box(.035,.14,.93,.165,'Paired analysis and validation',[
        'Paired 95% bootstrap intervals, exact McNemar tests, BH correction for 20 primary tests.',
        'Three harmfulness judges and two samples labelled by one human annotator.',
        'Separate checks: refusal judge, five decoding seeds, token budget and INT8.'])
    for y in [.765,.55,.33]:ax.annotate('',xy=(.5,y-.017),xytext=(.5,y+.017),arrowprops=dict(arrowstyle='->',color='#64748b',lw=1.5))
    ax.text(.5,.065,'Interpret joint safety and capability results without inferring an internal mechanism.',ha='center',fontsize=10,color='#334155')
    return png(fig)
