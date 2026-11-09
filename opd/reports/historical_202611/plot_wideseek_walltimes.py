import json,collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter,MultipleLocator
root=Path(__file__).resolve().parents[3]
d=json.loads((root/'tmp/pdfs/wideseek-walltimes.json').read_text());rows=d['records'];groups=collections.defaultdict(list)
for r in rows:groups[r['group']].append(r)
labels={
'qwen4b-width100-n3-gym-limits-decomposer':'Eval: Decomposer / 100 tasks x 3',
'qwen4b-width100-n3-gym-limits-simple':'Eval: single agent / 100 tasks x 3',
'qwen4b-width100-n3-judge35b':'Eval: judge35b run',
'prompt-8d5b07cb-coverage':'SFT collection: revised prompt',
'main-d6ea7a3d-coverage':'SFT collection: initial prompt',
'stock-qwen4b-wideseek-workplace20-twoh200-20261102-r6 / OPD':'Stock OPD r6: train + validation',
'sft-v120-wideseek-fullpool-workplace20-twoh200-20261103-r2 / OPD':'SFT-start OPD: train + validation (3 min)',
'sft-v120-wideseek-fullpool-workplace20-twoh200-20261103-r2 / baseline':'SFT-start baseline (3 min)',
}
for k in groups:
 if k.startswith('stock-') and k not in labels:
  suffix=k.split('20261102-')[1];labels[k]='Stock '+suffix.replace(' / ',' / ')
 if k.startswith('tight-paper'):labels[k]='Tight-budget eval: '+k.split(' / ')[1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(17,10.5));fig.subplots_adjust(left=.065,right=.65,top=.84,bottom=.16)
fig.text(.065,.945,'How long did WideSeek episodes take to finish?',fontsize=23,fontweight='bold',color='#1D354B')
fig.text(.065,.904,f'{len(rows):,} unique timed episodes | {len(groups)} run groups | Historical eval, SFT collection and OPD',fontsize=13,color='#526171')
fig.text(.065,.875,'Solid curves: normal finishes, any F1 score. Denominator: all timed terminal episodes in that group.',fontsize=11,color='#526171')
order=sorted(groups,key=lambda k:(-len(groups[k]),k));colors=plt.get_cmap('tab20').colors
for i,k in enumerate(order):
 rs=groups[k];ts=np.sort([r['seconds']/60 for r in rs if r['status']=='finished'])
 xx=np.r_[0,ts,50];yy=np.r_[0,np.arange(1,len(ts)+1)/len(rs)*100,len(ts)/len(rs)*100]
 ax.step(xx,yy,where='post',lw=2 if len(rs)>=100 else 1.2,alpha=.9,color=colors[i%20],label=f'{labels.get(k,k)}  [{len(ts)}/{len(rs)}]')
allts=np.sort([r['seconds']/60 for r in rows]);fin=np.sort([r['seconds']/60 for r in rows if r['status']=='finished'])
ax.step(np.r_[0,allts,50],np.r_[0,np.arange(1,len(allts)+1)*100/len(rows),100],where='post',color='#20262C',ls='--',lw=2.8,label=f'Pooled: ALL endings, including failures  [N={len(rows):,}]')
ax.step(np.r_[0,fin,50],np.r_[0,np.arange(1,len(fin)+1)*100/len(rows),len(fin)*100/len(rows)],where='post',color='#20262C',lw=3,label=f'Pooled: normal finishes  [{len(fin)}/{len(rows)}]')
for x,label in [(3,'3 min'),(15,'15 min'),(45,'45 min')]:
 ax.axvline(x,color='#7B8794',ls=':',lw=1.2,zorder=0);ax.text(x+.25,102,label,fontsize=10,color='#526171')
ax.set(xlim=(0,50),ylim=(0,106),xlabel='Episode wall time (minutes)',ylabel='Episodes finished by this time (%)')
ax.yaxis.set_major_formatter(PercentFormatter(100));ax.yaxis.set_major_locator(MultipleLocator(10));ax.xaxis.set_major_locator(MultipleLocator(5));ax.grid(alpha=.22);ax.set_axisbelow(True)
ax.legend(loc='upper left',bbox_to_anchor=(1.015,1.01),frameon=False,fontsize=9.2,handlelength=2.6,labelspacing=.9,title='Run group  [normal finishes / all timed episodes]',title_fontsize=10)
summary=[]
for t in [3,15,45]:
 normal=sum(r['status']=='finished' and r['seconds']<=t*60 for r in rows)
 ended=sum(r['seconds']<=t*60 for r in rows)
 summary.append(f'{t} min: {normal/len(rows):.1%} finished normally / {ended/len(rows):.1%} ended in any state')
fig.text(.065,.105,'   |   '.join(summary),fontsize=11,color='#1D354B')
fig.text(.065,.065,'Wall time = finished_at - started_at (includes cleanup / judging where recorded). Errors, timeouts and skips stay in denominators;\nongoing episodes without terminal results are excluded. Curves are descriptive, not a survival estimate or a controlled model comparison.',fontsize=10,color='#526171',linespacing=1.5)
fig.text(.065,.025,'Source: preserved Hertz-2 WideSeek artifacts. Deduplicated by task, mode, start time and execution identity. Snapshot: 3 Nov 2026.',fontsize=9,color='#697887')
fig.savefig(root/'output/pdf/wideseek-completion-walltime.pdf');fig.savefig(root/'tmp/pdfs/wideseek-completion-walltime.png',dpi=110)
print('\n'.join(summary));print('Normal finishes:',len(fin))
