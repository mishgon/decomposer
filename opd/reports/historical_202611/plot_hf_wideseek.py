import json,numpy as np,matplotlib.pyplot as plt,pymupdf
from matplotlib.backends.backend_pdf import PdfPages
rows=json.load(open('tmp/pdfs/hf-wideseek-token-lengths.json'))
output='output/pdf/wideseek-hf-token-budget-coverage.pdf'
summary=[]
with PdfPages(output) as pdf:
 for threshold in [.75,.9]:
  r=[x for x in rows if x['f1']>threshold];t=np.sort([x['response_tokens'] for x in r]);n=len(t);p80=t[int(np.ceil(.8*n))-1]
  fig,ax=plt.subplots(figsize=(11.7,7.3));fig.subplots_adjust(left=.09,right=.96,top=.84,bottom=.23)
  xmax=max(40960,int(np.ceil(t[-1]/8192))*8192)
  ax.step(np.r_[0,t,xmax],np.r_[0,np.arange(1,n+1)*100/n,100],where='post',color='#2563eb',lw=2.2)
  for b,col in [(8192,'#dc2626'),(16384,'#15803d'),(32768,'#7c3aed')]:
   pct=100*np.count_nonzero(t<=b)/n
   ax.axvline(b,color=col,ls='--',lw=1.3);ax.scatter([b],[pct],color=col,zorder=4,s=40)
   ax.text(b,4,f'{b//1024}K: {pct:.1f}%',rotation=90,va='bottom',ha='right',color=col,fontsize=10,bbox={'facecolor':'white','edgecolor':'none','alpha':.85})
  rate=100*np.count_nonzero(t<=p80)/n
  ax.axvline(p80,color='#b8860b',ls='--',lw=1.5);ax.scatter([p80],[rate],color='#b8860b',zorder=5,s=50)
  ax.annotate(f'Pareto optimal (80% finish rate)\n{p80:,} response tokens',xy=(p80,rate),xytext=(xmax*.57,53),color='#b8860b',fontsize=11,arrowprops={'arrowstyle':'->','color':'#b8860b'},bbox={'facecolor':'white','edgecolor':'none','alpha':.95})
  ax.set(xlim=(0,xmax),ylim=(0,102),xlabel='Cumulative response budget (student-format output + observations)',ylabel='Qualifying traces covered (%)')
  ticks=np.arange(0,xmax+1,8192);ax.set_xticks(ticks,[f'{x//1024}K' for x in ticks]);ax.set_yticks(np.arange(0,101,10));ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
  fig.suptitle(f'WideSeek teacher traces: F1 > {threshold}',fontsize=19,fontweight='bold',y=.96)
  fig.text(.5,.9,f'{n:,} published traces | train + validation pooled | Qwen3.5-4B tokenizer, non-thinking',ha='center',fontsize=11,color='#475569')
  fig.text(.09,.14,'Source: decomposer-datasets/decomposer-manager-sft, version 2.2.0; revision 3abe3ef827aa.',fontsize=9.5,color='#475569')
  fig.text(.09,.105,'Length = rendered full conversation minus initial generation prompt; includes tool observations, excludes teacher reasoning.',fontsize=9,color='#475569')
  fig.text(.09,.07,'100% means all qualifying published teacher traces, not all attempted tasks or predicted student success.',fontsize=9,color='#475569')
  fig.text(.09,.035,'Gold is the empirical 80th percentile. Raw OPD token assembly may differ slightly. Wall times are absent from this export.',fontsize=9,color='#475569')
  pdf.savefig(fig);plt.close(fig)
  summary.append({'threshold':threshold,'n':n,'p80':int(p80),'max':int(t[-1]),'coverage':{str(b):float(np.mean(t<=b)*100) for b in [8192,16384,32768]}})
print(json.dumps(summary))
d=pymupdf.open(output)
for i,p in enumerate(d):p.get_pixmap(matrix=pymupdf.Matrix(1.2,1.2)).save(f'tmp/pdfs/hf-wideseek-{i}.png')
