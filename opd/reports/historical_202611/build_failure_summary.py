import json,io,statistics,collections
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
root=Path(__file__).resolve().parents[3];data=json.loads((root/'tmp/pdfs/wideseek-failure-audit.json').read_text())
rows=data['wideseek'];work=data['workplace'];steps=list(range(0,65,8))
W,H=A4;M=42;CW=W-2*M
out=root/'output/pdf/wideseek-opd-failure-analysis.pdf';c=canvas.Canvas(str(out),pagesize=A4);c.setTitle('WideSeek OPD: Validation Failure Analysis');c.setAuthor('Artem Matrosov')
blue='#205C8D';red='#B74C50';gray='#526171';gold='#BC8735'
style=ParagraphStyle('body',fontName='Helvetica',fontSize=10.2,leading=14,textColor=colors.HexColor('#243646'))
def text(s,x,y,w=CW,size=10.2,color='#243646'):
 st=ParagraphStyle('p',parent=style,fontSize=size,leading=size*1.38,textColor=colors.HexColor(color));p=Paragraph(s,st);_,h=p.wrap(w,H);p.drawOn(c,x,y-h);return y-h-10
def page(n,title,sub):
 c.setFillColor(colors.HexColor(blue));c.rect(0,H-9,W,9,fill=1,stroke=0)
 c.setFont('Helvetica-Bold',20);c.drawString(M,H-52,title)
 text(sub,M,H-66,size=9,color=gray)
 c.setStrokeColor(colors.HexColor('#DCE3E9'));c.line(M,35,W-M,35)
 c.setFont('Helvetica',8);c.setFillColor(colors.HexColor(gray));c.drawString(M,22,'WideSeek OPD | Evidence audit | 3 November 2026');c.drawRightString(W-M,22,f'{n} / 3')
def table(headers,values,y,widths):
 cells=[[Paragraph(str(v),ParagraphStyle('t',parent=style,fontSize=8.8,leading=12)) for v in row] for row in [headers]+values]
 t=Table(cells,colWidths=widths,repeatRows=1);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EAF0F6')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor(blue)),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F7F9FB')]),('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7)]));_,h=t.wrap(CW,H);t.drawOn(c,M,y-h);return y-h-14
def chart(fig,y,height=215):
 b=io.BytesIO();fig.savefig(b,format='png',dpi=180,bbox_inches='tight',facecolor='white');plt.close(fig);b.seek(0);c.drawImage(ImageReader(b),M,y-height,width=CW,height=height,mask='auto');return y-height-15
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':gray,'xtick.color':gray,'ytick.color':gray})
def wr(step,src=None):return [r for r in rows if r['step']==step and (r['source']==src if src else r['source']!='wideseek/train')]
def ww(step):return [r for r in work if r['steps']==[step]]
def score(rs):return statistics.mean(r['score'] or 0 for r in rs)
page(1,'What the falling rewards mean','Completed run: stock-qwen4b-wideseek-workplace20-twoh200-20261102-r6 | 64 updates')
y=H-105
y=text('<b>Two problems coexist.</b> WideSeek rewards mostly measure failure to finish within the budget. Workplace shows an observed loss of task success, including both operational failures and completed-but-incorrect episodes.',M,y,size=12)
fig,ax=plt.subplots(1,2,figsize=(8,3))
for src,label,color in [('wideseek/train_probe','Train probe (4)',blue),('wideseek/heldout','Held-out (8)',gold)]:ax[0].plot(steps,[score(wr(s,src)) for s in steps],marker='o',label=label,color=color)
ax[0].plot(steps,[score(ww(s)) for s in steps],marker='o',label='Workplace (20)',color=red);ax[0].set(title='Recorded mean rewards',xlabel='Optimizer step',ylabel='Mean score',ylim=(0,.6));ax[0].legend(fontsize=7)
finished=[sum(r['status']=='finished' for r in wr(s)) for s in steps];timeouts=[sum(r['status']=='timeout' for r in wr(s)) for s in steps];errors=[12-a-b for a,b in zip(finished,timeouts)]
ax[1].bar(steps,finished,width=5,label='Finished',color=blue);ax[1].bar(steps,timeouts,bottom=finished,width=5,label='Timeout',color=gold);ax[1].bar(steps,errors,bottom=[a+b for a,b in zip(finished,timeouts)],width=5,label='Other failure',color=red);ax[1].set(title='WideSeek validation outcomes',xlabel='Optimizer step',ylabel='Episodes / 12',ylim=(0,14));ax[1].legend(fontsize=7,loc='upper center',ncol=3)
for a in ax:a.grid(axis='y',alpha=.2);a.set_axisbelow(True);a.set_xticks([0,16,32,48,64])
fig.tight_layout();y=chart(fig,y)
y=text('<b>Baseline already struggled to finish.</b> At step 0, 7/8 held-out tasks timed out. Over all nine validation rounds, 91/108 WideSeek attempts timed out, seven ended with other failures and only ten finished.',M,y)
y=text('<b>Training had the same problem.</b> Of 512 saved training episodes, 422 timed out (82.4%), 43 ended in errors and 47 finished (9.2%). This limits what task reward can tell us about answer quality.',M,y)
y=text('<b>OOD success fell from 9/20 to 3/20.</b> Seven baseline successes were lost and one new success appeared. This is an observed pass@1 change on a small fixed panel, not a statistical estimate of general capability.',M,y)
y=text('<b>Zero reward did not mean zero training signal.</b> The saved run config used k1 distillation with <font face="Courier">use_task_rewards=false</font>. Task rewards were monitoring signals; distillation still trained on the sampled trajectories.',M,y)
y=text('Decision: do not launch another long run unchanged. Separate completion-budget failures, generation latency and weight-loading correctness before attributing the curve to worse learned answers.',M,y,size=10.5)
c.showPage()
page(2,'Where WideSeek validation failed','12 attempts per checkpoint: four train probes and eight held-out tasks. Timeouts count as zero reward.')
y=H-104
vals=[]
for s in steps:
 rs=wr(s);calls=[v for r in rs for v in r.get('model_call_outcomes',[])];long=sum(v['status']!='success' and v['duration_seconds']>120 for v in calls)
 vals.append([s,sum(r['status']=='finished' for r in rs),sum(r['status']=='timeout' for r in rs),sum(r['status']=='error' for r in rs),f'{score(wr(s,"wideseek/train_probe")):.3f}',f'{score(wr(s,"wideseek/heldout")):.3f}',long])
y=table(['Step','Finished','Timeout','Other','Probe score','Held-out score','Calls >120s, cancelled'],vals,y,[39,58,57,47,78,81,CW-360])
y=text('<b>Late model calls consume the deadline.</b> At step 64, 11 coordinator calls ended in cancellation; ten had lasted over 120 seconds. At step 0 there were no logged coordinator-call cancellations. Successful coordinator calls at step 64 still had a median duration of 1.85 seconds.',M,y)
y=text('<b>Concrete example: task width-00000.</b> Its first call completed in 1.78 seconds and created an agent. The second call ran for 177.33 seconds before cancellation. No researcher call was recorded. All four final train-probe tasks showed this general pattern: roughly 179 seconds inside coordinator calls, no researcher calls.',M,y)
y=text('<b>What remains unknown.</b> The saved call logs contain completed outputs, not partial output from cancelled generation. We cannot distinguish queueing or serving stalls from very long unfinished completions. The logs do not establish that final-validation shutdown caused these cancellations.',M,y)
y=text('<b>Other failures have identifiable causes.</b> Across training: 29 graceful-shutdown violations, eight generations exhausted their budget without EOS, and six observations exceeded the remaining response budget. Validation adds six shutdown violations and one generation-without-EOS failure.',M,y)
y=text('<b>Budget accounting matters.</b> The 8,192-token response sequence includes coordinator tokens <i>and tool observations</i>. Masking observations out of the loss does not remove them from the sequence budget. The 180-second outer deadline also encloses agent cleanup and native evaluation.',M,y)
c.showPage()
page(3,'Workplace regression and next checks','Fixed 20-task OOD panel | Baseline 9/20 -> step 64 3/20 | No new evaluation was run for this audit')
y=H-104
by={(r['steps'][0],r['task']):r for r in work if len(r['steps'])==1}
changed=[]
for tid in sorted({r['task'] for r in work}):
 a,b=by[(0,tid)],by[(64,tid)]
 if a['score']==b['score']:continue
 reason='Completed, native check failed' if b['status']=='finished' and b['score']==0 else 'Newly passed' if b['score']==1 else 'Shutdown protocol violation' if 'shutdown' in (b['error'] or '') else 'Observations filled sequence budget'
 changed.append([tid,f'{a["score"]:.0f} -> {b["score"]:.0f}',reason,b['student_tokens']])
y=table(['Task ID','Score','Step-64 outcome','Student tokens'],changed,y,[55,65,285,CW-405])
y=text('Four of the seven lost successes ended in shutdown or observation-budget failures; three finished but failed the native state check. For example, task 231 returned a list of tasks but failed the native check. A fluent final response is not sufficient for success.',M,y)
y=text('<b>Recommended diagnostic sequence</b>',M,y,size=12)
y=text('<b>1. Verify the evaluated weights.</b> Use a frozen token replay to compare the saved checkpoint against freshly loaded and synchronized rollout weights. The separate receiver benchmark produced a suspicious merged-reference result; weight-version labels alone cannot establish numerical identity.',M,y)
y=text('<b>2. Isolate the latency effect.</b> Evaluate base and step 64 on the same small task subset at low concurrency, with identical prompts, sampling and budgets. Capture queue wait, generation time and partial token counts for cancelled requests. Do not start training for this check.',M,y)
y=text('<b>3. Attribute budget failures.</b> Report generated tokens, observation tokens, task completion and native score separately. Only then test one targeted budget or completion-policy change. Keep the current speed configuration fixed.',M,y)
y=text('<b>Evidence and scope.</b> Run revision: 8e0b363dc35ab41adeac9366a0a516849ea31f6f. Source on Hertz-2: <font face="Courier">wideseek/runs/...-r6/train/</font>, including episodes/*/{result,trajectory}.json, nested model_calls.jsonl, workplace-episodes/*/{result.json,policy_calls.jsonl}, trainer.log and hydra/.hydra/config.yaml. Reduced evidence is saved locally as tmp/pdfs/wideseek-failure-audit.json. Unknown or missing native scores were treated as zero only when reproducing the plotted aggregate; failure statuses remain separate.',M,y,size=8.7)
c.save();print(out)
