import json, io
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.utils import ImageReader

ROOT=Path(__file__).resolve().parents[3]
d=json.loads((ROOT/'tmp/pdfs/synth-opd-metrics.json').read_text()); s=d['series']
OUT=ROOT/'output/pdf/synth-opd-progress.pdf'
proxy=json.loads((ROOT/'tmp/pdfs/synth-opd-entropy-proxy.json').read_text())
C={'blue':'#2463A6','green':'#16856B','red':'#C24750','gray':'#697887','gold':'#BE842C'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':9,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#B5BEC8','grid.color':'#E3E8ED','axes.titleweight':'bold','savefig.facecolor':'white'})
def xy(tag):
 p={r['step']:r for r in s[tag]}; p=[p[k] for k in sorted(p)]
 return np.array([r['step'] for r in p]),np.array([r['value'] for r in p])
def base(title,ylabel,xlabel='Logged step'):
 fig,ax=plt.subplots(figsize=(5.6,2.85));ax.set(title=title,ylabel=ylabel,xlabel=xlabel);ax.grid(axis='y',alpha=.8);ax.set_axisbelow(True);return fig,ax
def line(ax,tag,label,color,**kw):
 x,y=xy(tag);ax.plot(x,y,label=label,color=color,**kw);return x,y
def img(fig):
 fig.tight_layout(pad=1.0);b=io.BytesIO();fig.savefig(b,format='png',dpi=210);plt.close(fig);b.seek(0);return ImageReader(b)
W,H=landscape(A4);pdf=canvas.Canvas(str(OUT),pagesize=(W,H));pdf.setTitle('Synth OPD - learning, transfer and training health');pdf.setAuthor('Artem Matrosov')
when=datetime.fromtimestamp(d['captured_at'],timezone.utc).strftime('%d %b %Y, %H:%M UTC')
def page(n,title,subtitle,notes):
 pdf.setFillColorRGB(.08,.14,.22);pdf.setFont('Helvetica-Bold',22);pdf.drawString(30,H-38,title)
 pdf.setFont('Helvetica',10);pdf.setFillColorRGB(.30,.36,.42);pdf.drawString(30,H-57,subtitle)
 pdf.setStrokeColorRGB(.83,.87,.90);pdf.line(30,66,W-30,66)
 pdf.setFont('Helvetica',8)
 for i,t in enumerate(notes):pdf.drawString(30,53-i*11,t)
 pdf.setFont('Helvetica',7);pdf.drawString(30,15,f'{d["run"]} | snapshot {when}');pdf.drawRightString(W-30,15,f'{n} / 3')
def place(fig,col,row):
 width=(W-70)/2; height=211
 pdf.drawImage(img(fig),30+col*(width+10),H-78-(row+1)*height-row*8,width=width,height=height,mask='auto')

latest_val=int(xy('val-core/synth/heldout/reward/mean@1')[0][-1])
latest_train=int(xy('training/global_step')[0][-1])
page(1,'Synth gains; transfer remains below baseline',f'Latest recorded validation: step {latest_val}; training scalars: log step {latest_train}. Completed run; final validation and checkpoint at step 64.',[
 'Synth uses native partial reward; Workplace20 uses binary task success. Panels: 4 train probes, 8 held-out synth tasks, 20 Workplace tasks.',
 'One sampled attempt per validation task. Small fixed panels, no uncertainty estimates or extrapolation; transfer decline is descriptive, not causal proof.',
 'Source: final TensorBoard event snapshot. Lines join observed points; the baseline is step 0 of this same run.'])
f,a=base('Fixed-panel validation','Mean reward / success rate','Validation step')
for tag,label,col in [('val-core/synth/train_probe/reward/mean@1','Synth train probes (4)',C['blue']),('val-core/synth/heldout/reward/mean@1','Synth held-out (8)',C['green']),('val-core/workplace-assistant/validation20/reward/mean@1','Workplace OOD (20)',C['red'])]:
 x,y=line(a,tag,label,col,marker='o',lw=2);a.annotate(f'{y[-1]:.3f}',(x[-1],y[-1]),xytext=(5,0),textcoords='offset points',color=col,fontsize=9)
a.set_ylim(0,1);a.set_xlim(-1,latest_val+5);a.set_xticks(range(0,latest_val+1,8));a.legend(loc='upper left');place(f,0,0)
f,a=base(f'Change from step 0 to step {latest_val}','Change in mean score',xlabel='')
tags=['val-core/synth/train_probe/reward/mean@1','val-core/synth/heldout/reward/mean@1','val-core/workplace-assistant/validation20/reward/mean@1'];delta=[xy(t)[1][-1]-xy(t)[1][0] for t in tags]
a.bar(['Train probes','Held-out synth','Workplace OOD'],delta,color=[C['blue'],C['green'],C['red']],width=.55);a.axhline(0,color=C['gray'],lw=.8);a.set_ylim(-.45,.9)
for i,v in enumerate(delta):a.text(i,v+(.035 if v>=0 else -.07),f'{v:+.3f}',ha='center',weight='bold')
place(f,1,0)
f,a=base('Training rollout reward','Native batch mean reward')
x,y=line(a,'critic/rewards/mean','Raw batch (8 episodes)',C['blue'],alpha=.32,lw=1);smooth=np.convolve(y,np.ones(5)/5,mode='valid');a.plot(x[4:],smooth,color=C['blue'],lw=2,label='Trailing 5 batches');a.set_ylim(0,1);a.legend();place(f,0,1)
f,a=base('Rollout length and token pressure','Response tokens')
x,y=xy('response_length/mean');a.fill_between(x,xy('response_length/min')[1],xy('response_length/max')[1],color=C['green'],alpha=.13,label='Batch min-max');a.plot(x,y,color=C['green'],label='Batch mean');a.axhline(8192,color=C['red'],ls='--',lw=1,label='Configured response budget');a.legend(loc='upper left');place(f,1,1)
pdf.showPage()

page(2,'How the policy is changing','Loss is supporting evidence; task rewards on page 1 remain the outcome measure.',[
 'Actor total loss equals the logged distillation loss here; they are plotted once. Task reward is not added to the OPD objective.',
 f'Proxy uses {sum(r["episodes"] for r in proxy["rows"])} train rollouts; mixed-version episodes excluded: {proxy["excluded_mixed_version_episodes"]}. KL compares rollout/update policies, not teacher quality.',
 'Entropy proxy: token-weighted -log p of sampled student tokens; excludes observations. Sampling filters/temperature prevent treating it as full entropy.'])
f,a=base('Distillation objective','Reported loss')
line(a,'actor/distillation/loss','Signed loss = actor total',C['blue'],lw=1.6);line(a,'actor/distillation/abs_loss','Absolute-loss metric',C['gold'],lw=1.4);a.axhline(0,color=C['gray'],lw=.7);a.legend();place(f,0,0)
f,a=base('Gradient magnitude','Reported gradient norm')
line(a,'actor/grad_norm','Gradient norm',C['green'],lw=1.6);a.set_ylim(bottom=0);place(f,1,0)
f,a=base('Update size and clipping','KL estimate / clipped fraction')
line(a,'actor/distillation/ppo_kl','Sampled PPO KL',C['blue'],lw=1.4);line(a,'actor/distillation/pg_clipfrac','Distillation clipped fraction',C['red'],lw=1.4);a.axhline(0,color=C['gray'],lw=.7);a.legend();place(f,0,1)
f,a=base('Entropy proxy: sampled-token surprisal','Mean -log p(token), nats','Rollout policy version')
rows=proxy['rows'];a.plot([r['version'] for r in rows],[r['mean_nll'] for r in rows],color=C['gold'],lw=1.7,marker='.',ms=4)
a.set_ylim(bottom=0);a.text(.03,.92,'Full-distribution entropy is not logged.',transform=a.transAxes,fontsize=8,color=C['gray'])
place(f,1,1)
pdf.showPage()

page(3,'Where wall-clock time goes','Operational metrics explain throughput; they are not learning-quality measurements.',[
 'Observed cadence uses consecutive wall times. Logged step time excludes checkpoints; asynchronous components can overlap. Event step = optimizer counter + 1.',
 'Idle ratios are scheduler metrics, not measured hardware GPU utilisation. Progress shows recorded optimizer counts, with no projected completion time.',
 f'Observed maxima: response clipping {max(xy("response_length/clip_ratio")[1]):.1%}; aborted responses {max(xy("response/aborted_ratio")[1]):.1%}; dropped stale samples {max(xy("fully_async/count/dropped_stale_samples")[1]):.0f}.'])
f,a=base('Per-step timing','Seconds')
p=s['training/global_step'];x=np.array([r['step'] for r in p]);t=np.array([r['time'] for r in p]);a.plot(x[1:],np.diff(t),color=C['gray'],alpha=.65,label='Observed log-to-log interval')
line(a,'timing_s/update_actor','Actor update',C['blue'],lw=1.3);line(a,'timing_s/save_checkpoint','Checkpoint save/offload',C['gold'],lw=1.3);a.legend(fontsize=7);a.set_ylim(bottom=0);place(f,0,0)
f,a=base('Time spent idle','Fraction of interval',xlabel='Logged step / rollout version')
line(a,'fully_async/rollouter/idle_ratio','Rollouter idle',C['red'],lw=1.4);line(a,'fully_async/trainer/idle_ratio','Trainer idle',C['blue'],lw=1.4);a.set_ylim(-.02,1.02);a.legend();place(f,1,0)
f,a=base('Recorded training progress','Optimizer counter','Hours since run start')
start=d['run_metadata']['started_at'];a.plot((t-start)/3600,[r['value'] for r in p],color=C['blue'],lw=2);a.axhline(64,color=C['gray'],ls='--',label='Planned 64 updates');a.set_ylim(0,68);a.legend();place(f,0,1)
f,a=base('Typical time components','Median seconds per logged step',xlabel='')
keys=['timing_s/update_actor','timing_s/timing_s/param_sync','timing_s/save_checkpoint'];v=[float(np.median(xy(k)[1])) for k in keys];a.bar(['Actor update','Parameter sync','Checkpoint'],v,color=[C['blue'],C['green'],C['gold']],width=.55)
for i,z in enumerate(v):a.text(i,z+3,f'{z:.0f}s',ha='center',weight='bold')
a.set_ylim(0,max(v)*1.25);place(f,1,1)
pdf.save()
print(OUT)
