from pathlib import Path
s=Path(__file__).with_name('build_wideseek_sft_report.py').read_text().split('latest_val=')[0]
s=s.replace('wideseek-sft-final.json','wideseek-32k-latest.json').replace('wideseek-sft-opd-report.pdf','wideseek-32k-opd-report.pdf')
exec(s)
def unavailable(title,col,row):
 f,a=base(title,'');a.text(.5,.5,'No completed training scalar record',ha='center',va='center',transform=a.transAxes,color=C['gray']);a.set_xticks([]);a.set_yticks([]);place(f,col,row)
page(1,'WideSeek 32K: baseline measured; run stopped early','SFT v1.2.0 | 32K response budget | 15-minute deadline | checkpoint 1 saved',[
 'Fatal stop: one policy-inference request exceeded its separate 300s timeout. No post-update validation was recorded.',
 'Step 0: train probes 0.628; held-out F1 0.617; Workplace OOD 19/20. These are starting-policy scores, not learning gains.',
 'Training episodes: 11 finished, 3 deadlines, 4 shutdown failures, 1 recursion limit, 1 inference timeout; no token-budget failure.'])
f,a=base('Fixed-panel validation','Mean reward / success rate','Validation step')
for tag,label,col in [('val-core/wideseek/train_probe/reward/mean@1','Train probes (4)',C['blue']),('val-core/wideseek/heldout/reward/mean@1','Held-out WideSeek (8)',C['green']),('val-core/workplace-assistant/validation20/reward/mean@1','Workplace OOD (20)',C['red'])]:
 x,y=line(a,tag,label,col,marker='o');a.annotate(f'{y[0]:.3f}',(0,y[0]),xytext=(8,8 if 'heldout' not in tag else -15),textcoords='offset points',color=col)
a.set(xlim=(-1,8),ylim=(0,1.05),xticks=[0,8]);a.legend(loc='lower right');place(f,0,0)
f,a=base('Change from step 0','Change in mean score','');a.text(.5,.5,'No post-update validation available',ha='center',va='center',transform=a.transAxes);a.set_xticks([]);place(f,1,0)
unavailable('Training rollout reward',0,1);unavailable('Rollout length and token pressure',1,1);pdf.showPage()
page(2,'How the policy is changing','Optimizer checkpoint 1 exists; the usual loss, gradient and clipping scalars were not recorded.',[
 'Missing metrics are left blank rather than inferred. No learning conclusion can be drawn from this snapshot.',
 f'Entropy proxy covers {sum(r["episodes"] for r in proxy["rows"])} training episodes; {proxy["excluded_mixed_version_episodes"]} mixed-version episodes excluded.',
 'Sampled-token surprisal uses student tokens only; it is not full-distribution entropy. Versions have small and changing task samples.'])
unavailable('Distillation objective',0,0);unavailable('Gradient magnitude',1,0);unavailable('Update size and clipping',0,1)
f,a=base('Entropy proxy: sampled-token surprisal','Mean -log p(token), nats','Rollout policy version');rows=proxy['rows'];a.plot([r['version'] for r in rows],[r['mean_nll'] for r in rows],marker='o',color=C['gold']);a.set_ylim(bottom=0);a.set_xticks([r['version'] for r in rows]);place(f,1,1);pdf.showPage()
page(3,'Where wall-clock time goes','Run stopped after 56.3 minutes, including startup and step-0 validation.',[
 'Checkpoint 1 was saved; later optimizer progress is unverified. Planned training length: 64 updates.',
 'Step-0 validation took 904.6 seconds. The fatal inference timeout is separate from the 900-second episode deadline.',
 'Idle ratio is a scheduler metric, not measured GPU utilization. No complete per-update timing breakdown was saved.'])
unavailable('Per-step timing',0,0)
f,a=base('Time spent idle','Fraction of interval','Rollout version');line(a,'fully_async/rollouter/idle_ratio','Rollouter idle',C['red'],marker='o');a.set_ylim(-.02,1.02);a.set_xticks([0,1]);a.legend();place(f,1,0)
f,a=base('Recorded training progress','Saved checkpoint number','');a.bar(['Saved','Planned'],[1,64],color=[C['blue'],C['gray']]);a.set_ylim(0,68);place(f,0,1)
unavailable('Typical time components',1,1);pdf.save();print(OUT)
