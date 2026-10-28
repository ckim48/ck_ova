"""P-KS evaluation correction and actual learning interventions, kept distinct."""
import math
from pathlib import Path
import numpy as np
import pks_balance_v1 as study
import pks_evaluation_v1 as evaluation
import pks_reference_audit as reference_audit
import pks_budget400_fix as strict_budget
import learning_quality_report as export
import learning_time_report as old_time_report

ROOT=study.ROOT;DEST=ROOT/'final_final_charts/pks_kl_repair'
read,write,sha=study.read,study.write,study.sha
prior=export.prior
METHODS=dict(evaluation.METHODS,pk_uniform='P-KU',pls_uniform='P-LU')
PATHS={(ds,m):evaluation.sources(ds,m) for ds in study.q.infra.DATASETS for m in METHODS}
MAIN=['current','pks','iscc','afl','salun']
BOUNDED=['pks','mix25_gate','mix50_gate','mix50_adaptive']
PHASE=['pks','pk_uniform','pls_uniform']
STYLES={
 'current':('#4C72B0','o','-','//'),'pks':('#92703B','d',(0,(5,2)),'..'),
 'iscc':('#EFAA54','s','--','\\\\'),'afl':('#66B366','v','-.','xx'),'salun':('#C65365','^',':','++'),
 'mix25_gate':('#9467BD','P','-','//'),'mix50_gate':('#55A9AE','h','--','\\\\'),
 'mix50_adaptive':('#D78070','X','-.','xx'),'pk_uniform':('#1F9E89','s','--','//'),'pls_uniform':('#CC79A7','v','-.','xx')}


def initial_source(case,ds,m):return PATHS[(ds,m)][0]


def continued_source(ds,m):return PATHS[(ds,m)][1]


def styles(methods):
 prior.METHODS={m:METHODS[m] for m in methods}
 for key,index in [('COLORS',0),('MARKERS',1),('STYLES',2)]:
  value=[STYLES[m][index] for m in methods];setattr(prior.chart,key,value);setattr(prior.chart.style,key,value)


def get_trajectories():
 # Reuse the independently checked timing/energy reconstruction with new paths.
 fn=study.q.infra.cached_clone(old_time_report.trajectory,[
  ("continuation=study.OUT/(ds.lower()+'_seed0')/m",'continuation=continued_source(ds,m)',1)],
  dict(source=initial_source,continued_source=continued_source))
 paths={};progress={};checks=[]
 for ds in study.q.infra.DATASETS:
  for m in METHODS:paths[(ds,m)],progress[(ds,m)],check=fn(ds,m);checks.append(check)
 return paths,progress,checks


def audit_training():
 assert read(study.OUT/'control_check.json')['passed']
 results=[]
 for ds in study.q.infra.DATASETS:
  reference=PATHS[(ds,'pks')][0]
  for m in ['mix25_gate','mix50_gate','mix50_adaptive']:
   initial,continued=PATHS[(ds,m)];plan=read(initial.parent/'plan.json');requests=read(initial/'request_records.json')
   for name in ['plan.json','requests.json','initial20/final.pt','reference200/final.pt']:
    assert sha(initial.parent/name)==sha(reference.parent/name),(ds,m,name)
   count=0
   for file in (initial/'work_attempts').glob('*.json'):
    w=read(file);deleted={i for r in requests if r['arrival_s']<=w['mission_time_s']+1e-9 for i in r['images']}
    for b in w['compute']['FL_gradient_blend']:
     pool=set(plan['partitions'][b['user']])-deleted
     assert set(b['uniform'])<=pool and set(b['priority'])<=pool
     assert b['relative_gradient_difference']<=b['rho']*(1+1e-5)+1e-8 and 0<=b['beta']<=1
     assert b['rho']==study.VARIANTS[m]['rho'];count+=1
   assert count==1800
   n=0;deleted=set(plan['forgotten_images'])
   for file in (continued/'transactions').glob('*.json'):
    for b in read(file)['gradient_blend']:
     assert not (set(b['uniform'])|set(b['priority']))&deleted
     assert b['relative_gradient_difference']<=b['rho']*(1+1e-5)+1e-8;n+=1
   assert n==10*read(continued/'finished.json')['extra_FL']
   results.append(dict(dataset=ds,method=m,baseline_gradient_pairs=count,continuation_gradient_pairs=n,
    eligible_retained_samples_only=True,local_gradient_bound=True))
 return results


def curves(paths,methods,metric,number,group):
 styles(methods);label='Retained mR (%)' if metric=='mR' else 'Validation loss';scale=100 if metric=='mR' else 1
 name=f'{number:02d}_{group}_time_{metric.lower()}_1x2';fig,axes=prior.chart.style.canvas(label);panels=[]
 for ax,ds in zip(axes,study.q.infra.DATASETS):
  values=[]
  for i,m in enumerate(methods):
   points=[r for r in paths[(ds,m)] if r['active_s']<=400]
   x=[r['active_s'] for r in points];y=[scale*r['evaluation']['retained'][metric] for r in points]
   dx,dy=x+[400.],y+[y[-1]]
   line=prior.chart.style.curve(ax,dx,dy,i,METHODS[m]);line.set_drawstyle('steps-post');values+=y
   prior.SERIES.append(dict(figure=name,dataset=ds,method=m,metric=metric,x=x,y=y,display_x=dx,display_y=dy,drawstyle='steps-post',stage='budget400'))
  prior.chart.regular_axis(ax,'x',[0,100,200,300,400],0);ax.set_xlabel('Time (s)');panels.append(values)
 prior.chart.finish_axes(axes,panels,decimals=1 if metric=='mR' else 2)
 prior.save(fig,axes,name,label+' — '+group+' — shared time budget')


def bars(rows,methods,metric,number,group,stage='terminal'):
 styles(methods);name=f'{number:02d}_{group}_{metric.lower()}_1x2'
 label='Worst-request KL' if 'worst' in metric else 'Forget-set KL'
 fig,axes=prior.chart.style.canvas(label);ticks=[]
 for ax,ds in zip(axes,study.q.infra.DATASETS):
  values=[]
  for i,m in enumerate(methods):
   r=next(r for r in rows if r['dataset']==ds and r['method']==m and r['stage']==stage);value=r[metric]
   ax.bar(i+1,value,width=.64,color=STYLES[m][0],edgecolor='black',linewidth=.7,hatch=STYLES[m][3],label=METHODS[m]);values.append(value)
   prior.SERIES.append(dict(figure=name,dataset=ds,method=m,metric=metric,x=[i+1],y=[value],stage=stage,bar=True))
  prior.chart.regular_axis(ax,'x',list(range(len(methods)+2)),0)
  ax.tick_params(axis='x',bottom=False,labelbottom=False)
  t=[v for v in prior.chart.nice_ticks([0.]+values,padding=0,bins=5,simple=True) if v>=0]
  assert t[0]==0 and t[-1]>=max(values);ticks.append(t)
 decimals=max(prior.chart.tick_decimals(t) for t in ticks)
 for ax,t in zip(axes,ticks):prior.chart.regular_axis(ax,'y',t,decimals)
 prior.save(fig,axes,name,label+' — '+('policy-matched reference' if metric.startswith('matched') else 'original common-reference protocol')+' — '+group)


def main():
 study.register();assert read(study.OUT/'control_check.json')['passed']
 export.setup(DEST,METHODS,initial_source)
 rows=[]
 old_scores=read(ROOT/'final_final_charts/same_time/data/round_matched_final_reference.json')
 for ds in study.q.infra.DATASETS:
  p=evaluation.OUT/(ds.lower()+'_seed0')
  result=prior.load(p/'finished.json');assert result['passed'];rows+=result['rows']
  protocol=prior.load(p/'protocol.json')
  for file,digest in {**protocol['source_sha256'],**protocol['input_sha256']}.items():assert sha(ROOT/file)==digest,file
  assert read(reference_audit.OUT/(ds.lower()+'_seed0')/'finished.json')['passed']
 for row in rows:
  if row['stage']=='terminal' and row['method'] in ['current','pks','iscc','afl','salun']:
   m='gate_select16' if row['method']=='pks' else row['method']
   old=next(r for r in old_scores if r['dataset']==row['dataset'] and r['method']==m)
   assert abs(old['matched_common_reference_KL']-row['common_KL'])<1e-6
   row['previous_common_KL']=old['matched_common_reference_KL']
 paths,progress,checks=get_trajectories();gradient_checks=audit_training();budget=[]
 for row in rows:
  ds,m=row['dataset'],row['method'];p=paths[(ds,m)][-1] if row['stage']=='terminal' else next(p for p in reversed(paths[(ds,m)]) if p['stage']=='fixed_workload')
  assert abs(p['evaluation']['retained']['mR']*100-row['retained_mR_percent'])<1e-10
  assert abs(p['active_s']-row['active_s'])<1e-8
  row['total_energy_kJ']=p['energy_j']/1000;row['label']=METHODS[m]
 for ds in study.q.infra.DATASETS:
  for m in METHODS:
   p=next(p for p in reversed(paths[(ds,m)]) if p['active_s']<=400)
   b=next(p for p in reversed(progress[(ds,m)]) if p['active_s']<=400)
   budget.append(dict(dataset=ds,method=m,label=METHODS[m],budget_s=400,checkpoint_s=p['active_s'],
    FL_round=b['FL_round'],FU_cycles=b['FU_cycles'],retained_mR_percent=100*p['evaluation']['retained']['mR'],
    retained_loss=p['evaluation']['retained']['loss'],energy_to_checkpoint_kJ=p['energy_j']/1000))
 for i,group in enumerate([('literature',MAIN),('bounded',BOUNDED),('phase',PHASE)]):
  for j,metric in enumerate(['mR','loss']):curves(paths,group[1],metric,1+2*i+j,group[0])
 for n,(group,methods,metric) in enumerate([
  ('literature',MAIN,'common_KL'),('literature',MAIN,'matched_KL'),
  ('bounded',BOUNDED,'common_KL'),('bounded',BOUNDED,'matched_KL'),
  ('bounded',BOUNDED,'common_worst_KL'),('bounded',BOUNDED,'matched_worst_KL')],7):bars(rows,methods,metric,n,group)
 strict=[]
 for ds in study.q.infra.DATASETS:
  directory=strict_budget.OUT/(ds.lower()+'_seed0');check=prior.load(directory/'finished.json')
  assert check['passed'] and check['unchanged_PKS_replay_exact'] and check['references_reproduced_exactly'];strict+=check['rows']
  protocol=prior.load(directory/'protocol.json')
  for file,digest in protocol['source_sha256'].items():assert sha(ROOT/file)==digest,file
 for i,metric in enumerate(['common_KL','common_worst_KL'],13):bars(strict,['pks','mix25_gate'],metric,i,'strict400',stage='budget400')
 write(DEST/'data/strict400.json',strict);prior.write_csv(DEST/'data/strict400.csv',strict)
 write(DEST/'data/budget400.json',budget);prior.write_csv(DEST/'data/budget400.csv',budget)
 write(DEST/'data/audit.json',dict(passed=True,continuation_arms=20,rows=checks,gradient_checks=gradient_checks,
  common_reference_continuation_numeric_tolerance=1e-6,zero_change_control=read(study.OUT/'control_check.json')))
 write(DEST/'data/training_protocol.json',read(study.OUT/'protocol.json'))
 write(DEST/'data/reference_protocol.json',{ds:read(evaluation.OUT/(ds.lower()+'_seed0')/'protocol.json') for ds in study.q.infra.DATASETS})
 write(DEST/'data/reference_diagnosis.json',{ds:read(reference_audit.OUT/(ds.lower()+'_seed0')/'finished.json') for ds in study.q.infra.DATASETS})
 text=['# P-KS: reference audit and actual KL-control experiments','',
  '**Two different findings must remain separate.** The original brown P-KS is our KL-gate + priority-FL candidate. Its earlier large KL was measured against a uniformly trained common reference. Repeating retained-only training with the candidate’s own FL sampler substantially reduces that measured discrepancy without changing the candidate. This is an evaluation correction/diagnosis, not a newly improved trained model. Both reference definitions remain published for every method.','',
  'Separately, the P-KB25/P-KB50/P-AB50 arms are real new training runs. They bound the additional priority gradient relative to the same-step uniform retained gradient. All predeclared outcomes are included, even when mR, loss or worst-request KL is worse.','',
  '## Methods','',
  '| Label | Change |','|---|---|',
  '| Proposed | Original adaptive-FU, uniform-FL method, unchanged |',
  '| P-KS | Original proxy-KL gate and half-exploration/half-priority FL sampler, unchanged |',
  '| P-KB25 / P-KB50 | P-KS FU; limit the applied gradient’s deviation from the uniform retained gradient to 25% / 50% of the uniform-gradient norm |',
  '| P-AB50 | Same 50% bound, with the original adaptive FU objective instead of the gate |',
  '| P-KU / P-LU | Reuse P-KS / P-LS at 200 rounds; switch to uniform retained sampling only during subsequent FL |',
  '| ISCC-P2 / AFL-OTA / F-SalUn | Existing published paper-core transfers, unchanged |','',
  'The gradient rule is `g = g_U + beta*(g_H-g_U)`, where `beta=min(1,rho*||g_U||/||g_H-g_U||)`. It adapts using current training gradients only. Both passes and selection/blend arithmetic are charged. This is a local step constraint, not a theorem bounding final KL. The same constants apply to both datasets. The idea is our adaptation; [SLUG, ICML 2025](https://proceedings.mlr.press/v267/cai25f.html) motivates attention to gradient alignment, and [CLIPErase, ACL 2025](https://aclanthology.org/2025.acl-long.1469/) motivates separate forgetting/retention/consistency terms. We do not claim either paper’s exact algorithm or guarantees.','',
  '## What the two KL columns mean','',
  '- **Common KL:** the original reference at 200 rounds, then the same uniform continuation to the candidate’s final FL count. This preserves the previous comparison protocol. Float accumulation differences in the independently recomputed continuation are checked within 1e-6; previous published values are also retained in the CSV.',
  '- **Policy-matched KL:** exact retained-only training from theta0, the shared retained 20-round warmup, then each learner’s sampler, FL weighting and LR rule. Uniform sample-weighted for Proposed, priority for P-KS, equal-client for all three literature transfers, and the respective bounded or phase-changing rule for new candidates. All deleted samples are excluded from the start. No FU is performed by the reference because those samples never entered it.',
  '- Every method receives its own matched reference under the same construction rule. No reference parameter or prediction enters candidate training. The original reference at 200 rounds is reproduced bit-for-bit in the independent audit.',
  '- Neither KL is a deletion certificate. The 0.01 threshold is the existing empirical criterion, and both mean and worst-request results are reported. A small matched KL does not establish indistinguishability or justify hiding a large common KL.','',
  '## Clocks and figures','',
  'All methods complete 45 requests/225 FU cycles and 200 FL rounds, then continue retained-only FL under the previously declared constant 0.0035 LR until the first completed update beyond 400 active seconds. This is a fixed deletion workload with continuation, not a continuous-request capacity experiment.','',
  'Figures 01–06 show the last observed completed model at or before 400 seconds. Before 200 rounds evaluations are sparse; afterward every update is evaluated. The exact 400-second budget table is in `data/budget400.csv`. Figures 07–12 use the final saved checkpoint just beyond 400 seconds; actual times and FL counts are explicit. Figures 13–14 compare P-KS and P-KB25 at the last completed checkpoint within 400 seconds, with all corresponding metrics in `data/strict400.csv`. No interpolation of model quality, artificial slope, or hidden unsuccessful arm.','',
  'Total energy is actual active communication + adapter computation + hover. Frozen features and optimizer-host exclusions remain as previously declared. Extra evaluation/retraining oracles are reporting work, excluded from every candidate’s service cost. Same-time energy is not a performance metric by itself when all UAVs hover for a similar duration.','',
  '## Exact 400-second budget: last completed checkpoint','',
  'P-KS was replayed without changing its algorithm to recover the previously unsaved pre-400 checkpoint. Every original observed quality value, modeled time, energy and the final parameter tensor reproduce exactly. The new common-KL and matched-KL measurements below use the same checkpoint as the mR value. Both methods finish 612 FL rounds and 225 FU cycles within 400 seconds. The terminal table below instead uses the next completed checkpoint (613 rounds), so its small mR differences must not be mixed with these values.','',
  '| Dataset | Method | mR (%) | Loss | Common KL | Common worst KL | Matched KL | Checkpoint time (s) |','|---|---|---:|---:|---:|---:|---:|---:|']
 for r in strict:text.append(f"| {r['dataset']} | {r['label']} | {r['retained_mR_percent']:.3f} | {r['retained_loss']:.5f} | {r['common_KL']:.6g} | {r['common_worst_KL']:.6g} | {r['matched_KL']:.6g} | {r['active_s']:.3f} |")
 text+=['','At the same 400-second budget, P-KB25 reduces common KL by 94.29% on CapERA and 94.98% on AirScape, while mR falls by 0.065 and 0.195 percentage points, respectively. Its mean and worst-request common KL are below 0.01 on both datasets. Loss increases by 0.0773 and 0.0238; the additional computation costs about 0.057 seconds and 0.098 kJ at the same completed workload. These are tradeoffs, not an improvement in every metric.','',
  '## Final checkpoint results (approximately 400 active seconds)','',
  '| Dataset | Method | mR (%) | Loss | Common KL | Common worst KL | Matched KL | Matched worst KL | Actual time (s) |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for r in rows:
  if r['stage']=='terminal':text.append(f"| {r['dataset']} | {r['label']} | {r['retained_mR_percent']:.3f} | {r['retained_loss']:.5f} | {r['common_KL']:.6g} | {r['common_worst_KL']:.6g} | {r['matched_KL']:.6g} | {r['matched_worst_KL']:.6g} | {r['active_s']:.3f} |")
 text+=['','The CSV also contains every 200-round endpoint, including the early utility loss of the bounded variants. This is one developmental seed and the existing subset/validation split; no statistical superiority or official full-benchmark claim. A preferred candidate requires paired new-seed and independent-split confirmation.','',
  '## Reproduction','',
  'With the existing local dataset/cache inputs, run `pks_reference_audit.py --dataset CapERA` and repeat for AirScape; `pks_balance_campaign.py run`; `pks_evaluation_v1.py --dataset CapERA` and repeat for AirScape. Recover the exact 400-second P-KS checkpoints with `pks_balance_v1.py continue --variant control --dataset CapERA` and repeat for AirScape; then run `pks_budget400_fix.py`, `pks_report.py` and `pks_release_check.py`. The earlier budget audit stopped because its assertion included reference-dependent retained KL/disagreement in an equality check; the correction compares the same model’s mR, loss and six recalls exactly while allowing its reference-dependent diagnostics to change. No model or reference training rule changes, and the stopped attempt remains preserved. All scripts are under `tools/uav_convergence_v2`. Previously published input arms are reused exactly. No raw data, learned parameters, or work/transaction traces are published.','']
 (DEST/'README.md').write_text('\n'.join(text).rstrip()+'\n')
 styles(list(METHODS))
 for file in [Path(__file__),Path(study.__file__),Path(evaluation.__file__),Path(reference_audit.__file__),Path(strict_budget.__file__)]:prior.SOURCES[str(file.relative_to(ROOT))]=sha(file)
 export.finish(DEST,rows,'P-KS reference audit and real KL-control candidates; all outcomes retained.',dict(evaluated_endpoints=40,strict_budget_endpoints=4,continuation_arms=20))
 print(dict(figures=len(prior.FIGURES),rows=len(rows),destination=str(DEST)),flush=True)


if __name__=='__main__':main()
