"""Observed quality under common active-time budgets and completed-work costs."""
import math
from pathlib import Path
import numpy as np
import learning_time_budget as study
import learning_quality_report as export
import learning_continuation_reference as matched

prior=export.prior
read,write,sha=study.read,study.write,study.sha
DEST=study.ROOT/'final_final_charts/same_time'
LABELS={'current':'Proposed','gate_select16':'P-KS','iscc':'ISCC-P2','afl':'AFL-OTA','salun':'F-SalUn'}
SOURCE_PATHS={(ds,m):study.source(ds,m) for ds in study.quality.infra.DATASETS for m in LABELS}


def source(case,ds,m):return SOURCE_PATHS[(ds,m)]


def trajectory(ds,m):
 arm=source('baseline',ds,m);warm,points,timeline=prior.trajectories(ds,m)
 p=prior.load(arm.parents[1]/'settings.json')['energy_model'];prior.physical.PARAMETERS=p
 works={w['transaction']:w for w in [prior.load(f) for f in (arm/'work_attempts').glob('*.json')]}
 tx=[prior.load(f) for f in sorted((arm/'transactions').glob('*.json'))]
 assert len(works)==len(tx)
 total=0.;cost={0:0.};last_active=0.;completed=0;progress=[dict(active_s=0.,FL_round=20,FU_cycles=0)]
 for t in tx:
  w=works[t['sequence']];active=prior.chart.active_time(timeline,t['end_s'])
  total+=float(prior.physical.compute_cost(w['compute']['device_macs'])[0].sum())
  total+=export.plots.comm(t['radio'],p['communication'])[0]+p['communication']['head_processing_w']*t['radio']['uplink_s']
  total+=10*float(prior.physical.propulsion_power(0))*(active-last_active);last_active=active;cost[t['sequence']]=total
  completed+=len(t['FU_request_ids'])
  progress.append(dict(active_s=active,FL_round=t['completed_FL_rounds'],FU_cycles=completed))
 assert completed==225
 values=[dict(active_s=0.,energy_j=0.,FL_round=20,FU_completed=0,evaluation=dict(retained=warm[-1][1]),stage='warm20')]
 for point in points:
  values.append(dict(active_s=prior.chart.active_time(timeline,point['mission_clock_s']),energy_j=cost[point['sequence']],
   FL_round=point['completed_FL_rounds'],FU_completed=point['requests']['procedure_completed'],evaluation=point['evaluation'],stage='fixed_workload'))
 continuation=study.OUT/(ds.lower()+'_seed0')/m
 curve=prior.load(continuation/'curve.json');done=prior.load(continuation/'finished.json');assert done['passed']
 assert math.isclose(curve[0]['energy_j'],total,rel_tol=1e-10)
 assert math.isclose(curve[0]['active_s'],last_active,abs_tol=1e-8)
 assert curve[0]['evaluation']==values[-1]['evaluation']
 plan=read(arm.parent/'plan.json');deleted=set(plan['forgotten_images']);audited=0
 all_jobs={r['job']:r for r in [prior.load(f) for f in (continuation/'jobs').glob('*.json')]};consumed=[]
 for i,point in enumerate(curve[1:],1):
  t=prior.load(continuation/'transactions'/f'{i:04d}.json');rm=t['radio'];previous=curve[i-1]
  assert point['FL_round']==200+i and point['extra_FL']==i
  assert math.isclose(t['active_s']-t['start_s'],t['timing']['duration_s'],abs_tol=1e-8)
  assert math.isclose(t['start_s'],previous['active_s'],abs_tol=1e-8)
  comp=float(prior.physical.compute_cost(t['device_macs'])[0].sum())+p['communication']['head_processing_w']*rm['uplink_s']
  comm,down=export.plots.comm(rm,p['communication']);hover=10*float(prior.physical.propulsion_power(0))*t['timing']['duration_s']
  assert math.isclose(comp,t['compute_j'],rel_tol=1e-10) and math.isclose(comm,t['communication_j'],rel_tol=1e-10)
  assert math.isclose(down,t['timing']['downlink_s'],abs_tol=1e-8)
  assert math.isclose(point['energy_j']-previous['energy_j'],comp+comm+hover,rel_tol=1e-9)
  assert rm['max_power_w']<=.32*(1+2e-6) and rm['conditional_rmse']<=rm['epsilon']*(1+3e-5)+1e-9
  for record in t['selection']:
   assert not set(record['selected'])&deleted and len(set(record['selected']))==16
  if m=='afl':
   for j in rm['selected_job_ids']:
    job=all_jobs[j];assert job['ready_s']<=t['start_s']+t['timing']['compute_upload_s']-rm['uplink_s']+1e-8
    assert not set(job['samples'])&deleted;consumed.append(j)
  values.append(dict(active_s=point['active_s'],energy_j=point['energy_j'],FL_round=point['FL_round'],FU_completed=45,evaluation=point['evaluation'],stage='retained_continuation'))
  progress.append(dict(active_s=point['active_s'],FL_round=point['FL_round'],FU_cycles=225));audited+=1
 if m=='afl':assert sorted(consumed)==sorted(all_jobs)
 assert audited==done['extra_FL'] and values[-1]['active_s']>=400 and values[-2]['active_s']<400
 return values,progress,dict(dataset=ds,method=m,verified_transactions=audited,energy_clock_and_workload=True,source=str(continuation.relative_to(study.ROOT)))


def main():
 study.register();export.setup(DEST,LABELS,source)
 paths={};progress={};checks=[];rows=[]
 for ds in study.quality.infra.DATASETS:
  for m in LABELS:
   paths[(ds,m)],progress[(ds,m)],check=trajectory(ds,m);checks.append(check)
   for budget in study.BUDGETS:
    candidates=[p for p in paths[(ds,m)] if p['active_s']<=budget];p=candidates[-1];e=p['evaluation']
    real=next(r for r in reversed(progress[(ds,m)]) if r['active_s']<=budget)
    rows.append(dict(dataset=ds,method=m,label=LABELS[m],budget_s=budget,quality_checkpoint_s=p['active_s'],
     quality_checkpoint_lag_s=budget-p['active_s'],evaluated_FL_round=p['FL_round'],completed_FL_round=real['FL_round'],
     completed_FU_cycles=real['FU_cycles'],evaluated_FU_completed=p['FU_completed'],
     retained_mR_percent=100*e['retained']['mR'],retained_loss=e['retained']['loss'],
     energy_to_evaluated_checkpoint_kJ=p['energy_j']/1000,stage=p['stage']))
 for number,(metric,label) in enumerate([('mR','Retained mR (%)'),('loss','Validation loss'),('FL_round','Completed FL rounds'),('FU_cycles','Completed FU cycles')],1):
  fig,axes=prior.chart.style.canvas(label);panel_values=[];name=f'{number:02d}_time_{metric.lower()}_1x2'
  for ax,ds in zip(axes,study.quality.infra.DATASETS):
   ys=[]
   for i,(m,title) in enumerate(LABELS.items()):
    is_quality=metric in ['mR','loss'];recorded=paths[(ds,m)] if is_quality else progress[(ds,m)]
    chosen=[p for p in recorded if p['active_s']<=400];x=[p['active_s'] for p in chosen]
    y=[(100 if metric=='mR' else 1)*p['evaluation']['retained'][metric] if is_quality else p[metric] for p in chosen]
    actual_x,actual_y=list(x),list(y)
    if x[-1]<400:x.append(400.);y.append(y[-1])
    line=prior.chart.style.curve(ax,x,y,i,title);line.set_drawstyle('steps-post');ys+=y
    prior.SERIES.append(dict(figure=name,dataset=ds,method=m,metric=metric,x=actual_x,y=actual_y,
     display_x=x,display_y=y,drawstyle='steps-post',last_observed_completed_checkpoint=True,no_smoothing=True))
   prior.chart.regular_axis(ax,'x',[0,100,200,300,400],0);ax.set_xlabel('Time (s)');panel_values.append(ys)
  prior.chart.finish_axes(axes,panel_values,decimals=1 if metric=='mR' else 2 if metric=='loss' else 0)
  prior.save(fig,axes,name,label+' under a shared time budget')
 targets={'CapERA':[17.,18.,20.,22.],'AirScape':[13.8,14.,14.5,15.]};hits=[]
 for ds in targets:
  for m in LABELS:
   for target in targets[ds]:
    first=next((p for p in paths[(ds,m)] if p['active_s']<=400 and p['FU_completed']==45 and 100*p['evaluation']['retained']['mR']>=target),None)
    hits.append(dict(dataset=ds,method=m,label=LABELS[m],target_mR_percent=target,reached=first is not None,
     time_s=None if first is None else first['active_s'],energy_kJ=None if first is None else first['energy_j']/1000,
     actual_mR_percent=None if first is None else 100*first['evaluation']['retained']['mR']))
 reference=[]
 for ds in study.quality.infra.DATASETS:
  p=matched.OUT/(ds.lower()+'_seed0')/'finished.json';r=prior.load(p);assert r['passed'];reference+=r['rows']
 write(DEST/'data/audit.json',dict(passed=True,rows=checks));write(DEST/'data/time_to_target.json',hits);prior.write_csv(DEST/'data/time_to_target.csv',hits)
 write(DEST/'data/round_matched_final_reference.json',reference);prior.write_csv(DEST/'data/round_matched_final_reference.csv',reference)
 write(DEST/'data/continuation_protocol.json',read(study.OUT/'protocol.json'))
 text=['# Same-time FL/FU continuation','',
  'Each method first completes the original200 FL rounds and45 requests (225 FU cycles), then spends remaining active service time on retained-only FL. Every method uses the same restarted constant0.0035 LR; the original cosine schedule ended at zero. Final-site channels remain fixed. P-KS is the predeclared KL-gate + selective-FL16 candidate, not the current Proposed. It is not promoted to the main method because its fixed-workload CapERA mean KL and worst-request KL exceed0.01.','',
  'No new deletion requests arrive during continuation. This tests use of saved time, not an online request-capacity benchmark. The FU-progress chart counts executed prescribed steps, not certified successful deletions.','',
  'Quality curves use the last observed completed checkpoint, without interpolation. Before200 rounds the original evaluations are sparse; after200 every completed model is evaluated. CSV includes actual completed FL/FU counts, quality-checkpoint time and lag. At400 seconds all methods have completed all deletions and have dense continuation observations.','',
  'Energy includes all active communication, compute and hover. At the same hover duration energy is similar; energy benefits must be assessed by time/energy to a quality target. Time-to-target tables require all45 FU procedures to have finished and retain unreached targets as missing. These are descriptive developmental thresholds, not tuning or stopping inputs.','',
  'The old reference200 KL is not a fair standalone forgetting score after hundreds of additional FL updates. Thus no such KL curve is used to claim forgetting superiority here. A separate final table extends the common retained-only reference to each actual final FL count using exact sample-weighted updates and the identical continuation LR. This is a round-matched common reference, not a method-specific oracle for selective/equal-client FL, and provides no deletion certificate. It is reporting-only. Final checkpoints are the first completed update beyond400 seconds; their actual timestamps are explicit. The common400-second utility table uses the last checkpoint at or before400.','',
  'Single development seed and frozen-feature adapter simulation; no official full-dataset score or statistical superiority claim.','',
  '| Dataset | Method | mR at400s (%) | Loss | Completed FL rounds | FU cycles |','|---|---|---:|---:|---:|---:|']
 for r in rows:
  if r['budget_s']==400:text.append(f"| {r['dataset']} | {r['label']} | {r['retained_mR_percent']:.3f} | {r['retained_loss']:.5f} | {r['completed_FL_round']} | {r['completed_FU_cycles']} |")
 text+=['','## Final round-matched common-reference diagnostics','','| Dataset | Method | FL rounds | Actual time (s) | Mean KL | Worst request KL |','|---|---|---:|---:|---:|---:|']
 for r in reference:text.append(f"| {r['dataset']} | {LABELS[r['method']]} | {r['FL_round']} | {r['active_s']:.3f} | {r['matched_common_reference_KL']:.6g} | {r['worst_request_KL']:.6g} |")
 text+=['','[Time and energy to each descriptive target](data/time_to_target.csv). Rebuild: `python tools/uav_convergence_v2/learning_time_report.py`.','']
 (DEST/'README.md').write_text('\n'.join(text))
 for file in [Path(__file__),Path(study.__file__),Path(matched.__file__)]:prior.SOURCES[str(file.relative_to(study.ROOT))]=sha(file)
 export.finish(DEST,rows,'Actual additional retained FL under common active-time budgets.',dict(continuation_arms=10))
 print(dict(figures=len(prior.FIGURES),budget_rows=len(rows),continuation_arms=10),flush=True)


if __name__=='__main__':main()
