from common import *
import numpy as np

def rows_for(method):
 dest=OUT/method;p=dest/'prefix_rounds.json'
 rows=json.loads(p.read_text()) if p.exists() else []
 return rows+[json.loads(p.read_text()) for p in sorted((dest/'rounds').glob('*.json'))]

def confirmed_hit(rows,predicate,count=3):
 run=[]
 for row in rows:
  if row['round']<25 or not predicate(row):run=[];continue
  run.append(row)
  if len(run)>=count:
   return dict(entry_round=run[0]['round'],confirmation_round=row['round'],
    energy_bound_j=row['energy_bound_j'],energy_expected_j=row['energy_expected_j'],uplink_s=row['uplink_s'])
 return None

def plateau(rows):
 eligible=[r for r in rows if r['round']>=25 and r['round']%5==0]
 for i in range(6,len(eligible)):
  group=eligible[i-6:i+1]
  if group[-1]['round']<60 or group[-1]['round']-group[0]['round']!=30:continue
  mr=np.array([x['retained_mR'] for x in group]);loss=np.array([x['retained_loss'] for x in group])
  if np.ptp(mr)<=.005 and np.ptp(loss)/max(abs(loss.mean()),1e-12)<=.01:
   return dict(first_observed_round=group[-1]['round'],window_start=group[0]['round'],mR_range=float(np.ptp(mr)),relative_loss_range=float(np.ptp(loss)/abs(loss.mean())),
    energy_bound_j=group[-1]['energy_bound_j'],uplink_s=group[-1]['uplink_s'],interpretation='Utility plateau; target-quality success assessed separately.')
 return None

def build_report():
 plan=json.loads((OUT/'plan.json').read_text());ref=rows_for('retrain')
 finalref=next((r for r in reversed(ref) if r['round']==200 and 'evaluation' in r),None)
 threshold=finalref['evaluation']['retained']['mR']*.95 if finalref else None
 summaries=[];curves=[]
 for method in plan['methods']:
  rows=rows_for(method);ev=[];dest=OUT/method
  state=json.loads((dest/'manifest.json').read_text()) if (dest/'manifest.json').exists() else dict(status='running' if rows else 'queued')
  for r in rows:
   if 'evaluation' not in r:continue
   q=r['evaluation'];ev.append(dict(method=method,round=r['round'],retained_mR=q['retained']['mR'],retained_loss=q['retained']['loss'],
    forget_KL=q['forget'].get('KL_retrain_to_candidate'),energy_bound_j=r['cumulative_tx_energy_bound_j'],energy_expected_j=r['cumulative_tx_energy_expected_j'],
    uplink_s=r['cumulative_uplink_s'],request_algorithm_completed=r['request_algorithm_completed']))
  curves+=ev
  joint=confirmed_hit(ev,lambda r: threshold is not None and r['request_algorithm_completed'] and r['retained_mR']>=threshold and r['forget_KL'] is not None and r['forget_KL']<=.01)
  utility=[]
  for name,values in [('mR',plan['utility_target_grid_mR']),('loss',plan['loss_target_grid'])]:
   for v in values:
    hit=confirmed_hit(ev,lambda r:r['retained_mR']>=v if name=='mR' else r['retained_loss']<=v)
    utility.append(dict(target=name,value=v,confirmed=hit,scope='Utility-only post-request target; no forgetting certification.'))
  m=dict(method=method,label=plan['method_labels'][method],role='main' if method in plan['main_methods'] else 'ablation',
   status=state['status'],completed_rounds=rows[-1]['round'] if rows else 0,last_evaluation=ev[-1] if ev else None,
   joint_target_confirmed=joint,utility_plateau=plateau(ev),utility_targets=utility)
  for name in ['mia','final_test']:
   p=dest/(name+'.json');m[name]=json.loads(p.read_text()) if p.exists() else None
  summaries.append(m)
 write(OUT/'summary.json',dict(reference_round=200,reference_ready=finalref is not None,joint_retained_mR_threshold=threshold,joint_forget_KL_threshold=.01,
  confirmation='Three consecutive evaluation points after round25; cost at confirmation. No target hit within200 is censored, not proof of divergence.',methods=summaries))
 write(OUT/'convergence.json',curves)
 if curves:
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  for filename,methods in [('main_comparison',plan['main_methods']),('crossmodal_ablation',['proposed','hybrid_no_crossmodal'])]:
   fig,axes=plt.subplots(2,3,figsize=(14,8))
   for method in methods:
    a=[r for r in curves if r['method']==method]
    if not a:continue
    for ax,x,y in [(axes[0,0],'round','retained_mR'),(axes[0,1],'energy_bound_j','retained_mR'),(axes[0,2],'uplink_s','retained_mR'),
      (axes[1,0],'round','retained_loss'),(axes[1,1],'round','forget_KL'),(axes[1,2],'round','energy_bound_j')]:
     b=[r for r in a if r[y] is not None]
     ax.plot([r[x] for r in b],[r[y] for r in b],'-o',markersize=2,label=plan['method_labels'][method]);ax.set_xlabel(x);ax.set_ylabel(y);ax.grid(alpha=.2)
   axes[0,0].legend(fontsize=7);fig.suptitle('RSITMD seed0: up to200 rounds; partial-overlap Hybrid; uplink TX costs')
   fig.tight_layout();fig.savefig(OUT/(filename+'.png'),dpi=170);fig.savefig(OUT/(filename+'.pdf'));plt.close(fig)
 lines=['RSITMD200-round study: main four methods plus paired crossmodal ablation.',
  'Costs include reused training prefixes. Energy is a modeled uplink waveform bound; time is occupied uplink time.',
  'Joint success: request procedure complete, retained mR >=95% of final200-round retrain, forget KL <=0.01, three consecutive evaluations.',
  'An empirical plateau does not establish task success or mathematical convergence. Target failure is censored. Single seed.', '']
 for m in summaries:
  lines.append(f"{m['label']}: {m['status']}; rounds={m['completed_rounds']}; joint={m['joint_target_confirmed']}; plateau={m['utility_plateau']}")
 (OUT/'results.txt').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':build_report()
