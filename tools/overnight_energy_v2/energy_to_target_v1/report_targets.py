"""All prespecified utility targets; no selective winning thresholds."""
import json,csv,sys
from pathlib import Path
import numpy as np

def csv_write(p,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for x in rows for k in x))
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def report(root,plots=True):
 plan=json.loads((root/'plan.json').read_text());costs=[];summary=[];curves=[]
 for method in plan['methods']:
  d=root/method;p=d/'rounds.jsonl'
  if not p.exists():continue
  rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
  mf=json.loads((d/'manifest.json').read_text());initial=json.loads((d/'initial_evaluation.json').read_text())
  targets={k:list(v) for k,v in plan['target_sweep'].items()}
  targets['loss']=sorted(set(targets['loss']+[initial['loss']-v for v in plan['relative_loss_improvements']]))
  targets['accuracy']=sorted(set(targets['accuracy']+[initial['accuracy']+v for v in plan['relative_accuracy_improvements']]))
  for metric,levels in targets.items():
   for target in levels:
    for consecutive in [1,3]:
     hit=None
     for i in range(len(rows)-consecutive+1):
      w=rows[i:i+consecutive]
      if [x['round'] for x in w]!=list(range(w[0]['round'],w[0]['round']+consecutive)):continue
      if all(x['validation'][metric]<=target if metric=='loss' else x['validation'][metric]>=target for x in w):hit=w[-1];break
     x=dict(method=method,metric=metric,target=target,consecutive_rounds=consecutive,
      already_met_before_request=initial[metric]<=target if metric=='loss' else initial[metric]>=target,
      status='reached' if hit else 'not_reached_in_observed_window',run_status=mf['status'],
      scope='Utility target only; not a matched deletion-quality certificate',common_prefix_cost_excluded=True)
     if hit:x.update(round=hit['round'],post_request_rounds=hit['round']-60,
      energy_bound_j=hit['cumulative_communication_energy_j'],expected_energy_j=hit['cumulative_expected_communication_energy_j'],
      uplink_s=hit['cumulative_uplink_s'],delay_without_optimizer_s=hit['cumulative_delay_without_optimizer_or_overhead_s'],
      delay_with_optimizer_s=hit['cumulative_delay_with_optimizer_without_overhead_s'],
      individual_requests_delivered=hit['individual_request_deliveries_total'],request_users_total=hit['request_users_total'])
     costs.append(x)
  plateau=None
  for i in range(9,len(rows)):
   w=rows[i-9:i+1]
   if [x['round'] for x in w]!=list(range(w[0]['round'],w[0]['round']+10)):continue
   losses=np.array([x['validation']['loss'] for x in w]);acc=np.array([x['validation']['accuracy'] for x in w])
   if abs(losses[5:].mean()-losses[:5].mean())<=.001 and np.ptp(losses)<=.005 and abs(acc[5:].mean()-acc[:5].mean())<=.002:
    plateau=rows[i];break
  last=rows[-1]
  summary.append(dict(method=method,status=mf['status'],observed_post_request_rounds=len(rows),
   loss=last['validation']['loss'],accuracy=last['validation']['accuracy'],energy_bound_j=last['cumulative_communication_energy_j'],
   plateau_status='criterion_observed' if plateau else 'not_observed',plateau_round=None if plateau is None else plateau['round'],
   plateau_energy_j=None if plateau is None else plateau['cumulative_communication_energy_j'],
   plateau_interpretation='Ten-round validation stability proxy; not proof of optimizer convergence or useful accuracy'))
  for r in rows:curves.append(dict(method=method,round=r['round'],loss=r['validation']['loss'],accuracy=r['validation']['accuracy'],
   energy_bound_j=r['cumulative_communication_energy_j'],uplink_s=r['cumulative_uplink_s']))
 csv_write(root/'target_costs.csv',costs);csv_write(root/'summary.csv',summary);csv_write(root/'curves.csv',curves)
 (root/'target_costs.json').write_text(json.dumps(costs,indent=2));(root/'summary.json').write_text(json.dumps(summary,indent=2))
 if curves and plots:
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,axes=plt.subplots(2,2,figsize=(11,7))
  for method in plan['methods']:
   r=[x for x in curves if x['method']==method]
   if not r:continue
   for ax,metric in [(axes[0,0],'loss'),(axes[0,1],'accuracy')]:
    ax.plot([x['round']-60 for x in r],[x[metric] for x in r],label=method);ax.set_ylabel(metric);ax.set_xlabel('Post-request rounds')
   for ax,metric in [(axes[1,0],'loss'),(axes[1,1],'accuracy')]:
    ax.plot([x['energy_bound_j']*1000 for x in r],[x[metric] for x in r],label=method);ax.set_ylabel(metric);ax.set_xlabel('Cumulative communication energy bound (mJ)')
  for ax in axes.flat:ax.grid(alpha=.25)
  axes[0,0].legend();fig.suptitle('Energy to utility target; common pretraining cost excluded')
  fig.tight_layout();fig.savefig(root/'energy_to_target.png',dpi=180);fig.savefig(root/'energy_to_target.pdf');plt.close(fig)
 return summary

if __name__=='__main__':print(json.dumps(report(Path(sys.argv[1])),indent=2))
