from common import *
import csv
import numpy as np

def build_report():
 plan=json.loads((OUT/'plan.json').read_text());rows=[];summary=[];targets=[]
 refpath=OUT/'retrain/rounds.jsonl'
 ref=[json.loads(x) for x in refpath.read_text().splitlines()] if refpath.exists() else []
 threshold=ref[-1]['evaluation']['retained']['mR']*.95 if ref and 'evaluation' in ref[-1] else None
 for method in plan['methods']:
  dest=OUT/method;p=dest/'rounds.jsonl';manifest=dest/'manifest.json'
  if not p.exists():continue
  trajectory=[json.loads(line) for line in p.read_text().splitlines()]
  state=json.loads(manifest.read_text()) if manifest.exists() else dict(status='running')
  evaluated=[]
  for r in trajectory:
   if 'evaluation' not in r:continue
   m=r['evaluation'];row=dict(method=method,round=r['round'],retained_mR=m['retained']['mR'],retained_loss=m['retained']['loss'],
     forget_KL=m['forget'].get('KL_retrain_to_candidate'),energy_bound_j=r['cumulative_tx_energy_bound_j'],
     energy_expected_j=r['cumulative_tx_energy_expected_j'],uplink_s=r['cumulative_uplink_s'],request_algorithm_completed=r['request_algorithm_completed'])
   evaluated.append(row);rows.append(row)
  if not evaluated:continue
  joint=[r for r in evaluated if r['request_algorithm_completed'] and threshold is not None and r['retained_mR']>=threshold and r['forget_KL']<=.01]
  last=evaluated[-1];mia=dest/'mia.json';test=dest/'final_test.json'
  summary.append(dict(method=method,status=state['status'],final=last,
    joint_target_first_observed=joint[0] if joint else None,
    mia=json.loads(mia.read_text()) if mia.exists() else None,test=json.loads(test.read_text()) if test.exists() else None))
  for name,values in [('mR',plan['utility_target_grid_mR']),('loss',plan['loss_target_grid'])]:
   for value in values:
    hits=[r for r in evaluated if (r['retained_mR']>=value if name=='mR' else r['retained_loss']<=value)]
    targets.append(dict(method=method,target=name,value=value,first_observed=hits[0] if hits else None,
                        scope='utility only; does not certify deletion; evaluations every5 rounds except request round'))
 write(OUT/'summary.json',dict(joint_target_retained_mR=threshold,joint_target_forget_KL=.01,methods=summary,utility_targets=targets))
 if rows:
  with (OUT/'convergence.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,axes=plt.subplots(2,3,figsize=(14,8))
  for method in plan['methods']:
   r=[x for x in rows if x['method']==method]
   if not r:continue
   for ax,xkey,ykey in [(axes[0,0],'round','retained_mR'),(axes[0,1],'energy_bound_j','retained_mR'),(axes[0,2],'uplink_s','retained_mR'),
       (axes[1,0],'round','retained_loss'),(axes[1,1],'round','forget_KL'),(axes[1,2],'round','energy_bound_j')]:
    ax.plot([x[xkey] for x in r],[x[ykey] for x in r],'-o',markersize=3,label=method);ax.set_xlabel(xkey);ax.set_ylabel(ykey);ax.grid(alpha=.2)
  axes[0,0].legend(fontsize=8);fig.suptitle('RSITMD seed0 pilot: modeled uplink TX costs; joint quality gate required')
  fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=180);fig.savefig(OUT/'comparison.pdf');plt.close(fig)
 lines=['RSITMD image-caption retrieval: five-arm initial experiment','Costs: uplink TX only; energy values are model bounds, not measured hardware joules.',
        'Joint target: retained mR >=95% of final finite-budget retrain; forget bidirectional KL <=0.01; request procedure complete.',
        'A no-hit or incomplete method is not a low-energy winner. Single seed; no statistical superiority claim.','']
 for x in summary:
  r=x['final'];lines.append(f"{x['method']}: {x['status']}; round={r['round']}; mR={r['retained_mR']:.6f}; loss={r['retained_loss']:.6f}; forgetKL={r['forget_KL']:.6g}; TXbound={r['energy_bound_j']:.6g}J; uplink={r['uplink_s']:.6g}s; joint_target={'reached' if x['joint_target_first_observed'] else 'not reached'}")
 (OUT/'results.txt').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':build_report()
