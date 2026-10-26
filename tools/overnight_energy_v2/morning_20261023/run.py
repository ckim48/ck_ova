"""Read-only consolidation of completed comparison and pending overnight work."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,runpy,hashlib,datetime
ROOT=Path.cwd();BASE=ROOT/'results/overnight_energy_v2';PROXY=BASE/'proxy_objective_v2'
OUT=BASE/'morning_20261023';OUT.mkdir(parents=True,exist_ok=True)
assert not (OUT/'comparison_audit.json').exists()
helpers=runpy.run_path(str(ROOT/'tools/overnight_energy_v2/final_metrics_audit_v1/run.py'),run_name='audit_helpers')
tests=helpers['self_check']();summary=json.loads((PROXY/'summary.json').read_text());results=[]
for name in ['proposed','aircomp_only','aircomp_oma','oma_only','hybrid_no_crossmodal']:
 d=PROXY/name;paths=sorted((d/'rounds').glob('*.json'));rows=[json.loads(p.read_text()) for p in paths]
 assert [r['round'] for r in rows]==list(range(rows[0]['round'],rows[-1]['round']+1))
 evaluated=[]
 for row in rows:
  e=row.get('evaluation')
  if not e or row['round']%5:continue
  assert len(e['per_request'])==5
  evaluated.append(dict(round=row['round'],retained=e['retained'],forget=e['forget'],
   worst_request_KL=max(v['KL_retrain_to_candidate'] for v in e['per_request'].values()),
   requests_handled=row['request_algorithm_completed'] is True and row['logical_requests']==5,
   energy_j=row['cumulative_tx_energy_bound_j'],uplink_s=row['cumulative_uplink_s']))
 last=evaluated[-1];strict=helpers['target'](evaluated);old=next(x for x in summary['methods'] if x['method']==name)
 mia_path=d/'mia_200.json'
 results.append(dict(method=name,training_complete=rows[-1]['round']==200,last_committed_round=rows[-1]['round'],
  last_evaluated_round=last['round'],retained_mR=last['retained']['mR'],retained_loss=last['retained']['loss'],
  mean_forget_KL=last['forget']['KL_retrain_to_candidate'],worst_request_KL=last['worst_request_KL'],
  forget_i2t_top1_disagreement=last['forget']['i2t_top1_disagreement'],forget_t2i_top1_disagreement=last['forget']['t2i_top1_disagreement'],
  per_request=rows[-1].get('evaluation',{}).get('per_request'),all_five_requests_processed=last['requests_handled'],
  total_energy_j=rows[-1]['cumulative_tx_energy_bound_j'],total_uplink_s=rows[-1]['cumulative_uplink_s'],
  strict_joint_target=strict,previous_mean_only_target=old['joint_target'],
  final_mia=json.loads(mia_path.read_text()) if mia_path.exists() else None,
  prefix_proxy_cost=json.loads((d/'prefix_and_proxy_cost.json').read_text()),
  source_hashes={'last_round':hashlib.sha256(paths[-1].read_bytes()).hexdigest(),'summary':hashlib.sha256((PROXY/'summary.json').read_bytes()).hexdigest()}))
mem=json.loads((BASE/'final_metrics_audit_v1/report.json').read_text())
report=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 target_definition='mR>=0.29304036883016427, mean and worst-request forget KL<=0.01, all5 requests processed,3 consecutive five-round evaluations starting>=25. Stricter than old mean-only summary.',
 completed_transport_comparison=results,memprune={k:v for k,v in mem.items() if k!='matched_horizon_reference'},
 latest_memprune_matched_horizon=mem['matched_horizon_reference'][-1] if mem['matched_horizon_reference'] else None,
 checks=tests,limitations=['Single development seed0, not a general ranking or a certified deletion guarantee.',
 'Method-specific MemPrune final reference and MIA pending; unequal local optimizer and step budget forbids same-round utility winner inference.',
 'TX waveform energy bounds and occupied uplink time include matching prefixes/proxy construction; local/SIC/flight time excluded.',
 'Latest cross-modal ablation intentionally paused for higher-priority request-level experiment, not a failed method. Do not rank incomplete total costs.',
 'The older retrieval_long200_v3 summary uses a different objective. Do not mix its thresholds/results with proxy_objective_v2.'])
p=OUT/'comparison_audit.json';p.write_text(json.dumps(report,indent=2,allow_nan=False))
print(json.dumps({'file':str(p),'rows':[{'method':x['method'],'complete':x['training_complete'],'round':x['last_committed_round'],'target':x['strict_joint_target']} for x in results]}))
