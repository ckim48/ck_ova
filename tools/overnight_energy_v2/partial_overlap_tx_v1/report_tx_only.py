"""Read-only reaccounting of original trajectories into uplink-only metrics."""
from pathlib import Path
import csv, hashlib, json, sys, time
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
OUT = ROOT/'results/overnight_energy_v2/partial_overlap_tx_v1/tx_only_trajectories'

def write_csv(path, rows):
    if not rows: return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

def reaccount(rows, overhead):
    result=[]; energy=0.; expected=0.; duration=0.
    for row in rows:
        tx = row['transmission']
        if tx.get('status') != 'feasible': break
        eb = float(tx['energy_bound_j'])-overhead
        ee = float(tx['expected_energy_j'])-overhead
        assert eb >= -1e-9 and ee >= -1e-9
        energy += max(0., eb); expected += max(0., ee); duration += float(tx['uplink_s'])
        result.append(dict(round=row['round'], loss=row['validation']['loss'], accuracy=row['validation']['accuracy'],
                           total_delay_s=duration, tx_energy_bound_j=energy, tx_energy_expected_j=expected,
                           requests_required=row['request_users_total'], requests_individually_delivered=row['individual_request_deliveries_total']))
    return result

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary=[]; curves=[]; targets=[]; inputs=[]
    for campaign in ['deletion_window_screen_v1','energy_to_target_v1','partial_overlap_tx_v1/learning_comparison']:
        source=ROOT/'results/overnight_energy_v2'/campaign
        if not (source/'plan.json').exists(): continue
        plan=json.loads((source/'plan.json').read_text()); c=plan['config']
        overhead=c['users']*c['base_overhead_j']+c['head_overhead_j']
        for method in plan['methods']:
            p=source/method/'rounds.jsonl'
            if not p.exists(): continue
            raw=p.read_bytes(); lines=raw.decode().splitlines(); rows=[]
            for i, line in enumerate(lines):
                if not line.strip(): continue
                try: rows.append(json.loads(line))
                except json.JSONDecodeError:
                    if i != len(lines)-1: raise
            if not rows: continue
            r=reaccount(rows, overhead)
            if not r: continue
            status=json.loads((source/method/'manifest.json').read_text())['status']
            initial=json.loads((source/method/'initial_evaluation.json').read_text())
            inputs.append(dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(raw).hexdigest(),removed_fixed_overhead_per_round_j=overhead))
            summary.append(dict(campaign=campaign,method=method,status=status,observed_rounds=len(r),**r[-1]))
            curves.extend(dict(campaign=campaign,method=method,**x) for x in r)
            levels={k:list(v) for k,v in plan['target_sweep'].items()}
            levels['loss']=sorted(set(levels['loss']+[initial['loss']-v for v in [0,.001,.002,.005,.01,.02,.05]]))
            levels['accuracy']=sorted(set(levels['accuracy']+[initial['accuracy']+v for v in [0,.001,.002,.005,.01,.02,.05]]))
            for metric, vals in levels.items():
                for target in vals:
                    for count in [1,3]:
                        hit=None
                        for i in range(len(r)-count+1):
                            window=r[i:i+count]
                            if [x['round'] for x in window] != list(range(window[0]['round'],window[0]['round']+count)): continue
                            if all(x[metric]<=target if metric=='loss' else x[metric]>=target for x in window): hit=window[-1]; break
                        item=dict(campaign=campaign,method=method,metric=metric,target=target,consecutive_rounds=count,
                                  status='reached' if hit else 'not_reached_in_observed_window',run_status=status,
                                  already_met_before_request=initial[metric]<=target if metric=='loss' else initial[metric]>=target)
                        if hit: item.update(hit)
                        targets.append(item)
    payload=dict(updated_unix=time.time(), metric_definition='Total delay: cumulative occupied uplink transmission time only. Energy: cumulative uplink transmit energy. All receiver/reconstruction/SIC/optimization/training/evaluation/control/downlink time and energy excluded for all methods.',
                 qualifications=['Original trajectory durations remain actual recorded frame durations; partial overlap is not retroactively applied to old trajectories.',
                                 'No checkpoint, training source or original result modified.',
                                 'Expected energy assumes uncorrelated random codewords; bound retains the manuscript waveform-correlation upper bound.',
                                 'Incomplete or infeasible tasks are excluded from task-completion rankings. Utility targets alone do not prove matched forgetting quality.'],
                 inputs=inputs,summary=summary)
    (OUT/'summary.json').write_text(json.dumps(payload,indent=2)+'\n')
    (OUT/'target_costs.json').write_text(json.dumps(targets,indent=2)+'\n')
    write_csv(OUT/'summary.csv',summary); write_csv(OUT/'curves.csv',curves); write_csv(OUT/'target_costs.csv',targets)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    if '--after-continuation' in sys.argv:
        q=ROOT/'results/overnight_energy_v2/energy_to_target_v1/queue_status.json'
        while True:
            s=json.loads(q.read_text())['status']
            if s.startswith(('completed','blocked','stopped','failed')): break
            time.sleep(30)
    main()
