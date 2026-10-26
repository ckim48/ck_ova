"""Independent physical-accounting checks, including all eligible feedback heads."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from receiver_pilot import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True)
    args=p.parse_args();r=args.project.resolve();sys.path[:0]=[str(r),str(r/'tools/overnight_energy_v2')]
    from phase_split_grid import make_inputs
    result=dict(audit_sha256=sha(__file__),cases={})
    for name in ['receiver_control_v1','receiver_control_v1_request91']:
        out=r/'results/overnight_energy_v2'/name;plan=json.loads((out/'plan.json').read_text());c=plan['config']
        rows=json.loads((out/'radio_results.json').read_text())
        with np.load(out/'prepared.npz') as z:L,Q=[z[k].copy() for k in ['learning','quantized']]
        payload=make_inputs(L,Q,c)[4]
        checks=[]
        for row in rows:
            path=out/(row['name']+'_design.npz')
            if row['status']!='feasible' or not path.exists():continue
            with np.load(path) as z:design={k:z[k].copy() for k in z.files}
            a=design['analog_rms'];d=design['digital_sqrt_power'];s=design['receiver'];H=design['normalized_channels'];rho=design['rho']
            symbols=design['symbols'];digital_symbols=design['digital_symbols'];dims=design['dimensions'];users=design['users']
            orthogonal=row['name']=='oma_grid';digital=row['name'] in ['hybrid','oma_grid']
            if orthogonal:
                power=np.maximum(a*a,d*d);energy=((symbols*a*a+digital_symbols*d*d)/c['symbol_rate_hz']).sum(axis=1)
            else:
                power=(a+d)**2;energy=(symbols*(a+d)**2/c['symbol_rate_hz']).sum(axis=1)
            calculated_energy=float(energy.sum()+c['users']*c['base_overhead_j']+c['head_overhead_j'])
            reported=row['radio_energy_bound_j']
            assert abs(calculated_energy-reported)<=max(1e-11,reported*1e-6)
            assert np.max(power)<=c['power_limit_w']*(1+1e-6)
            assert np.max(energy+c['base_overhead_j'])<=c['available_energy_j']*(1+1e-6)
            assert np.all(symbols>=dims) and np.max(symbols)/c['symbol_rate_hz']<=c['coherence_time_s']*(1+1e-10)
            mismatch=np.abs(s*H*a-rho).sum(axis=0)
            bound=float(np.linalg.norm(mismatch+s*np.sqrt(dims/(2*symbols))))
            assert abs(bound-row['error_bound'])<=max(1e-12,bound*1e-6)
            worst_rate_margin=None
            if digital:
                for m in range(len(dims)):
                    active=np.flatnonzero(payload[users,m]>0)
                    active=active[np.argsort(-H[active,m],kind='stable')]
                    for rank,index in enumerate(active):
                        signal=H[index,m]**2*d[index,m]**2
                        interference=sum(H[j,m]**2*d[j,m]**2 for j in active[rank+1:])
                        if not orthogonal:interference+=(H[:,m]@a[:,m])**2
                        gamma=np.expm1(np.log(2)*payload[users[index],m]/digital_symbols[m])
                        # Compare normalized powers, avoiding unstable large-rate logs.
                        margin=signal-gamma*(1+interference)
                        worst_rate_margin=margin if worst_rate_margin is None else min(worst_rate_margin,margin)
                        assert signal+1e-6*max(1.,signal)>=gamma*(1+interference)
            checks.append(dict(name=row['name'],radio_energy_j=calculated_energy,max_power_w=float(np.max(power)),
                calculated_error_bound=bound,worst_normalized_sic_power_margin=worst_rate_margin,status='passed'))
        trials=[x for x in rows if x.get('role')=='head_trial_physical' and x.get('status')=='feasible' and x.get('eligible_same_application_budget')]
        candidates=[x for x in rows if x.get('role')=='physical_model' and x['name'].startswith('feedback_') and x.get('status')=='feasible' and x.get('eligible_same_application_budget')]
        best_trial=min(trials,key=lambda x:x['energy_bound_j']) if trials else None
        selected=min(candidates,key=lambda x:x['energy_bound_j']) if candidates else None
        result['cases'][name]=dict(physical_checks=checks,all_head_trial_count=len([x for x in rows if x.get('role')=='head_trial_physical']),
            eligible_head_trial_count=len(trials),best_eligible_feedback_head=best_trial,
            current_feedback_selection=selected,
            eligible_head_selection_complete=(best_trial is None and selected is None) or (best_trial is not None and selected is not None and selected['selected_trial']==best_trial['name']),
            existing_core_sources_intact=all(sha(r/n)==v for n,v in plan['core_sha256'].items()))
    old=json.loads((r/'results/overnight_energy_v2/burst_joint_v3/plan.json').read_text())
    result['old_campaign_source_integrity']={n:sha(r/n)==v for n,v in old['source_sha256'].items()}
    assert all(result['old_campaign_source_integrity'].values())
    target=r/'results/overnight_energy_v2/receiver_control_v1/final_audit.json';write(target,result)
    print(json.dumps(dict(cases={k:{'checks':len(v['physical_checks']),'all_heads_considered':v['eligible_head_selection_complete']} for k,v in result['cases'].items()},old_sources_intact=True)))


if __name__=='__main__':main()
