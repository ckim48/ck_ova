"""Paired frozen-VLM-update radio sweep; all infeasible solver outcomes retained."""
import json
from pathlib import Path
import time
import numpy as np
import combined_iq_diagnostic_v1 as study


def main():
    protocol=study.register();dest=study.OUT/'snapshot_sweep';dest.mkdir(exist_ok=True)
    sources=sorted((study.OUT/'snapshots').glob('*/*.npz'))
    assert len(sources)==10, 'Wait for both complete balanced replays.'
    rows=[];proofs=[]
    for path in sources:
        data=np.load(path);info=study.read(path.with_suffix('.json'))
        l,u,ell=data['l'],data['u'],data['ell'];c=info['config']
        cosine=float(np.dot(l.sum(0),u.sum(0))/(np.linalg.norm(l.sum(0))*np.linalg.norm(u.sum(0))))
        for gain in protocol['snapshot_grid']['channel_gain_db']:
            channels=data['channels']*10**(gain/20)
            for ratio in protocol['snapshot_grid']['tolerance_ratios']:
                for mode in ['combined','iq']:
                    key=f'{info["dataset"]}_{info["joint_index"]}_{gain}_{ratio}_{mode}'
                    cached=dest/(key+'.json')
                    if cached.exists():
                        saved=study.read(cached);rows.append(saved['row'])
                        if saved.get('proof'):proofs.append(saved['proof'])
                        continue
                    radio=study.transport(mode,ratio)
                    best,audit=radio.design(l,u,channels,c,ell)
                    row=dict(dataset=info['dataset'],joint_index=info['joint_index'],
                        gain_db=gain,tolerance_ratio=ratio,mode=mode,
                        status='feasible' if best is not None else 'no_feasible_design_found',
                        cosine_FL_FU=cosine,source=str(path.relative_to(study.ROOT)),source_sha256=study.sha(path))
                    proof=None
                    if best is not None:
                        eps=np.array(audit['component_epsilon'])**2
                        usage=np.divide(best['mse'],eps,out=np.zeros_like(eps),where=eps>0)
                        target=np.array(audit['normalized_distortion_limits'])
                        du=best['distortion']/target
                        row.update(head=best['head'],n=best['n'],uplink_s=best['uplink_s'],
                            RF_energy_j=float(best['energy'].sum()),uplink_phase_energy_j=best['objective'],
                            FL_C8_usage=usage[0],FU_C8_usage=usage[1],C9_usage=float(max(du)),
                            allowed_total_MSE=float(eps.sum()),aggregate_MSE=float(best['mse'].sum()))
                        if mode=='iq':
                            meta=dict(head=best['head'],block_variance=best['variance'].tolist(),
                                frame_count=best['n'],user_block_power_w=best['power'].tolist(),
                                energy_bound_j=float(best['energy'].sum()),max_power_w=float(best['power'].max()),
                                component_epsilon=audit['component_epsilon'],realized_error_to_raw=0.)
                            mapped=study.map_iq_to_combined(l,u,channels,c,meta)
                            proof=dict(dataset=info['dataset'],joint_index=info['joint_index'],gain_db=gain,
                                tolerance_ratio=ratio,IQ_RF_energy_j=row['RF_energy_j'],
                                constructed_Combined_RF_energy_j=mapped['energy_bound_j'],
                                same_aggregate_MSE=bool(np.allclose(np.array(mapped['block_variance']).sum(0),best['variance'].sum(0))),
                                no_greater_RF_energy=bool(mapped['energy_bound_j']<=row['RF_energy_j']*(1+1e-8)+1e-18))
                            assert proof['same_aggregate_MSE'] and proof['no_greater_RF_energy']
                            proofs.append(proof)
                    rows.append(row);study.write(cached,dict(row=row,proof=proof))
        print(info['dataset'],info['joint_index'],'completed',len(rows),'points',flush=True)
    study.write(dest/'results.json',rows);study.write(dest/'constructive_checks.json',proofs)
    study.write(dest/'finished.json',dict(unix=time.time(),points=len(rows),snapshots=len(sources),
        no_feasible_count=sum(r['status']!='feasible' for r in rows),constructive_checks=len(proofs),
        source_sha256=study.sha(Path(__file__))))


if __name__=='__main__':main()
