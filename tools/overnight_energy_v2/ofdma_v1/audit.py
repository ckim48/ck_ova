"""CPU fixed-update energy/deadline audit; no model convergence claims."""
from pathlib import Path
import os,sys,json,time,hashlib,traceback
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import radio_ofdma as radio
import ofdma,numpy as np
ROOT=radio.ROOT;OUT=ROOT/'results/overnight_energy_v2/ofdma_v1'


def write(name,x):
    OUT.mkdir(parents=True,exist_ok=True);p=OUT/name;t=p.with_suffix('.tmp')
    t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)


def main():
    write('cpu_checks.json',ofdma.checks())
    if '--cpu-only' in sys.argv:print('CPU_CHECKS_PASSED');return
    snapshot=ROOT/'results/overnight_energy_v2/request_projection_v4/pilot/sum/common_updates.npz'
    z=np.load(snapshot);l,u=z['learning'],z['unlearning']
    source=ROOT/'results/overnight_energy_v2/retrieval_long200_v3'
    plan=json.loads((source/'plan.json').read_text());channels=np.load(source/'channels.npy')
    from phase_split_grid import make_inputs
    dims=make_inputs(l,u,plan['config'])[6]
    write('protocol.json',dict(version='ofdma_v1',created_unix=time.time(),
      replacement={'aircomp_oma':'AirComp+OFDMA','oma_only':'OFDMA-only'},
      old_is_tdma=True,subcarriers=128,channel='same flat/noise-normalized channels',
      capacity='L*f*log2(1+h^2*P/f); sum f<=1; integer 128-tone allocation; P<=0.32 W per UAV',
      energy='sum_u L/W*P_u; users share the same time interval; not sum of user times',
      interpretation='CPU fixed real updates only. Energy minimized at each common deadline. Smallest successful tested duration is a feasible grid result, not proven minimum latency.',
      timing_issue='Old TDMA frame=64d and partial-overlap frame=62d forced a 3.125% gap; these were not minimum-completion-time estimates.',
      local_compute='Previous time excluded local compute. Both FL and FU are calculated from the same round-start theta. No automatic parallel compute benefit is attributed to NOMA. Sequential FU based on an FL-updated theta would be a different algorithm for all access modes.',
      budgets='Same bandwidth/noise PSD, per-UAV total RF power, battery, canonical Q8 FU, absolute aggregation tolerance and head enumeration.',
      hardware_scope='Ideal coding/CSI, flat channels, zero CP/guard overhead shared with prior ideal model; no frequency-selective diversity or modem runtime claims.',
      snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
      references=['https://arxiv.org/abs/1605.08518'],
      sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}))
    rows=[]
    # First validate the OFDMA replacement on the original frame, then scan
    # common shorter budgets. Full NOMA gets the same full interval, not 62/64.
    for mult in [64,8,16,24,32,48]:
        c=dict(plan['config'],symbols_per_coordinate=mult,ofdma_subcarriers=128)
        for method in ['aircomp_ofdma','ofdma_only','hybrid']:
            write('status.json',dict(stage='fixed_update_audit',pid=os.getpid(),multiplier=mult,method=method,unix=time.time()))
            t=time.time();v,r=radio.communicate(l,u,channels,c,method,100021)
            # Keep complete allocation evidence, concise table separately.
            write(f'design_{mult}_{method}.json',r)
            row={k:r.get(k) for k in ['status','energy_bound_j','expected_energy_j','uplink_s','head','error_bound','epsilon','analog_uplink_s','digital_uplink_s','optimizer_wall_s']}
            row.update(method=method,multiplier=mult,deadline_s=float((dims*mult).sum()/c['symbol_rate_hz']),wall_s=time.time()-t)
            rows.append(row);write('fixed_update_results.json',rows);print(json.dumps(row),flush=True)
    write('finished.json',dict(unix=time.time(),scope='fixed-update CPU audit; no new GPU learning run'))
    write('status.json',dict(stage='completed',unix=time.time()))


if __name__=='__main__':
    try:main()
    except Exception as e:write('status.json',dict(stage='failed',error=repr(e)));traceback.print_exc();raise
    finally:radio.close()
