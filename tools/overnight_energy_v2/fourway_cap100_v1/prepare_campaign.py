"""Preregister a common 0.10 J TX cap and four system comparisons."""
from pathlib import Path
import sys,json,time,importlib.util
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
OUT=ROOT/'results/overnight_energy_v2/fourway_cap100_v1'
sys.path.insert(0,str(HERE))
import radio_common as common
import radio_guard_cap as guard
import radio_digital_cap as digital
spec=importlib.util.spec_from_file_location('fourway_clip_parent',ROOT/'tools/overnight_energy_v2/cliperase_fu_v1/run.py')
clip=importlib.util.module_from_spec(spec);spec.loader.exec_module(clip)
core=clip.core
# Digital first verifies that the formerly cap-infeasible path can train.
ARMS={
 'parallel_digital_sum':dict(parallel=True,crossmodal=True,period=0,digital=False,main=True,objective='cliperase_inspired'),
 'proposed_fu_guard_cliperase':dict(parallel=True,crossmodal=True,period=0,digital=False,main=True,objective='cliperase_inspired'),
 'separate_aircomp_periodic':dict(parallel=False,crossmodal=True,period=5,digital=False,main=True,objective='original_safe_proxy'),
 'aircomp_ofdma_periodic':dict(parallel=False,crossmodal=True,period=5,digital=True,main=True,objective='original_safe_proxy')}
LABELS={
 'parallel_digital_sum':'Parallel digital OFDMA',
 'proposed_fu_guard_cliperase':'Proposed + FU protection + CLIPErase-inspired loss',
 'separate_aircomp_periodic':'Separate AirComp',
 'aircomp_ofdma_periodic':'AirComp + OFDMA'}
THRESHOLD=0.17688802083333333
def write(name,obj):core.write(OUT/name,obj)

def prepare():
    plan,requests=clip.prepare();OUT.mkdir(parents=True,exist_ok=True)
    parent=ROOT/'results/overnight_energy_v2/fu_protected_aircomp_v1/protocol.json'
    hashes=dict(json.loads(parent.read_text())['source_hashes'])
    hashes[str(parent.relative_to(ROOT))]=core.sha(parent)
    snapshot=clip.OUT/'fixed_training_updates.npz'
    hashes[str(snapshot.relative_to(ROOT))]=core.sha(snapshot)
    for p in HERE.glob('*.py'):hashes[str(p.relative_to(ROOT))]=core.sha(p)
    protocol=dict(version='fourway_cap100_v1',registered_unix=time.time(),source_hashes=hashes,
      arms=ARMS,labels=LABELS,network_tx_cap_per_transaction_j=.10,previous_cap_j=.05,
      authorization='User requested a larger energy cap and four primary metrics: total energy, final model performance, convergence rate, total time. The previous three arms plus a digital parallel baseline are authorized.',
      common_settings='Same 200 total FL rounds, initial20 checkpoint and charged common prefix,45 exogenous FU requests,5 FU steps/request,data/seed/channels/FL sampling,1MHz total bandwidth,128 tones,Q6 on both digital schemes,0.32W per UAV,0.5J per UAV per transaction,1.31072s transaction deadline. All four receive network TX cap0.10J.',
      objective_and_policy='Retain the requested algorithms: proposed and digital parallel use the same CLIPErase-inspired loss and immediate sum scheduler. Separate AirComp and AirComp+OFDMA keep their existing safe-proxy loss and periodic FU scheduler. This compares complete systems, not an isolated transport-only or loss-only ablation. Periodic service is a policy, not an inherent OFDMA restriction.',
      radio='Proposed sums raw FL+FU and limits total reconstruction RMS by the FU norm. Digital sums FL+FU locally and quantizes once Q6, decodes each UAV sum over disjoint integer subcarriers sharing128 tones. FL/FU are not individually recovered. Separate AirComp transmits FL and FU in distinct transactions; hybrid uses digital OFDMA for FU only.',
      main_metrics=dict(total_energy='Cumulative TX energy bound, including common prefix and every adaptation/FU upload. Not compute/flight/RX energy.',
        final_performance='Retained validation mean recall (mR; six I2T/T2I R1/R5/R10 values) and retained loss after200FL+all45 request procedures.',
        convergence='Retained mR/loss curves versus completed FL rounds, cumulative TX joules and pure uplink seconds; existing fixed mR target confirmed at three consecutive distinct scheduled5-FL evaluations after all45 procedures complete.',
        total_time='Cumulative occupied uplink, plus service clock including idle waiting. Both exclude local computation/SIC/control/downlink. Server wall-clock training runtime recorded separately and is not UAV end-to-end latency.'),
      utility_target_mR=THRESHOLD,
      target_scope='Utility convergence reporting only; this is the already registered95%-retrain validation threshold, not a hyperparameter selection rule or deletion-success criterion. Keep existing deletion/MIA diagnostics in raw results, off the four-metric headline table. Missing targets and physical failures remain explicit.',
      convergence_check_rule='Use completed_FL_rounds divisible by5; if multiple evaluations at the same FL count use the last. Three successive scheduled points passing target and all45 procedures; report first crossing and confirmation costs. No interpolation or smoothing.',
      preflight='All20480 coordinates from an existing real training snapshot, never test/reference tuning. Check proposed joint, digital joint/FL-only/FU-only, original analog FL/FU and original digital FU; OFDMA analytical/exhaustive tests; original scheduler tests.',
      versioning='Original hashes and result folders preserved. Radio source copies only change cap/imports. Runtime replaces the too-small2000-frame safety guard by2000 times maximum transaction duration plus arrival horizon; no delay is truncated.',
      execution='One GPU lock, four sequential200-round arms. Do not restart paused historical campaigns or heartbeat. No other algorithm sweeps.',
      limitations='Single seed, ideal CSI/coding/no modeled CP overhead. Raw quality diagnostics preserved; low TX energy or utility target does not certify deletion.')
    path=OUT/'protocol.json'
    if path.exists():protocol=json.loads(path.read_text())
    else:write('protocol.json',protocol);write('requests.json',requests)
    assert protocol['network_tx_cap_per_transaction_j']==common.base.CAP==.10
    assert json.loads((OUT/'requests.json').read_text())==requests
    for n,h in protocol['source_hashes'].items():assert core.sha(ROOT/n)==h,n
    return plan,requests

def main():
    plan,_=prepare();c=plan['config'];z=np.load(clip.OUT/'fixed_training_updates.npz')
    l,u,old,ch=z['FL'],z['FU'],z['old_FU'],z['channels'];zero=np.zeros_like(l)
    kernel=common.base.old.ofdma.checks();assert kernel['passed']
    sched=clip.stream.self_check();records={}
    cases=[('proposed_joint',guard.analog,l,u),('digital_joint',digital.analog,l,u),
           ('digital_FL_only',digital.analog,l,zero),('digital_FU_only',digital.analog,zero,u),
           ('original_analog_FL',common.analog,l,zero),('original_analog_FU',common.analog,zero,old),
           ('original_digital_FU',lambda ll,uu,ch,c,seed:common.digital_fu(uu,ch,c),zero,old)]
    for name,fn,ll,uu in cases:
        v,m=fn(ll,uu,ch,c,500000);records[name]=m
        assert v is not None and m['status']=='feasible',(name,m)
        assert m['conditional_rmse']<=m['epsilon']*(1+3e-5)+1e-9
        assert m['energy_bound_j']<=.10*(1+2e-6)
        assert m['max_power_w']<=c['power_limit_w']*(1+2e-6)
        assert m['max_user_energy_j']<=c['available_energy_j']*(1+2e-6)
        assert m['uplink_s']<=c['round_uplink_deadline_s']+1e-12
        if 'subcarrier_counts' in m:assert np.all(np.asarray(m['subcarrier_counts']).sum(0)<=128)
        assert np.isfinite(v).all()
        print(name,{k:m.get(k) for k in ['energy_bound_j','uplink_s','conditional_rmse','epsilon']},flush=True)
    # Spawn workers import the versioned literal cap, not the old .05 constant.
    assert common.base.__file__.endswith('radio_parallel_cap.py')
    assert guard.base is digital.base is common.base
    write('preflight.json',dict(passed=True,unix=time.time(),dimension=l.shape[1],OFDMA_kernel=kernel,
        scheduler_checks=sched,records=records,config=c,network_tx_cap_j=.10))
    write('queue_registered.json',dict(unix=time.time(),arms=list(ARMS),status='registered_after_preflight',
        GPU_lock='results/overnight_energy_v2/association_objective_gpu.lock',sequential=True))
    print('PREFLIGHT_PASSED',flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:write('preflight_failure.json',dict(unix=time.time(),error=repr(e)));raise
    finally:common.base.close()
