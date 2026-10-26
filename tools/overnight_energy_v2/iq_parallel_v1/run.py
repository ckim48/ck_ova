"""Registered continuation: I/Q transport and head control, no source overwrite."""
from pathlib import Path
import os,sys,json,time,fcntl,hashlib,subprocess,traceback,importlib.util
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
OUT=ROOT/'results/overnight_energy_v2/iq_parallel_v1'
SOURCE=ROOT/'results/overnight_energy_v2/streaming_requests_v1'
sys.path.insert(0,str(ROOT/'tools/overnight_energy_v2/streaming_requests_v1'))
spec=importlib.util.spec_from_file_location('iq_stream_parent',ROOT/'tools/overnight_energy_v2/streaming_requests_v1/run.py')
stream=importlib.util.module_from_spec(spec);spec.loader.exec_module(stream)
sys.path.insert(0,str(HERE))
import radio_iq
import checks
from types import SimpleNamespace
from scheduler import Queue
core=stream.core
ARMS={
 'iq_sum':dict(parallel=True,crossmodal=True,period=0,digital=False,main=True,transport='iq',project=False),
 'iq_project':dict(parallel=True,crossmodal=True,period=0,digital=False,main=True,transport='iq',project=True),
 'packed_sum':dict(parallel=True,crossmodal=True,period=0,digital=False,main=True,transport='packed_sum',project=False),
 'separate_packed_project':dict(parallel=True,crossmodal=True,period=0,digital=False,main=True,transport='separate',project=True)}
CURRENT=None
ORIGINAL_VECTORS=stream.vectors

def write(name,obj):core.write(OUT/name,obj)
def status(**obj):write('status.json',dict(unix=time.time(),pid=os.getpid(),**obj))

def prepare():
    stream.OUT=SOURCE
    plan,requests=stream.prepare() # verify every frozen parent source and reused input
    stream.OUT=OUT
    OUT.mkdir(parents=True,exist_ok=True)
    parent=json.loads((SOURCE/'protocol.json').read_text())
    hashes=dict(parent['source_hashes'])
    for p in sorted(HERE.glob('*.py')):hashes[str(p.relative_to(ROOT))]=core.sha(p)
    hashes[str((SOURCE/'protocol.json').relative_to(ROOT))]=core.sha(SOURCE/'protocol.json')
    spec=dict(version='iq_parallel_v1',registered_unix=time.time(),arms=ARMS,source_hashes=hashes,
      authorization='User explicitly requested continuing with I/Q experiments after the running streaming campaign; earlier unrelated campaigns stay paused.',
      predecessor=dict(pid=134207,start='881579733',command_contains='streaming_requests_v1/run.py',finished=str(SOURCE/'finished.json')),
      hypothesis='Separate recovery alone does not improve the ideal sum. Test transport only versus post-reception global control; no assumed proposed winner.',
      stages='CPU analytic/physical checks now; actual full20480-coordinate training-update preflight under GPU lock; then all four feasible trajectories through200 FL rounds and45 request procedures. No promotion based on final-reference/test outcomes.',
      workloads='Identical causal wall-of-uplink arrivals, batches, five FU steps, FU rate0.0035, cosine FL schedule, fixed safe proxy, cross-modal objective and reference as streaming_requests_v1. Pure uplink excludes compute/decoding/downlink/control.',
      transport='I/Q: FL and FU occupy real and imaginary axes of each complex channel symbol after perfect channel phase inversion. Each branch has its own receive scale. Their RF powers ADD under a SINGLE0.32W per-UAV power limit.',
      fairness='All new arms may pack two real coordinates per complex symbol. Raw-sum AirComp uses both I/Q axes and may need half as many symbols as two separately recovered real aggregates. No extra bandwidth, RF chains, antennas or digital payload is supplied to I/Q.',
      separate_control='separate_packed_project computes FL and FU from the SAME round-start model, transmits them in two orthogonal time phases with I/Q coordinate packing, then applies the IDENTICAL global head controller. It is distinct from the older FU-first periodic learning scheduler; the same-state control isolates transport.',
      dimension='Each128-real-coordinate component needs64 complex symbols when packed alone; simultaneous FL/FU needs128 complex symbols. Optimize common half-frame grid and all ten heads; separated arm examines every positive integer FL/FU time split at each total length.',
      radio_limits=dict(bandwidth_symbols_per_s=1e6,user_power_w=.32,user_energy_j=.5,network_energy_j=.05,half_frame_grid=list(radio_iq.COUNTS)),
      error='Each separately recovered component has RMS budget0.2*its block-norm bound, before head control; independent variances sum to the same total raw-sum budget used by packed_sum. Nonlinear post-controller error is measured, not asserted bounded by that pre-controller MSE.',
      controller='If estimated global FL dot FU is negative, remove the FL component opposing the composite FU direction: L_tilde=L-min(0,L dot U)/||U||^2*U; output L_tilde+U. Identity when FU norm^2<=1e-24. This is a predefined first-order heuristic, not a reproduction of a published method or guaranteed deletion.',
      information='Controller receives ONLY noisy aggregate FL and FU plus the head local updates. True aggregates are recorded for diagnostics only; no final reference/test/validation labels select a coefficient. It cannot recover individual UAV/request updates.',
      diagnostics='Log global FL/FU cosine, norm ratio, cancellation, projection amount, transport error and controller output error. Fixed-update residual-phase stress0/1/3/5 degrees is diagnostic only, without selection.',
      scoring='Use the same new814-image gallery and final retained-set reference; report retained mR/loss, mean and worst request KL, prediction differences, MIA, procedure/deadline counts, TX energy and occupied/elapsed uplink. Also report single final quality pass separately from the inherited three-consecutive-check confirmation. Fast packing can finish all FL before the last request; do not call an unconfirmed single final pass a confirmed target. No quality-target energy ranking for unmet targets.',
      common_costs='Inherited round20 initialization and safe-proxy construction are unchanged and charged identically. Their historical costs are not retroactively reduced by I/Q packing. Report adaptation costs too.',
      limitations='Single seed. Perfect phase CSI, synchronization and linear RF. Phase-error sensitivity is NOT trained robustness. Current five-step FU/proxy limitations are intentionally unchanged to isolate the new hypothesis.',
      references=[
       'https://www.analog.com/en/resources/glossary/iq.html',
       'https://openaccess.thecvf.com/content/ICCV2025/html/Patel_Learning_to_Unlearn_while_Retaining_Combating_Gradient_Conflicts_in_Machine_ICCV_2025_paper.html'])
    p=OUT/'protocol.json'
    if p.exists():spec=json.loads(p.read_text())
    else:write('protocol.json',spec);write('requests.json',requests)
    for name,digest in spec['source_hashes'].items():assert core.sha(ROOT/name)==digest,name
    assert json.loads((OUT/'requests.json').read_text())==requests
    return plan,requests

def communicate(l,u,channels,c,seed):
    assert CURRENT in ARMS
    opt=ARMS[CURRENT]
    return radio_iq.communicate(l,u,channels,c,opt['transport'],opt['project'],seed)

def vectors(*args,**kwargs):
    l,u,stats=ORIGINAL_VECTORS(*args,**kwargs)
    _,diag=radio_iq.controller(l.sum(0),u.sum(0),False)
    diag['FU_to_FL_norm_ratio']=diag['FU_norm']/diag['FL_norm'] if diag['FL_norm']>1e-12 else None
    diag['sum_to_component_norm_ratio']=diag['sum_norm']/(diag['FL_norm']+diag['FU_norm']) if diag['FL_norm']+diag['FU_norm']>1e-12 else None
    stats.append(dict(kind='diagnostic_only_true_global_update',**diag))
    return l,u,stats

def main():
    global CURRENT
    plan,requests=prepare()
    if '--prepare' in sys.argv:
        write('cpu_checks.json',checks.self_check());print('PREPARED_AND_CPU_CHECKED');return
    own=(OUT/'launch.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'started.json').exists(),'Duplicate campaign'
    finished=SOURCE/'finished.json';assert finished.exists(),'Predecessor has not finished'
    assert json.loads(finished.read_text())['status']=='all_registered_arms_finished_or_reported_incomplete'
    assert stream.base.identity(134207) is None or stream.base.identity(134207)['start']!='881579733','Predecessor still live'
    gpu=(ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a');fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied'
    assert json.loads((OUT/'cpu_checks.json').read_text())['passed']
    write('started.json',dict(unix=time.time(),identity=stream.base.identity(os.getpid())))
    status(stage='loading_model')
    from train import Engine
    from dataset import RetrievalData
    engine=Engine(RetrievalData(),dict(plan));channels=np.load(core.SOURCE/'channels.npy')
    p=ROOT/'results/overnight_energy_v2/proxy_objective_v2/proxies/analog'
    proxy_meta=json.loads((p/'complete.json').read_text());proxy_theta=core.load_theta(p/'round_020.pt',engine.device)
    prefix=json.loads((SOURCE/'common_prefix_cost.json').read_text());write('common_prefix_cost.json',prefix)
    theta=core.load_theta(core.SOURCE/'proposed/round_020.pt',engine.device)
    q=Queue(requests,True,0);ids=[r['id'] for r in q.pending(0)]
    status(stage='fixed_training_update_preflight')
    l,u,local=ORIGINAL_VECTORS(engine,theta,plan,q,0,0,'joint',ids,core.Teacher(engine,proxy_theta),{},True)
    np.savez_compressed(OUT/'fixed_training_updates.npz',FL=l,FU=u,channels=channels)
    write('preflight.json',dict(**checks.fixed_update(l,u,channels,plan['config']),local=local))
    ref_theta=core.load_theta(core.SOURCE/'retrain/round_200.pt',engine.device)
    ref_eval,ref_probs=engine.evaluate(ref_theta,None);ref_mia,_=engine.mia_scores(ref_theta)
    threshold=.95*ref_eval['retained']['mR']
    old_ref=json.loads((SOURCE/'reference_metrics.json').read_text())
    assert abs(threshold-old_ref['retained_mR_threshold'])<1e-7
    write('reference_metrics.json',dict(metrics=ref_eval,retained_mR_threshold=threshold,reporting_only=True));write('reference_mia.json',ref_mia)
    stream.OUT=OUT;stream.ARMS=ARMS;stream.vectors=vectors
    stream.radio=SimpleNamespace(analog=communicate)
    stream.report()
    for name,opt in ARMS.items():
        CURRENT=name
        stream.run_arm(engine,plan,requests,proxy_theta,ref_probs,threshold,channels,name,opt,prefix)
        path=OUT/name/'manifest.json';manifest=json.loads(path.read_text())
        if manifest['status']=='completed_200FL_and_requests':
            e=manifest['evaluation']
            manifest['final_joint_quality_pass']=bool(manifest['requests']['procedure_completed']==45 and e['retained']['mR']>=threshold and e['forget']['KL_retrain_to_candidate']<=.01 and e['worst_request_KL']<=.01)
            manifest['confirmation_note']='Final quality pass is distinct from three consecutive eligible evaluations. No extra training or duplicate evaluations are added to manufacture confirmation.'
            core.write(path,manifest);stream.report()
    prepare();stream.OUT=OUT;stream.report()
    write('finished.json',dict(unix=time.time(),status='all_registered_IQ_arms_finished_or_reported_incomplete'))
    status(stage='completed')

if __name__=='__main__':
    try:main()
    except Exception as e:OUT.mkdir(parents=True,exist_ok=True);status(stage='failed',error=repr(e));traceback.print_exc();raise
