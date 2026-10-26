"""Retrain OFDMA replacements with the unchanged proxy-objective protocol.

Legacy internal IDs are retained for the frozen training API only. The protocol
and display labels identify the actual OFDMA implementation, never TDMA.
"""
from pathlib import Path
import sys,os,json,time,fcntl,importlib.util,traceback
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import radio_ofdma as radio
import numpy as np
ROOT=radio.ROOT
spec=importlib.util.spec_from_file_location('ofdma_proxy_core',ROOT/'tools/overnight_energy_v2/proxy_objective_v1/run.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
OUT=ROOT/'results/overnight_energy_v2/ofdma_v1/learning'
PRIOR=ROOT/'results/overnight_energy_v2/proxy_objective_v2'
core.OUT=OUT;core.METHODS=['aircomp_oma','oma_only']
core.LABELS={**core.LABELS,'aircomp_oma':'AirComp+OFDMA','oma_only':'OFDMA-only'}
original_prefix=core.prefix;original_report=core.report


def transport(l,u,h,c,method,seed):
    return radio.communicate(l,u,h,dict(c,ofdma_subcarriers=128),
        {'aircomp_oma':'aircomp_ofdma','oma_only':'ofdma_only'}.get(method,method),seed)


def prefix(method):
    if method!='oma_only':return original_prefix(method)
    src=OUT/'ofdma_initial20';return src,json.loads((src/'prefix_rounds.json').read_text())


def report(plan):
    original_report(plan)
    p=OUT/'summary.json';j=json.loads(p.read_text())
    j.update(version='ofdma_v1',main_comparison='OFDMA replacements; unchanged proxy objective; original NOMA/AirComp results retained as paired controls.',
        scope='Single seed; same fixed 64d deadline for energy/convergence comparison. These times are not minimum completion delays. TX-only energy and uplink-only delay; full local compute latency unmeasured.',
        paired_controls=str(PRIOR/'summary.json'),
        internal_ids={'aircomp_oma':'aircomp_ofdma','oma_only':'ofdma_only'})
    for row in j['methods']:
        row['physical_method']=j['internal_ids'][row['method']]
        hits=[];row['all_request_joint_target']=None
        for f in sorted((OUT/row['method']/'rounds').glob('*.json')):
            r=json.loads(f.read_text());e=r.get('evaluation')
            if not e:continue
            worst=max(x['KL_retrain_to_candidate'] for x in e['per_request'].values() if x['images'])
            ok=r['round']>=25 and e['retained']['mR']>=j['retained_mR_threshold'] and worst<=.01
            hits=hits+[r] if ok else []
            if len(hits)>=3:
                row['all_request_joint_target']=dict(entry_round=hits[0]['round'],confirmation_round=r['round'],
                    energy_bound_j=r['cumulative_tx_energy_bound_j'],uplink_s=r['cumulative_uplink_s'],worst_request_KL=worst);break
    core.write(p,j)


def initial20(engine,plan,records,channels):
    from train import update_theta
    dest=OUT/'ofdma_initial20';dest.mkdir(exist_ok=True)
    assert not (dest/'prefix_rounds.json').exists(),'Explicit recovery required'
    theta=engine.theta0.clone();rows=[];energy=0.;elapsed=0.
    for k in range(1,21):
        core.status(stage='ofdma_initial_learning',round=k);engine.plan['learning_rate']=plan['learning_rate']
        l,u,_=engine.updates(theta,records[k-1],k,'retrain',[[] for _ in range(10)])
        v,r=transport(l,u,channels,plan['config'],'oma_only',100000+k)
        if v is None:raise RuntimeError('OFDMA initial learning has no feasible design at round '+str(k))
        energy+=r['energy_bound_j'];elapsed+=r['uplink_s'];theta=update_theta(theta,v)
        rows.append(dict(round=k,radio=r,cumulative_tx_energy_bound_j=energy,
                         cumulative_tx_energy_expected_j=energy,cumulative_uplink_s=elapsed))
        core.save_theta(dest/f'round_{k:03d}.pt',theta,k)
    core.write(dest/'prefix_rounds.json',rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    claim=(OUT/'launch.lock').open('a');fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'started.json').exists(),'Refuse duplicate launch'
    for mode in ['aircomp_ofdma','ofdma_only']:
        check=json.loads((OUT.parent/f'design_64_{mode}.json').read_text())
        assert check['status']=='feasible' and check['access']=='integer_subcarrier_OFDMA',mode
    assert json.loads((OUT.parent/'cpu_checks.json').read_text())['passed']
    lock=(ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import subprocess
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    plan=json.loads((core.SOURCE/'plan.json').read_text())
    for name,digest in plan['sources'].items():assert core.sha(ROOT/name)==digest,name
    for name,digest in plan['frozen_engine_hashes'].items():assert core.sha(core.FROZEN/name)==digest,name
    for name,digest in plan['runner_hashes'].items():assert core.sha(core.LONG/name)==digest,name
    core.write(OUT/'started.json',dict(pid=os.getpid(),unix=time.time()))
    core.write(OUT/'protocol.json',dict(version='ofdma_v1',seed=0,rounds=200,evaluation_every=5,
      source=str(PRIOR),learning_objective_unchanged=True,request_projection_used=False,
      internal_api_ids={'aircomp_oma':'AirComp+OFDMA','oma_only':'OFDMA-only'},
      scope='Retrain only changed physical baselines. Reuse exact matching AirComp-only prefix/proxy; recompute OFDMA-only prefix and proxy with all costs charged. Old NOMA/AirComp 200-round results remain paired controls.',
      timing='Fixed common 64d energy/convergence comparison. Separate CPU deadline scan estimates feasible timing grid; no minimum-delay or local-compute claims.',
      frozen_plan_sha256=core.sha(core.SOURCE/'plan.json'),
      new_sources={p.name:core.sha(p) for p in HERE.glob('*.py')}))
    from train import Engine
    from dataset import RetrievalData
    core.status(stage='loading_model');engine=Engine(RetrievalData(),dict(plan))
    records=json.loads((core.SOURCE/'records.json').read_text());forget=records[20]['forget']
    channels=np.load(core.SOURCE/'channels.npy')
    analog=core.load_theta(PRIOR/'proxies/analog/round_020.pt',engine.device)
    ac=json.loads((PRIOR/'proxies/analog/complete.json').read_text())
    assert ac['status']=='completed'
    proxies={'analog':(analog,ac)}
    core.prefix=prefix;core.report=report
    # First provide the directly requested AirComp+OFDMA comparison.
    core.physical(engine,plan,records,forget,proxies,transport,channels,'aircomp_oma',40)
    initial20(engine,plan,records,channels)
    schedules=core.proxy_schedule(records,set(plan['forgotten_images']))
    core.write(OUT/'proxy_records.json',schedules)
    proxies['oma']=core.train_proxy(engine,schedules,transport,channels,plan['config'],'oma')
    core.physical(engine,plan,records,forget,proxies,transport,channels,'oma_only',40)
    for method in core.METHODS:core.physical(engine,plan,records,forget,proxies,transport,channels,method,200)
    core.write(OUT/'finished.json',dict(unix=time.time()));core.status(stage='completed')


if __name__=='__main__':
    try:main()
    except Exception as e:core.status(stage='failed',error=repr(e));traceback.print_exc();raise
    finally:radio.close()
