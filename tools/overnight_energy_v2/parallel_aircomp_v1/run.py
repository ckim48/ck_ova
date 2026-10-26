"""User-authorized parallel FL/FU comparison; original campaigns stay paused."""
from pathlib import Path
import sys,os,json,time,fcntl,hashlib,subprocess,importlib.util,traceback
import radio_parallel as radio
import numpy as np
ROOT=radio.ROOT;HERE=Path(__file__).resolve().parent;OUT=ROOT/'results/overnight_energy_v2/parallel_aircomp_v1'
spec=importlib.util.spec_from_file_location('parallel_proxy_core',ROOT/'tools/overnight_energy_v2/proxy_objective_v1/run.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
core.OUT=OUT
MODES={'proposed':'joint','separate_aircomp':'separate','hybrid_no_crossmodal':'joint','aircomp_oma':'ofdma'}
LABELS={'proposed':'Parallel AirComp + cross-modal preservation (candidate)',
    'separate_aircomp':'Separate AirComp + identical cross-modal preservation',
    'hybrid_no_crossmodal':'Parallel AirComp without retained relation KL',
    'aircomp_oma':'AirComp + actual OFDMA + identical cross-modal preservation'}
core.METHODS=list(MODES);core.LABELS=LABELS;core.mode=lambda method:MODES[method]

def identity(pid):
    try:
        p=Path('/proc')/str(pid);s=(p/'stat').read_text().split(') ',1)[1].split()
        return dict(pid=pid,start=s[19],state=s[0],cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())
    except FileNotFoundError:return None

def prefix(method):
    src=core.SOURCE/'proposed';rows=json.loads((src/'prefix_rounds.json').read_text())
    return src,[r for r in rows if r['round']<=20]

def report(plan):
    methods=[]
    for method in core.METHODS:
        d=OUT/method;rows=[json.loads(p.read_text()) for p in sorted((d/'rounds').glob('*.json'))]
        evs=[r for r in rows if 'evaluation' in r];hits=[];target=None
        for row in evs:
            e=row['evaluation'];worst=max(x['KL_retrain_to_candidate'] for x in e['per_request'].values() if x['images'])
            ok=row['round']>=25 and e['retained']['mR']>=.29304036883016427 and e['forget']['KL_retrain_to_candidate']<=.01 and worst<=.01 and row['request_algorithm_completed']
            hits=hits+[row] if ok else []
            if len(hits)>=3 and target is None:target=dict(entry_round=hits[0]['round'],confirmation_round=row['round'],energy_j=row['cumulative_tx_energy_bound_j'],uplink_s=row['cumulative_uplink_s'])
        m=json.loads((d/'manifest.json').read_text()) if (d/'manifest.json').exists() else dict(status='pending_or_running')
        entry=dict(method=method,label=LABELS[method],status=m['status'],completed_round=rows[-1]['round'] if rows else 20,joint_target=target)
        if evs:
            r=evs[-1];e=r['evaluation'];entry['last_evaluation']=dict(round=r['round'],retained=e['retained'],forget=e['forget'],per_request=e['per_request'],worst_request_KL=max(x['KL_retrain_to_candidate'] for x in e['per_request'].values() if x['images']))
        if rows:entry['costs']={k:rows[-1][k] for k in ['cumulative_tx_energy_bound_j','cumulative_uplink_s']}
        ms=sorted(d.glob('mia_*.json'))
        if ms:entry['mia']=json.loads(ms[-1].read_text())
        methods.append(entry)
    alias=dict(methods[0]);alias.update(method='conventional_joint',label='Conventional Joint AirComp + identical VLM objective',exact_alias_of='proposed',independent_run=False)
    methods.append(alias)
    core.write(OUT/'summary.json',dict(methods=methods,
        identification='Proposed and conventional joint AirComp are identical under the current definition. The fifth label is an explicit exact alias, not independent evidence or an inferior baseline.',
        target=dict(retained_mR=.29304036883016427,mean_and_worst_forget_KL=.01,successive_checks=3,check_interval=5,request_steps_complete=True),
        limitations='Single seed0 pilot. Pure uplink delay and TX waveform energy, common inherited prefix/proxy charged. Restricted time/error-allocation grid and local AO, not a global optimum. No quality tuning from final reference or test. No unlearning guarantee from KL or MIA.'))

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    plan=json.loads((core.SOURCE/'plan.json').read_text())
    old=json.loads((ROOT/'results/overnight_energy_v2/adaptive_ofdma_v2/protocol.json').read_text())
    files={p:d for p,d in old['source_hashes'].items()}
    for p in HERE.glob('*.py'):files[str(p.relative_to(ROOT))]=core.sha(p)
    protocol=dict(version='parallel_aircomp_v1',registered_unix=time.time(),seed=0,max_rounds=200,first_stage=40,
        authorization='User explicitly requested adding the conventional joint AirComp control and running this new five-label comparison; earlier unrelated campaigns remain paused.',
        methods=LABELS,alias={'conventional_joint':'proposed'},distinct_trajectories=4,
        identity_limitation='No unique task-weighted or other new transceiver rule has been implemented or claimed. Both joint labels use exactly the same scalar-MSE channel-inversion solver with all ten heads, identical update composition, time grid, power and energy constraints.',
        objective='Minimize pure uplink time on the declared grid subject to network TX energy <=0.05 J per round; tie-break by TX energy. All schemes use 0.32 W per user, 0.5 J per-user limit, 1 MHz and identical channels.',
        count_grid=list(radio.COUNTS),separate_phase_grid='All integer FL/FU splits for each total frame count; optimize split over heads and total count.',
        separate_quality='FL and FU computed from the SAME round-start model; summed once after reception. Exact channel inversion gives zero analog bias. Independent noise variances add across phases. A single convex subproblem jointly allocates their variance and energy; no arbitrary half-error or half-time split.',
        analog_solver='COMMON exact channel-inversion design family for all analog schemes. Set x=L/(W*receiver^2); energy is linear in x, noise MSE=sum d/(2W*x), and power/battery constraints are linear. Solve convex minimum energy within each candidate time then select shortest feasible time across all ten heads. This restricted family is not the unrestricted biased-AO optimum. Conventional Joint uses precisely the same family and remains an exact alias.',
        ofdma='Actual 128-tone integer OFDMA with bandwidth-scaled noise, identical total bandwidth and per-user total power. Digital FU Q6 with scale bits; quantization counted within the raw aggregate RMS budget. No forced quantization for analog schemes.',
        learning='Frozen retained contrastive loss plus bidirectional retained-only-proxy FU KL and retained relation KL. Shared learning rates, same initialized prompts, data batches and request schedule.',
        ablation='Remove ONLY retained relation KL from FU; retain the bidirectional proxy FU objective. This is not a non-VLM architecture and not removal of every cross-modal dependency.',
        requests='Existing reproducible five simultaneous UAV requests, FU steps21..25. No claim of continuous request arrivals throughout200 rounds.',
        prefix='All arms share saved no-request20-round analog model and separately charged20-round retained-only proxy. Prefix/proxy are fixed common setup costs, not recomputed as delay-optimal initialization. Report adaptation-only costs separately where needed.',
        evaluation='All feasible unique arms through40 then200 regardless of comparative quality; unchanged final/matched-horizon reference only for reporting, never tuning or teacher. MIA at stage endpoints. No test selection.',
        scope='Implementation/control study, not proof of novelty or proposed superiority; no local computation parallelism is implied. Radio solver revision chosen before any new learned outcomes: aborted AO preflight was computationally expensive; its log/protocol are preserved. Every analog arm receives the same exact-inversion family, with optimal continuous variance allocation for fixed head and lengths.',source_hashes=files)
    p=OUT/'protocol.json'
    if p.exists():protocol=json.loads(p.read_text())
    else:core.write(p,protocol)
    for name,sha in protocol['source_hashes'].items():assert core.sha(ROOT/name)==sha,name
    for name,sha in plan['frozen_engine_hashes'].items():assert core.sha(core.FROZEN/name)==sha,name
    return plan,protocol

def main():
    plan,protocol=prepare()
    if '--prepare' in sys.argv:print('PREPARED');return
    checks=json.loads((OUT/'preflight.json').read_text());assert checks['passed']
    for name,sha in checks['new_source_hashes'].items():assert core.sha(HERE/name)==sha,name
    own=(OUT/'launch.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'started.json').exists(),'Duplicate launch'
    lock=(ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU still occupied'
    core.write(OUT/'started.json',dict(unix=time.time(),identity=identity(os.getpid())))
    core.status(stage='loading_model',campaign='parallel_aircomp_v1')
    from train import Engine
    from dataset import RetrievalData
    engine=Engine(RetrievalData(),dict(plan));records=json.loads((core.SOURCE/'records.json').read_text());forget=records[20]['forget'];channels=np.load(core.SOURCE/'channels.npy')
    p=ROOT/'results/overnight_energy_v2/proxy_objective_v2/proxies/analog';meta=json.loads((p/'complete.json').read_text());assert meta['status']=='completed'
    proxies={'analog':(core.load_theta(p/'round_020.pt',engine.device),meta)}
    core.prefix=prefix;core.report=report
    for stop in [40,200]:
        for method in core.METHODS:core.physical(engine,plan,records,forget,proxies,radio.communicate,channels,method,stop)
        core.write(OUT/f'stage_{stop}_finished.json',dict(unix=time.time(),settings_unchanged=True))
    prepare();report(plan);core.write(OUT/'finished.json',dict(unix=time.time(),status='completed'));core.status(stage='completed')

if __name__=='__main__':
    try:main()
    except Exception as e:core.status(stage='failed',error=repr(e));traceback.print_exc();raise
    finally:radio.close()
