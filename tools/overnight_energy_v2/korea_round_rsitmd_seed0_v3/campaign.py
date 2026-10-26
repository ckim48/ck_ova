"""Authorized repair: remaining three RSITMD seed0 arms, same round-arrival pilot."""
from pathlib import Path
import argparse, copy, fcntl, hashlib, importlib.util, json, os, subprocess, sys, time, traceback
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='1'
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/'results/overnight_energy_v2/korea_round_rsitmd_seed0_v3'
PILOT=ROOT/'results/overnight_energy_v2/korea_joint_seed0_pilot_v2'
OLD=ROOT/'results/overnight_energy_v2/korea_mission_v1'
sys.path.insert(0,str(HERE))
from comparison_schedule import SerialRoundQueue, replay
spec=importlib.util.spec_from_file_location('completed_pilot',HERE.parent/'korea_joint_seed0_pilot_v2/campaign.py')
pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)
old,base,core,clip,repeat=pilot.old,pilot.base,pilot.core,pilot.clip,pilot.repeat
import numpy as np
METHODS=['separate_aircomp_periodic','aircomp_ofdma_periodic','sequential_digital_periodic']
write=core.write
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def state(**kw): old.state(OUT,**kw)

def adapted_arm():
    return repeat.clone_function(old.run_arm,[
        ("q=clip.stream.Queue(requests,opt['parallel'],opt['period'])","q=SerialRoundQueue(requests,opt['parallel'],opt['period'])",1),
        ("site,endwindow=mission.window_at(t,geo)\n  if site is None:horizon_before(endwindow);t=endwindow;continue\n  kind,ids=mission.decide(q,t,fl,geo)",
         "site=min(fl//45,3);endwindow=geo['stops'][site]['end_s']\n  t=max(t,geo['stops'][site]['start_s'])\n  q.sync(fl,t,occupied)\n  kind,ids=q.decision(t,fl)",1),
        ("l,u,local=vecfn(engine,theta,plan,q,t,fl,kind,ids,proxy,tentative,opt['crossmodal'])",
         "l,u,local=vecfn(engine,theta,plan,q,t,fl,kind,ids,proxy,tentative,opt['crossmodal'])\n  assert kind in ['fl','fu']\n  assert np.isfinite(l).all() and np.isfinite(u).all()\n  if kind=='fu':assert np.linalg.norm(l)==0 and np.linalg.norm(u)>0\n  if kind=='fl':assert np.linalg.norm(u)==0",1),
        ("FL_gradient_calls=10 if kind in ['fl','joint'] else 0,new_teacher_requests=",
         "FL_vector_norm=float(np.linalg.norm(l)),FU_vector_norm=float(np.linalg.norm(u)),\n    FL_gradient_calls=10 if kind in ['fl','joint'] else 0,new_teacher_requests=",1),
        ("audit.update(status='transmitted',transaction=seq)",
         "audit.update(status='transmitted',transaction=seq,FL_receiver_apply_s=end if kind=='fl' else None,FU_receiver_apply_s=end if ids else None,FL_round_after=20+fl)",1),
        ("if end>endwindow+1e-9:","if end>endwindow+1e-9:\n   raise RuntimeError('Actual radio duration exceeds the fixed per-site window; no retiming or auto-retry')",1),
        ("mission_clock_s=at,","mission_clock_s=at,clock_policy='round_triggered_comparison',",1)
    ],{'SerialRoundQueue':SerialRoundQueue})

def register():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'protocol.json').exists(),'Already registered; verify rather than overwrite'
    inherited=pilot.verify()
    proposed=PILOT/'rsitmd_seed0'
    assert read(proposed/'finished.json')['status']=='proposed_seed0_pilot_completed'
    plan=read(proposed/'plan.json');requests=read(proposed/'requests.json');geo=read(OLD/'mission_spec.json')
    requested={i for r in requests for i in r['images']}
    assert len(plan['gallery_images'])==814 and requested<=set(plan['gallery_images'])
    assert len(requests)==45 and len(requested)==558 and set(requested)==set(plan['forgotten_images'])
    assert sorted(set(r['release_before_total_FL_round'] for r in requests))==[26,42,59,75,96,120,141,165,186]
    assert all(r['arrival_s'] is None and r['deadline_s'] is None for r in requests)
    write(OUT/'rsitmd_seed0/plan.json',plan);write(OUT/'rsitmd_seed0/requests.json',requests)
    checks={k:replay(requests,geo,d) for k,d in [('analog',dict(fl=.02048,fu=.02048)),('slow',dict(fl=.6,fu=.4)),('stress',dict(fl=1.,fu=.4))]}
    assert checks['analog']['counts']==dict(joint=0,fl=180,fu=45)
    adapted_arm()
    hashes=dict(inherited['source_hashes'])
    for p in [PILOT/'protocol.json',proposed/'plan.json',proposed/'requests.json',proposed/'gallery_check.json',proposed/'reference_metrics.json',proposed/'reference_mia.json',proposed/'common_prefix_cost.json',proposed/'reused_checkpoints.json',proposed/'finished.json']:
        hashes[str(p.relative_to(ROOT))]=sha(p)
    for p in (proposed/'proposed_fu_guard_cliperase').rglob('*'):
        if p.is_file() and p.suffix in ['.json','.pt']:hashes[str(p.relative_to(ROOT))]=sha(p)
    for p in HERE.glob('*.py'):hashes[str(p.relative_to(ROOT))]=sha(p)
    for p in (OUT/'rsitmd_seed0').glob('*.json'):hashes[str(p.relative_to(ROOT))]=sha(p)
    for method in METHODS:
        assert base.ARMS[method]['parallel'] is False and base.ARMS[method]['period']==5
    write(OUT/'cpu_preflight.json',checks)
    write(OUT/'protocol.json',dict(version=HERE.name,registered_unix=time.time(),dataset='RSITMD',seed=0,methods={m:base.ARMS[m] for m in METHODS},
        authorization='User asked to fix all remaining methods and show RSITMD results after the completed proposed seed0 pilot. Exactly the three remaining methods at seed0.',
        config=inherited['config'],source_hashes=hashes,reuse_proposed=str((proposed/'proposed_fu_guard_cliperase').relative_to(ROOT)),
        common='Same initial20/proxy20/reference200,814-image evaluation gallery,request samples,round releases,total200FL,45requests*5steps,4map channels and45FL/site as completed proposed pilot.',
        scheduler='Round arrivals copied byte-for-byte from completed pilot. Serial arms keep period5 communication frames=.1024s on mission-service clock, persistent FU batch pauses FL; no joint transmissions. No old30s deadline comparison.',
        unchanged='Original per-method loss,solver,digitalQ6/OFDMA128tones/1MHz,0.10J TXcap,0.32W/user,seed0,learning rate. Full-system comparison, not isolated radio causality.',
        timing_energy='TX upper bound and pure uplink exclude uncalibrated computation and other hardware costs. Synthetic mission clock is not measured end-to-end time.',
        preservation='Old sources/results/queues remain immutable; no proposed rerun, no extra seed or dataset, no tuning or automatic retry. Per-arm failures preserved and independent next arm continues.',
        provenance='Actual Korean OSM coordinates, synthetic mission and existingRSITMD replay; ©OpenStreetMap contributors/ODbL.'))
    write(OUT/'queue_registered.json',dict(unix=time.time(),methods=METHODS,status='registered_remaining_three_RSITMD_seed0'))
    print('REGISTERED_THREE_RSITMD_ARMS_CPU_PASSED',flush=True)

def verify():
    p=read(OUT/'protocol.json')
    for name,h in p['source_hashes'].items():assert sha(ROOT/name)==h,name
    return p

def run():
    protocol=verify();dest=OUT/'rsitmd_seed0'
    own=(OUT/'launch.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'started.json').exists(),'No duplicate or automatic retry'
    write(OUT/'started.json',dict(unix=time.time(),identity=base.identity(os.getpid())))
    gpu=(ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a');state(stage='waiting_for_GPU_lock')
    fcntl.flock(gpu,fcntl.LOCK_EX)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied without shared lock'
    verify();state(stage='loading_model')
    proposed=PILOT/'rsitmd_seed0';plan=read(dest/'plan.json');requests=read(dest/'requests.json');geo=read(OLD/'mission_spec.json');channels=np.load(OLD/'channels_by_site.npy')
    coverage=read(proposed/'gallery_check.json');files={k:ROOT/v for k,v in coverage['checkpoint_paths'].items()}
    write(dest/'reused_checkpoints.json',dict(files={str(p.relative_to(ROOT)):sha(p) for p in files.values()}))
    write(dest/'gallery_check.json',coverage)
    import train
    from dataset import RetrievalData
    engine=train.Engine(RetrievalData(),dict(plan))
    warm,proxy,ref=[core.load_theta(files[k],engine.device) for k in ['initial20','proxy20','reference200']]
    refmetrics,refprob=engine.evaluate(ref,None);refmia,_=engine.mia_scores(ref);threshold=.95*refmetrics['retained']['mR']
    write(dest/'reference_metrics.json',dict(metrics=refmetrics,retained_mR_threshold=threshold,reporting_only=True));write(dest/'reference_mia.json',refmia)
    prefix=read(files['prefix_cost']);write(dest/'common_prefix_cost.json',prefix)
    fullq=SerialRoundQueue(requests);fullq.sync(180,0.,0.)
    initial=clip.stream.evaluation(engine,warm,refprob,fullq,0.)
    assert len(initial['per_request'])==45
    write(dest/'initial_evaluation_preflight.json',dict(passed=True,evaluation=initial,gallery_sha256=coverage['repaired_gallery_sha256']))
    # Real initial-state vector checks before any of the three ranked runs.
    state(stage='actual_initial_update_preflight')
    q=pilot.RoundQueue(requests);first=min(r['release_after_adaptation_FL'] for r in requests);q.sync(first,0.,0.);kind,ids=q.decision(0.,first)
    l,u,meta=clip.vectors(engine,warm,plan,q,0.,first,kind,ids,core.Teacher(engine,proxy),{},True)
    l2,old_u,old_meta=clip.ORIGINAL_VECTORS(engine,warm,plan,q,0.,first,kind,ids,core.Teacher(engine,proxy),{},True)
    assert np.allclose(l,l2,rtol=1e-6,atol=1e-8)
    prev=base.OUT;base.OUT=dest
    try:
        for site,ch in enumerate(channels):base.physical_checks(l,u,old_u,ch,plan['config'],f'actual_site{site}_preflight.json')
    finally:base.OUT=prev
    write(dest/'actual_preflight_work.json',dict(cliperase=meta,safe_proxy=old_meta,scope='Validation work, separate from experimental trajectory and costs.'))
    done=[];failures=[]
    for method in METHODS:
        verify();state(stage='running',method=method,finished=done,failures=failures);started=time.time();arm=dest/method
        try:
            adapted_arm()(engine,plan,requests,geo,channels,method,base.ARMS[method],prefix,warm,proxy,refprob,threshold,dest,0)
            manifest=read(arm/'manifest.json')
            assert manifest['status']=='completed_200FL_and_requests',manifest['status']
            tx=[read(p) for p in sorted((arm/'transactions').glob('*.json'))]
            counts={k:sum(r['kind']==k for r in tx) for k in ['joint','fl','fu']}
            assert counts['joint']==0 and counts['fl']==180
            records=read(arm/'request_records.json');assert len(records)==45 and all(r['steps']==5 for r in records)
            assert sum(len(t['FU_request_ids']) for t in tx)==225
            write(arm/'serial_transmission_verification.json',dict(passed=True,counts=counts,requests=45,steps_per_request=5,FL_pauses_for_FU=True))
        except Exception as exc:
            failures.append(dict(method=method,error=repr(exc)))
            write(arm/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),unix=time.time()))
        write(arm/'runtime_seconds.json',dict(started_unix=started,finished_unix=time.time(),elapsed_server_wall_seconds=time.time()-started,scope='Simulator runtime, not UAV calibrated time'))
        done.append(method)
    write(dest/'finished.json',dict(status='three_registered_arms_terminal',methods=done,failures=failures,unix=time.time()))
    write(OUT/'finished.json',dict(status='three_registered_arms_terminal',methods=done,failures=failures,unix=time.time()))
    state(stage='completed',finished=done,failures=failures)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--register',action='store_true');parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    try:
        if args.register:register()
        elif args.verify:verify();print('REGISTERED_SOURCES_MATCH',flush=True)
        else:run()
    except Exception as exc:
        write(OUT/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),unix=time.time()));state(stage='failed',error=repr(exc));raise
    finally:base.parent.common.base.close()
