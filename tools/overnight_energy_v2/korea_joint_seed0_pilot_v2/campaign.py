"""Two authorized proposed-only seed0 pilots; frozen algorithms, new scheduler."""
from pathlib import Path
import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'results/overnight_energy_v2/korea_joint_seed0_pilot_v2'
OLD = ROOT / 'results/overnight_energy_v2/korea_mission_v1'
sys.path.insert(0, str(HERE))
from round_schedule import RoundQueue, prepare_requests, replay
from gallery import prepare_plan

spec = importlib.util.spec_from_file_location('frozen_mission', ROOT/'tools/overnight_energy_v2/korea_mission_v1/campaign.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
base, core, clip, repeat = old.base, old.core, old.clip, old.repeat
import numpy as np
METHOD = 'proposed_fu_guard_cliperase'
write = core.write


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def adapted_arm():
    # Exact-count substitutions fail closed if the frozen implementation changes.
    return repeat.clone_function(old.run_arm, [
        ("q=clip.stream.Queue(requests,opt['parallel'],opt['period'])", 'q=RoundQueue(requests)', 1),
        ("site,endwindow=mission.window_at(t,geo)\n  if site is None:horizon_before(endwindow);t=endwindow;continue\n  kind,ids=mission.decide(q,t,fl,geo)",
         "site=min(fl//45,3);endwindow=geo['stops'][site]['end_s']\n  t=max(t,geo['stops'][site]['start_s'])\n  q.sync(fl,t,occupied)\n  kind,ids=q.decision(t,fl)", 1),
        ("l,u,local=vecfn(engine,theta,plan,q,t,fl,kind,ids,proxy,tentative,opt['crossmodal'])",
         "l,u,local=vecfn(engine,theta,plan,q,t,fl,kind,ids,proxy,tentative,opt['crossmodal'])\n  assert kind in ['fl','joint'] and fl<180\n  assert np.isfinite(l).all() and np.isfinite(u).all()\n  if kind=='joint':assert np.linalg.norm(l)>0 and np.linalg.norm(u)>0, 'Zero component in claimed joint transmission'", 1),
        ("FL_gradient_calls=10 if kind in ['fl','joint'] else 0,new_teacher_requests=",
         "FL_vector_norm=float(np.linalg.norm(l)),FU_vector_norm=float(np.linalg.norm(u)),shared_model_before_FL_FU=True,\n    FL_gradient_calls=10 if kind in ['fl','joint'] else 0,new_teacher_requests=", 1),
        ("audit.update(status='transmitted',transaction=seq)",
         "audit.update(status='transmitted',transaction=seq,FL_receiver_apply_s=end,FU_receiver_apply_s=end if ids else None,single_aggregate_apply=True)", 1),
        ("mission_clock_s=at,", "mission_clock_s=at,clock_policy='round_triggered_pilot',", 1),
    ], {'RoundQueue': RoundQueue})


def register():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'protocol.json').exists(), 'Already registered; verify instead of overwrite'
    preparation = json.loads((OUT/'preparation.json').read_text())
    old_protocol = old.verify()
    geo = json.loads((OLD/'mission_spec.json').read_text())
    checks = {}
    hashes = dict(old_protocol['source_hashes'])
    hashes[str(ROOT.joinpath('tools/overnight_energy_v2/korea_mission_v1/campaign.py').relative_to(ROOT))] = sha(ROOT/'tools/overnight_energy_v2/korea_mission_v1/campaign.py')
    for dataset in ('RSITMD', 'RSICD'):
        prep = OUT/'preparation'/dataset.lower()
        req = prepare_requests(json.loads((prep/'original_requests.json').read_text()), geo)
        dest = OUT/f'{dataset.lower()}_seed0'
        write(dest/'registered_requests.json', req)
        checks[dataset] = replay(req, geo)
        for name, expected in preparation['datasets'][dataset]['source_files'].items():
            assert sha(ROOT/name) == expected, name
            hashes[name] = expected
        for p in prep.glob('*.json'):
            hashes[str(p.relative_to(ROOT))] = sha(p)
        hashes[str((dest/'registered_requests.json').relative_to(ROOT))] = sha(dest/'registered_requests.json')
    adapted_arm()  # Validate all frozen function substitutions before queueing.
    for p in HERE.glob('*.py'):
        hashes[str(p.relative_to(ROOT))] = sha(p)
    for p in (OLD/'mission_spec.json', OLD/'channels_by_site.npy', OUT/'preparation.json'):
        hashes[str(p.relative_to(ROOT))] = sha(p)
    write(OUT/'cpu_preflight.json', checks)
    write(OUT/'protocol.json', dict(version=HERE.name, registered_unix=time.time(),
        runs=[dict(dataset=d, seed=0, method=METHOD) for d in ('RSITMD','RSICD')],
        authorization='User explicitly requested only our method, one seed per dataset, to compare with previous results. Seed0 fixed before results.',
        dependency=dict(folder='korea_mission_v1', pid=256450, start_ticks='884912396', argv_suffix='korea_mission_v1/campaign_queue.py'),
        config=old_protocol['config'], method=base.ARMS[METHOD], source_hashes=hashes,
        scheduler='Initial20 reused + continuous180 FL updates. 45 updates at each unchanged map site. FU samples/bursts unchanged. Release at the FL ordinal immediately after each old mission arrival, counted from the old release calendar. No FL time-arrival gate. Every pending FU is included with FL; same pre-update theta; one reconstructed sum applied at common reception end.',
        request_total_FL_rounds=sorted(set(r['release_before_total_FL_round'] for r in req)),
        timing='Fixed old route and hover windows retained. FL rounds within a site run continuously in the uplink-only clock. FU arrivals are now round-triggered. Old30s deadline is not evaluated or compared. Local computation/recovery/control/downlink absent; not measured end-to-end time.',
        reuse='Initial20,proxy20,reference200 checkpoints and prefix costs reused with hashes. Reference evaluation only. Same objective, learning rates, physical solver and seed0 random streams.',
        comparison='Exploratory proposed-only pilot versus saved seed0 results. Scheduler differs, so historical baseline advantage is not a controlled causal comparison. RSITMD old mission failed; no invented old final score. All evaluations use same repaired gallery for the given dataset.',
        energy='Prefix plus adaptation TX upper bound only. Compute-inclusive energy calibration remains unavailable; no invented watts/zero accounting.',
        provenance='Actual OSM coordinates; synthetic mission and RSITMD/RSICD replay. Copyright OpenStreetMap contributors, ODbL. No real flight claim.'))
    write(OUT/'queue_registered.json', dict(unix=time.time(), status='registered_two_proposed_seed0_pilots', after='korea_mission_v1'))
    print('PILOT_REGISTERED_CPU_PASSED', flush=True)


def verify():
    p = json.loads((OUT/'protocol.json').read_text())
    for f,h in p['source_hashes'].items():
        assert sha(ROOT/f) == h, f
    return p


def run(dataset):
    protocol = verify()
    dest = OUT/f'{dataset.lower()}_seed0'
    own = (dest/'launch.lock').open('a')
    fcntl.flock(own, fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (dest/'started.json').exists(), 'No duplicate pilot'
    gpu = (ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a')
    old.state(dest, stage='waiting_for_GPU_lock')
    fcntl.flock(gpu, fcntl.LOCK_EX)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    verify()
    prep = OUT/'preparation'/dataset.lower()
    coverage = json.loads((prep/'coverage_check.json').read_text())
    plan = json.loads((prep/'plan.json').read_text())
    plan.update(seed=0, config=protocol['config'])
    req = json.loads((dest/'registered_requests.json').read_text())
    geo = json.loads((OLD/'mission_spec.json').read_text())
    channels = np.load(OLD/'channels_by_site.npy')
    files = {k:ROOT/v for k,v in coverage['checkpoint_paths'].items()}
    write(dest/'started.json', dict(unix=time.time(),identity=base.identity(os.getpid())))
    write(dest/'plan.json', plan)
    write(dest/'requests.json', req)
    write(dest/'gallery_check.json', coverage)
    write(dest/'reused_checkpoints.json', dict(files={str(p.relative_to(ROOT)):sha(p) for p in files.values()}))
    import train
    from dataset import RetrievalData as RSITMDData
    engine = train.Engine((RSITMDData if dataset=='RSITMD' else base.RetrievalData)(), dict(plan))
    warm, proxy, ref = [core.load_theta(files[k],engine.device) for k in ('initial20','proxy20','reference200')]
    refmetrics,refprob = engine.evaluate(ref,None)
    refmia,_ = engine.mia_scores(ref)
    threshold = .95*refmetrics['retained']['mR']
    write(dest/'reference_mia.json',refmia)
    write(dest/'reference_metrics.json',dict(metrics=refmetrics,retained_mR_threshold=threshold,reporting_only=True))
    prefix = json.loads(files['prefix_cost'].read_text())
    write(dest/'common_prefix_cost.json',prefix)
    # Evaluate every requested image before any pilot learning: catches the old KeyError.
    fullq = RoundQueue(req)
    fullq.sync(180,0.,0.)
    initial = clip.stream.evaluation(engine,warm,refprob,fullq,0.)
    assert len(initial['per_request']) == 45
    write(dest/'initial_evaluation_preflight.json',dict(passed=True,evaluation=initial,gallery_sha256=coverage['repaired_gallery_sha256']))
    # One fixed initial-state real joint vector, checked at all unchanged sites.
    q = RoundQueue(req)
    first = min(r['release_after_adaptation_FL'] for r in req)
    q.sync(first,0.,0.)
    kind,ids = q.decision(0.,first)
    l,u,meta = clip.vectors(engine,warm,plan,q,0.,first,kind,ids,core.Teacher(engine,proxy),{},True)
    assert kind=='joint' and np.linalg.norm(l)>0 and np.linalg.norm(u)>0
    physical = []
    for site,ch in enumerate(channels):
        v,m = base.parent.guard.analog(l,u,ch,plan['config'],500000)
        assert v is not None and np.isfinite(v).all()
        assert m['conditional_rmse']<=m['epsilon']*(1+3e-5)+1e-9
        assert m['energy_bound_j']<=.1*(1+2e-6)
        assert m['max_power_w']<=plan['config']['power_limit_w']*(1+2e-6)
        physical.append(dict(site=site,radio=m))
    write(dest/'actual_joint_preflight.json',dict(passed=True,FL_norm=float(np.linalg.norm(l)),FU_norm=float(np.linalg.norm(u)),sites=physical,work=meta,scope='Validation work recorded separately from experimental learning and prefix.'))
    started = time.time()
    adapted_arm()(engine,plan,req,geo,channels,METHOD,base.ARMS[METHOD],prefix,warm,proxy,refprob,threshold,dest,0)
    arm = dest/METHOD
    write(arm/'runtime_seconds.json',dict(started_unix=started,finished_unix=time.time(),elapsed_server_wall_seconds=time.time()-started,scope='Simulator wall time, not UAV computation calibration.'))
    manifest = json.loads((arm/'manifest.json').read_text())
    assert manifest['status']=='completed_200FL_and_requests',manifest['status']
    transactions = [json.loads(p.read_text()) for p in sorted((arm/'transactions').glob('*.json'))]
    counts = {k:sum(r['kind']==k for r in transactions) for k in ('joint','fl','fu')}
    assert counts==dict(joint=45,fl=135,fu=0),counts
    assert len(transactions)==180 and manifest['completed_FL_rounds']==200
    write(dest/'joint_transmission_verification.json',dict(passed=True,counts=counts,receiver='One aggregate applied once at the same end for both FL and FU. Nonzero component norms stored per attempt.'))
    # Saved models are only evaluated, never retrained or used as teachers.
    comparisons = []
    prior_base = ROOT/'results/overnight_energy_v2'/('fourway_cap100_v1' if dataset=='RSITMD' else 'rsicd_cap100_v1')
    for label,path in [('previous_static_proposed',prior_base/METHOD/'final.pt'),('previous_mission_proposed',OLD/f'{dataset.lower()}_seed0'/METHOD/'final.pt')]:
        if not path.exists():
            comparisons.append(dict(label=label,status='missing_failed_prior_run',path=str(path.relative_to(ROOT))))
            continue
        theta = core.load_theta(path,engine.device)
        ev = clip.stream.evaluation(engine,theta,refprob,fullq,0.)
        mia,_ = engine.mia_scores(theta)
        comparisons.append(dict(label=label,status='evaluated_same_gallery',path=str(path.relative_to(ROOT)),sha256=sha(path),evaluation=ev,mia=mia))
    write(dest/'historical_same_gallery_evaluations.json',dict(gallery_sha256=coverage['repaired_gallery_sha256'],candidates=comparisons,note='Historical training schedules differ. Per-request service steps in this evaluation queue are placeholders; use original request records for completion counts.'))
    write(dest/'finished.json',dict(status='proposed_seed0_pilot_completed',unix=time.time(),counts=counts))
    old.state(dest,stage='completed',method=METHOD,FL_round=200,counts=counts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--register',action='store_true')
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--dataset',choices=['RSITMD','RSICD'])
    args = parser.parse_args()
    dest = OUT if not args.dataset else OUT/f'{args.dataset.lower()}_seed0'
    try:
        if args.register:register()
        elif args.verify:verify();print('PILOT_VERIFIED',flush=True)
        else:
            assert args.dataset
            run(args.dataset)
    except Exception as exc:
        write(dest/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),unix=time.time()))
        old.state(dest,stage='failed',error=repr(exc))
        raise
    finally:
        base.parent.common.base.close()
