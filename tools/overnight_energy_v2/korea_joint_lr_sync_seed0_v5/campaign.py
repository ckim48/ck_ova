"""Fixed two-arm diagnostic of synchronized FL/FU annealing; preserve v2.

This is an algorithm/schedule change, not a correction to historical data.
No validation/reference result controls any learning-rate decision.
"""
from pathlib import Path
import argparse, copy, fcntl, hashlib, importlib.util, json, os, subprocess, sys, time, traceback

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'results/overnight_energy_v2/korea_joint_lr_sync_seed0_v5'
PILOT = ROOT / 'tools/overnight_energy_v2/korea_joint_seed0_pilot_v2'
sys.path.insert(0, str(PILOT))
spec = importlib.util.spec_from_file_location('unchanged_joint_pilot', PILOT / 'campaign.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
import numpy as np
base, core, old, repeat = pilot.base, pilot.core, pilot.old, pilot.repeat
METHOD = pilot.METHOD
write, sha = core.write, core.sha


def synchronized_vectors():
    # Use a real source file: the inherited seeded runner inspects and clones
    # this function once more. A synthetic compile filename is insufficient.
    namespace = dict(pilot.clip.vectors.__globals__)
    source = HERE / 'vectors.py'
    exec(compile(source.read_text(), str(source), 'exec'), namespace)
    return namespace['vectors']


def adapted_arm():
    # A private module namespace prevents changing the frozen runner's globals.
    from types import SimpleNamespace
    clip = SimpleNamespace(**pilot.clip.__dict__)
    clip.vectors = synchronized_vectors()
    factory = repeat.clone_function(pilot.adapted_arm, [
        ("mission_clock_s=at,clock_policy='round_triggered_pilot',",
         "mission_clock_s=at,clock_policy='round_triggered_pilot',learning_rate_policy='shared_FL_cosine',", 1),
        ("shared_model_before_FL_FU=True,",
         "shared_model_before_FL_FU=True,FL_learning_rate=core.lr_at(21+fl),FU_learning_rate=core.lr_at(21+fl) if ids else None,", 1),
        ("{'RoundQueue': RoundQueue}", "{'RoundQueue': RoundQueue, 'clip': synchronized_clip}", 1),
    ], {'synchronized_clip': clip})
    return factory()


def register():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'protocol.json').exists(), 'Already registered; do not overwrite'
    previous = pilot.verify()
    vector_fn = synchronized_vectors()
    repeat.clone_function(vector_fn, [('np.random.default_rng(', 'seeded_rng(', 2)],
                          {'seeded_rng': lambda k: repeat.seed_rng(k, 0)})
    adapted_arm()
    geo = json.loads((pilot.OLD / 'mission_spec.json').read_text())
    hashes = dict(previous['source_hashes'])
    checks = {}
    for dataset in ('RSITMD', 'RSICD'):
        source = pilot.OUT / f'{dataset.lower()}_seed0'
        assert (source / 'finished.json').exists()
        requests = json.loads((source / 'registered_requests.json').read_text())
        checks[dataset] = pilot.replay(requests, geo)
        for request in requests:
            first = request['release_before_total_FL_round']
            assert all(core.lr_at(k) > 0 for k in range(first, first + request['required_steps']))
        for path in [source / 'registered_requests.json', source / METHOD / 'manifest.json',
                     source / METHOD / 'final.pt', pilot.OUT / 'preparation' / dataset.lower() / 'coverage_check.json']:
            hashes[str(path.relative_to(ROOT))] = sha(path)
    for path in HERE.glob('*.py'):
        hashes[str(path.relative_to(ROOT))] = sha(path)
    schedule = [dict(round=k,FL_lr=core.lr_at(k),FU_lr=core.lr_at(k),old_FU_lr=.0035) for k in range(21,201)]
    assert all(r['FL_lr'] == r['FU_lr'] for r in schedule)
    assert all(schedule[i+1]['FL_lr'] <= schedule[i]['FL_lr'] for i in range(len(schedule)-1))
    write(OUT / 'cpu_preflight.json', dict(passed=True,scheduler=checks,schedule=schedule,
          all_225_request_steps_have_positive_learning_rate=True,
          no_validation_reference_or_observed_loss_used_for_steps=True))
    write(OUT / 'protocol.json', dict(version=HERE.name,registered_unix=time.time(),
          authorization='2026-10-24 user requested fixing the final loss rise and convergence. Two proposed-only seed0 diagnostics; no sweep or extra seed.',
          runs=[dict(dataset=d,seed=0,method=METHOD) for d in ('RSITMD','RSICD')],
          config=previous['config'],source_hashes=hashes,
          implementation_repair='v4 stopped before any learning transaction: inspect.getsource on generated vector function. v5 uses a real source file; same registered numeric policy.',
          change='FU SGD rate uses core.lr_at(21+fl), identical to simultaneous FL; previously FU was fixed 0.0035. Original objective weights unchanged.',
          status_of_change='Algorithm/schedule diagnostic, not evidence of a historical logging bug. Old fixed FU rate was intentional. Convergence is not guaranteed.',
          no_selection='One predeclared shared-cosine schedule, two fixed datasets/seed0. No tuning, automatic retry or best-checkpoint selection. Report the final round200 and every observation.',
          invariants='Same initial20,180 adaptive FL,45 requests each5FU steps,45joint/135FL-only, arrival rounds26/42/59/75/96/120/141/165/186, four map sites, repaired gallery, losses, seeds, radio solver, and thresholds.',
          reference='Only reporting; never gradients, learning-rate choice, model selection or stopping.',
          required_reporting='Before/after retained loss,mR,mean/worst requestKL,both disagreement directions,MIA/reference gap,all request steps,component norms,TX upper bound,pure uplink. A lower retained loss alone does not establish better unlearning.',
          accounting='No numerical total energy without calibration. Prefix charged per arm. Check/preflight computation counted separately. Pure uplink is not end-to-end latency.',
          provenance='Actual OSM Daejeon coordinates with synthetic mission and existing RSITMD/RSICD images; no measured flight. Copyright OpenStreetMap contributors, ODbL.'))
    write(OUT / 'preparation.json',dict(source=str(pilot.OUT.relative_to(ROOT)),checkpoints='Same SHA-verified initial20/proxy20/reference200 as v2.'))


def verify():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    for name, expected in protocol['source_hashes'].items():
        assert sha(ROOT/name) == expected, name
    return protocol


def run(dataset, protocol):
    dest = OUT / f'{dataset.lower()}_seed0'
    dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'started.json').exists(), 'No repeated arm'
    source = pilot.OUT / f'{dataset.lower()}_seed0'
    prep = pilot.OUT / 'preparation' / dataset.lower()
    coverage = json.loads((prep/'coverage_check.json').read_text())
    plan = json.loads((prep/'plan.json').read_text())
    plan.update(seed=0,config=protocol['config'])
    requests = json.loads((source/'registered_requests.json').read_text())
    geo = json.loads((pilot.OLD/'mission_spec.json').read_text())
    channels = np.load(pilot.OLD/'channels_by_site.npy')
    files = {key:ROOT/value for key,value in coverage['checkpoint_paths'].items()}
    for name,obj in [('started.json',dict(unix=time.time(),identity=base.identity(os.getpid()))),
                     ('plan.json',plan),('requests.json',requests),('gallery_check.json',coverage),
                     ('reused_checkpoints.json',dict(files={str(p.relative_to(ROOT)):sha(p) for p in files.values()}))]:
        write(dest/name,obj)
    import train
    from dataset import RetrievalData as RSITMDData
    engine = train.Engine((RSITMDData if dataset=='RSITMD' else base.RetrievalData)(),dict(plan))
    warm,proxy,ref = [core.load_theta(files[key],engine.device) for key in ('initial20','proxy20','reference200')]
    refmetrics,refprob = engine.evaluate(ref,None)
    refmia,_ = engine.mia_scores(ref)
    threshold = .95*refmetrics['retained']['mR']
    write(dest/'reference_metrics.json',dict(metrics=refmetrics,retained_mR_threshold=threshold,reporting_only=True))
    write(dest/'reference_mia.json',refmia)
    prefix = json.loads(files['prefix_cost'].read_text())
    write(dest/'common_prefix_cost.json',prefix)
    # Numerical parity at a late-round index, using the initial model only.
    # Complete earlier requests in this diagnostic queue; do not alter training.
    q = pilot.RoundQueue(requests)
    q.sync(165,0.,0.)
    for r in q.items.values():
        if r['release_after_adaptation_FL'] < 165:
            r['steps']=r['required_steps'];r['completed_s']=0.
    kind,ids = q.decision(0.,165)
    assert kind=='joint' and len(ids)==5
    proxy_teacher = core.Teacher(engine,proxy)
    l0,u0,_ = pilot.clip.vectors(engine,warm,plan,q,0.,165,kind,ids,proxy_teacher,{},True)
    l1,u1,meta = synchronized_vectors()(engine,warm,plan,q,0.,165,kind,ids,proxy_teacher,{},True)
    factor = core.lr_at(186)/.0035
    assert np.array_equal(l0,l1)
    assert np.allclose(u1,u0*factor,rtol=5e-6,atol=1e-10)
    assert np.linalg.norm(l1)>0 and np.linalg.norm(u1)>0
    write(dest/'actual_update_preflight.json',dict(passed=True,round=186,FL_unchanged=True,
          FU_scale_factor=factor,old_FU_norm=float(np.linalg.norm(u0)),new_FU_norm=float(np.linalg.norm(u1)),
          local=meta,gradient_calls=dict(FL=20,FU=10),teacher_feature_work='Two diagnostic vector builds; separate from training costs.'))
    started=time.time()
    adapted_arm()(engine,plan,requests,geo,channels,METHOD,base.ARMS[METHOD],prefix,warm,proxy,refprob,threshold,dest,0)
    arm=dest/METHOD
    manifest=json.loads((arm/'manifest.json').read_text())
    assert manifest['status']=='completed_200FL_and_requests',manifest['status']
    transactions=[json.loads(p.read_text()) for p in sorted((arm/'transactions').glob('*.json'))]
    counts={k:sum(t['kind']==k for t in transactions) for k in ('joint','fl','fu')}
    assert counts==dict(joint=45,fl=135,fu=0)
    records=json.loads((arm/'request_records.json').read_text())
    assert len(records)==45 and all(r['steps']==5 for r in records)
    assert len(transactions)==180 and manifest['completed_FL_rounds']==200
    for t in transactions:
        for item in t['local']:
            assert item['FU_learning_rate']==item['FL_learning_rate']>0
    write(dest/'joint_transmission_verification.json',dict(passed=True,counts=counts,request_steps=225,
          every_joint_both_nonzero=True,equal_learning_rates=True,same_receiver_apply=True))
    write(arm/'runtime_seconds.json',dict(elapsed_server_wall_seconds=time.time()-started,scope='Simulator wall time only.'))
    write(dest/'finished.json',dict(status='completed',unix=time.time(),counts=counts))
    old.state(dest,stage='completed',FL_round=200,counts=counts)
    del engine
    import gc,torch
    gc.collect();torch.cuda.empty_cache()


def main():
    protocol=verify()
    assert not (OUT/'started.json').exists(), 'No duplicate campaign'
    with (OUT/'campaign.lock').open('a') as own, (ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a') as gpu:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU already occupied'
        write(OUT/'started.json',dict(unix=time.time(),identity=base.identity(os.getpid())))
        results=[]
        for dataset in ('RSITMD','RSICD'):
            verify()
            write(OUT/'status.json',dict(stage='running',dataset=dataset,unix=time.time(),pid=os.getpid()))
            try:
                run(dataset,protocol)
                results.append(dict(dataset=dataset,status='completed'))
            except Exception as exc:
                write(OUT/f'{dataset.lower()}_seed0/failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),unix=time.time()))
                results.append(dict(dataset=dataset,status='failed',error=repr(exc)))
        write(OUT/'finished.json',dict(unix=time.time(),results=results,no_retries=True))
        write(OUT/'status.json',dict(stage='terminal',results=results,unix=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--register',action='store_true');parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    try:
        if args.register:register();print('REGISTERED',flush=True)
        elif args.verify:verify();print('VERIFIED',flush=True)
        else:main()
    finally:
        base.parent.common.base.close()
