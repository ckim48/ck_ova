"""Isolated, GPU-serialized pilot of persistent request memory; no radio ranking."""
from pathlib import Path
import os, sys, json, time, fcntl, traceback, importlib.util, subprocess
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
spec=importlib.util.spec_from_file_location('persistent_proxy_core',ROOT/'tools/overnight_energy_v2/proxy_objective_v1/run.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
sys.path.insert(0,str(HERE))
import controller
OUT=ROOT/'results/overnight_energy_v2/head_persistent_v1'
PRIOR=ROOT/'results/overnight_energy_v2/proxy_objective_v2'
ARMS=['sum','local_memory','head_memory']
STOP=60

def write(p,x):core.write(p,x)
def status(**x):write(OUT/'status.json',dict(unix=time.time(),pid=os.getpid(),**x))
def worst(e):return max(x['KL_retrain_to_candidate'] for x in e['per_request'].values() if x['images'])

def prepare():
    plan=json.loads((core.SOURCE/'plan.json').read_text())
    for name,digest in plan['sources'].items():assert core.sha(ROOT/name)==digest,name
    for name,digest in plan['frozen_engine_hashes'].items():assert core.sha(core.FROZEN/name)==digest,name
    for name,digest in plan['runner_hashes'].items():assert core.sha(core.LONG/name)==digest,name
    paths=[HERE/'run_pilot.py',HERE/'controller.py',ROOT/'tools/overnight_energy_v2/proxy_objective_v1/run.py',
           core.SOURCE/'proposed/round_020.pt',PRIOR/'proxies/ideal/round_020.pt',core.SOURCE/'records.json']
    protocol=dict(version='head_persistent_v1',seed=0,arms=ARMS,stop=STOP,
        common_prefix='same round20 checkpoint; common steps21..25, then three independent continuations26..60',
        setting='Existing RSITMD image-caption requests, all requested samples excluded from subsequent FL. No engineered adverse data or replay of forbidden pairs. This first pilot does not yet introduce new streaming requests.',
        objective='Same retain CE + bidirectional retain-only-proxy forget KL + retained crossmodal relation KL. Move relation contribution from U into L; their unquantized sum is unchanged.',
        messages='L_u=-eta*(alpha*g_retain+beta*g_relation), U_u=-eta*beta*g_forget. Same Q8 canonical U in all arms; raw data and evaluation references are unavailable to head.',
        memory='Normalize each requesting UAV mean decoded pure-FU vector over21..25. Joint visual/text coordinates. Anchor at common post-FU round25, not a claim that deletion has succeeded.',
        controller='min_d 0.5||d-A||^2+0.5||[-B(theta-theta25+d)]_+||^2; lambda=1, fixed before pilot. Exact small dual active-set solve. Soft surrogate, not a deletion or utility guarantee.',
        local_control='Each requesting UAV guards its own weighted FL contribution using its own direction and alpha-weighted global drift. Other UAVs are unmodified. No access to other private directions.',
        strong_aircomp_control='After acquiring identical B and anchor scalars, an AirComp head can run the identical controller on the aggregate. Its quality trajectory is exactly head_memory here; do not perform redundant training or claim intrinsic NOMA advantage. Bootstrap memory/feedback costs belong to subsequent physical comparison.',
        data_access='Forget data read for local FU only during21..25; later algorithm uses derived directions and retained batches. Independent evaluator retains probes for reporting only.',
        metrics='retained loss/mR, mean/worst request KL, matched-horizon KL, prediction disagreements, observed surrogate violations, stalled steps, MIA at60. Same independent references, no test access.',
        gate='At40 and60 worst-request KL at least1% lower than sum, meanKL <=1.001*sum+1e-6, retainedmR >=sum-.01, retainedloss<=sum+.02, and nonzero correction. This is a development allocation gate, not statistical significance.',
        resource_scope='Ideal-link/Q8 algorithm pilot; energy and time ranking unavailable. Paid proxy and prefix provenance retained; no ideal energy-zero win.',
        sources={str(p.relative_to(ROOT)):core.sha(p) for p in paths})
    old=OUT/'pilot_protocol.json'
    if old.exists():assert json.loads(old.read_text())==protocol,'Protocol/source changed'
    write(old,protocol);write(OUT/'controller_checks.json',controller.checks())
    return plan,protocol

def repartitioned_updates(engine,theta,rec,k,forget,teacher):
    lr=engine.plan['learning_rate'];n=len(rec['retained']);dim=len(theta)-1
    l=np.zeros((n,dim));u=np.zeros_like(l);original=np.zeros_like(l);rows=[]
    count=sum(map(len,rec['retained']));nf=sum(map(len,forget))
    for user,ids in enumerate(rec['batches']):
        g,loss=engine.gradient(theta,ids,k);g=g[:-1].detach().cpu().numpy().astype(float)
        l[user]=-lr*len(rec['retained'][user])/count*g;original[user]=l[user]
        row=dict(user=user,retained_loss=loss)
        if forget[user] and 21<=k<=25:
            fids=np.random.default_rng(9000+k*10+user).choice(forget[user],min(16,len(forget[user])),replace=False).astype(int).tolist()
            fg,fl=core.forgetting_gradient(engine,theta,fids,teacher)
            rg,rl=engine.gradient(theta,ids,k,'relation')
            fg=fg[:-1].detach().cpu().numpy().astype(float);rg=rg[:-1].detach().cpu().numpy().astype(float)
            beta=len(forget[user])/nf
            u[user]=-lr*beta*fg;l[user]-=lr*beta*rg
            original[user]-=lr*beta*(fg+rg)
            row.update(forget_proxy_KL=fl,retained_relation_KL=rl)
        rows.append(row)
    err=float(np.linalg.norm(l+u-original)/max(np.linalg.norm(original),1e-30))
    assert err<1e-12
    return l,u,rows,err

def canonical(l,u,c):
    from phase_split_grid import make_inputs
    x=make_inputs(l,u,c)
    return x[2].reshape(len(u),-1)[:,:u.shape[1]]

def evaluation(engine,theta,k,forget):
    e,_=core.evaluate(engine,theta,k,forget)
    return e

def run(plan):
    import torch
    from train import Engine,update_theta
    from dataset import RetrievalData
    engine=Engine(RetrievalData(),dict(plan))
    initial=core.load_theta(core.SOURCE/'proposed/round_020.pt',engine.device)
    engine.set_teacher(initial)
    teacher=core.Teacher(engine,core.load_theta(PRIOR/'proxies/ideal/round_020.pt',engine.device))
    records=json.loads((core.SOURCE/'records.json').read_text());forget=records[20]['forget']
    forbidden=set(i for pool in forget for i in pool)
    for rec in records[20:STOP]:
        assert all(not forbidden.intersection(ids) for ids in rec['retained'])
    theta=initial.clone();acc=np.zeros((10,len(theta)-1));all_results={}
    for k in range(21,26):
        t0=time.time();status(stage='common_fu',round=k)
        engine.plan['learning_rate']=core.lr_at(k)
        l,u,stats,err=repartitioned_updates(engine,theta,records[k-1],k,forget,teacher)
        q=canonical(l,u,plan['config']);acc+=q
        theta=update_theta(theta,(l+q).sum(0))
        row=dict(round=k,local=stats,repartition_relative_error=err,wall_s=time.time()-t0)
        if k==21:np.savez_compressed(OUT/'common_transport_snapshot.npz',learning=l,unlearning=u,canonical_unlearning=q)
        write(OUT/'common'/f'{k:03d}.json',row)
        print(json.dumps(dict(stage='common_fu',round=k,wall_s=row['wall_s'])),flush=True)
    core.save_theta(OUT/'common/round_025.pt',theta,25)
    common=theta.clone();anchor=theta[:-1].detach().cpu().numpy().astype(float)
    basis=controller.unit_rows(acc);local_basis=[controller.unit_rows(x[None]) for x in acc]
    assert len(basis)<=8
    np.savez_compressed(OUT/'memory_state.npz',basis=basis,anchor_scores=basis@anchor,client_directions=acc)
    write(OUT/'memory_metadata.json',dict(requests=len(basis),dimension=len(anchor),
        float32_basis_plus_anchor_bytes=int((basis.size+len(basis))*4),
        full_gradient_history_required=False,deletion_completed_claim=False))
    common_eval=evaluation(engine,common,25,forget)
    write(OUT/'common/evaluation_025.json',common_eval)
    for arm in ARMS:
        theta=common.clone();all_results[arm]={'25':common_eval}
        for k in range(26,STOP+1):
            t0=time.time();status(stage='pilot',arm=arm,round=k,stop=STOP)
            engine.plan['learning_rate']=core.lr_at(k)
            l,u,stats,err=repartitioned_updates(engine,theta,records[k-1],k,forget,teacher)
            assert not np.any(u)
            a=l.sum(0);drift=theta[:-1].detach().cpu().numpy().astype(float)-anchor
            if arm=='sum':d=a;diag=dict(active=0,correction_ratio=0.)
            elif arm=='head_memory':d,diag=controller.protect(a,drift,basis)
            else:
                count=sum(map(len,records[k-1]['retained']));parts=[];diags=[]
                for user,x in enumerate(l):
                    alpha=len(records[k-1]['retained'][user])/count
                    v,info=controller.protect(x,alpha*drift,local_basis[user]);parts.append(v);diags.append(info)
                d=np.sum(parts,0);diag=dict(local=diags,active=sum(x['active'] for x in diags),
                    correction_ratio=float(np.linalg.norm(d-a)/max(np.linalg.norm(a),1e-15)))
            assert np.all(np.isfinite(d))
            theta=update_theta(theta,d)
            row=dict(round=k,arm=arm,controller=diag,stalled=bool(np.linalg.norm(d)<1e-12),local=stats)
            if k%5==0:
                e=evaluation(engine,theta,k,forget);row['evaluation']=e;all_results[arm][str(k)]=e
                core.save_theta(OUT/arm/f'round_{k:03d}.pt',theta,k)
            row['wall_s']=time.time()-t0;write(OUT/arm/'rounds'/f'{k:03d}.json',row)
            print(json.dumps(dict(stage='pilot',arm=arm,round=k,wall_s=row['wall_s'])),flush=True)
        mia,_=engine.mia_scores(theta);write(OUT/arm/'mia.json',mia)
        write(OUT/'interim_results.json',all_results)
    checks={}
    for k in ['40','60']:
        a=all_results['sum'][k];b=all_results['head_memory'][k]
        checks[k]=dict(worst_improved=worst(b)<=.99*worst(a),
            mean_preserved=b['forget']['KL_retrain_to_candidate']<=1.001*a['forget']['KL_retrain_to_candidate']+1e-6,
            retained_mR=b['retained']['mR']>=a['retained']['mR']-.01,
            retained_loss=b['retained']['loss']<=a['retained']['loss']+.02)
    changed=any(json.loads(p.read_text())['controller']['correction_ratio']>1e-6 for p in (OUT/'head_memory/rounds').glob('*.json'))
    result=dict(results=all_results,checks=checks,changed=changed,
        promotion_candidate=changed and all(all(x.values()) for x in checks.values()),
        interpretation='Ideal-link algorithm screen, not NOMA performance or physical energy ranking. Cached-memory AirComp can share the head_memory quality trajectory after paying acquisition costs.')
    write(OUT/'pilot_results.json',result)
    write(OUT/'finished.json',dict(unix=time.time(),promotion_candidate=result['promotion_candidate']))
    status(stage='completed_pilot',promotion_candidate=result['promotion_candidate'])

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plan,protocol=prepare()
    if '--prepare' in sys.argv:print('PREFLIGHT_PASSED');return
    claim=(OUT/'pilot.lock').open('a');fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'started.json').exists(),'Refuse duplicate experiment'
    status(stage='queued_waiting_for_gpu_lock',arms=ARMS,stop=STOP)
    lock=(ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    # Allow prior CUDA teardown; never overlap an uncoordinated active GPU process.
    for _ in range(30):
        gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
        if not gpu.stdout.strip():break
        time.sleep(2)
    else:raise RuntimeError('Another GPU process remains active after lock acquisition')
    for name,digest in protocol['sources'].items():assert core.sha(ROOT/name)==digest,name
    write(OUT/'started.json',dict(unix=time.time(),pid=os.getpid()))
    status(stage='loading_model');run(plan)
    for name,digest in protocol['sources'].items():assert core.sha(ROOT/name)==digest,name

if __name__=='__main__':
    try:main()
    except Exception as exc:
        status(stage='failed',error=repr(exc));traceback.print_exc();raise
