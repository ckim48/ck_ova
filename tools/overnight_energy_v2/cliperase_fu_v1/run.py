"""Proposed-only FU-objective experiment; frozen baseline trajectories untouched."""
from pathlib import Path
import os,sys,json,time,fcntl,subprocess,traceback,importlib.util
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
OUT=ROOT/'results/overnight_energy_v2/cliperase_fu_v1'
SOURCE=ROOT/'results/overnight_energy_v2/streaming_requests_v1'
sys.path.insert(0,str(ROOT/'tools/overnight_energy_v2/streaming_requests_v1'))
spec=importlib.util.spec_from_file_location('cliperase_stream_parent',ROOT/'tools/overnight_energy_v2/streaming_requests_v1/run.py')
stream=importlib.util.module_from_spec(spec);spec.loader.exec_module(stream)
sys.path.insert(0,str(HERE))
from losses import forget_loss,retain_losses,self_check,WEIGHTS
from scheduler import Queue,FL_STEPS
core=stream.core
ARMS={'parallel_cliperase_paper':dict(parallel=True,crossmodal=True,period=0,digital=False,main=True)}
ORIGINAL_VECTORS=stream.vectors

def write(name,obj):core.write(OUT/name,obj)
def status(**obj):write('status.json',dict(unix=time.time(),pid=os.getpid(),**obj))

def prepare():
    stream.OUT=SOURCE
    plan,requests=stream.prepare()
    stream.OUT=OUT;OUT.mkdir(parents=True,exist_ok=True)
    parent=json.loads((SOURCE/'protocol.json').read_text())
    hashes=dict(parent['source_hashes'])
    for p in sorted(HERE.glob('*.py')):hashes[str(p.relative_to(ROOT))]=core.sha(p)
    hashes[str((SOURCE/'protocol.json').relative_to(ROOT))]=core.sha(SOURCE/'protocol.json')
    evidence={p.name:core.sha(p) for p in (HERE/'evidence').glob('*') if p.is_file()}
    protocol=dict(version='cliperase_fu_v1',registered_unix=time.time(),arms=ARMS,source_hashes=hashes,evidence_sha256=evidence,
      source_paper='https://aclanthology.org/2025.acl-long.1469/',
      source_repository='https://github.com/Tianyu-yang-anna/ClipErase-ACL/tree/859fc63b2e9404d7d45d76bed2e3d2f482b59ec6',
      authorization='User requested applying CLIPErase-style cross-modal FU only to our method; comparison methods remain frozen.',
      base='Completed streaming_requests_v1/parallel_crossmodal. Same original summed parallel AirComp, not I/Q; isolate the FU loss before combining radio changes.',
      change='Replace safe-proxy output KL plus cross-modal retain KL with paper Eq.(2)-(5): positive-pair normalized cosine minimization FM, symmetric retain CLIP contrastive RM, and embedding-dimension softmax KL CM.',
      weights=WEIGHTS,
      adaptation='Prompt-only frozen encoders, one SGD step at0.0035 per FU service step, five steps per request, all five forget captions. Original paper trains encoder weights with Adam/epochs. No claim of faithful whole-paper reproduction.',
      teacher='Per-request model frozen at first service, identical availability to old per-request preservation teacher. CM is retain-only and stop-gradient. Safe proxy is no longer used in FU; its historical common prefix cost remains charged for conservative like-for-like reporting.',
      paper_code_discrepancy='ACL final Eq.(2) minimizes positive dot products and Eq.(4) uses embedding distributions. Public tasks/image_retrieval.py instead uses negative forget CE and retrieval-logit KL; model.py duplicates main.py. This implementation follows paper equations, not that code path.',
      fixed='Same initial round20 checkpoint, training/validation split, exogenous45 requests, retained sampling seeds,200 total FL steps,5 FU steps/request, learning rates, channels, ten heads, power/energy limits, component RMS rule and evaluation schedule as frozen parent.',
      clock='Pure uplink occupied and elapsed service time; excludes local computation, recovery/decoding, control and downlink. Request arrival times are identical; service duration may change as update norms change.',
      stages='Synthetic CPU loss/gradient tests; actual full20480-coordinate training-only gradient and radio preflight; one200-FL proposed-only trajectory if numerical and physical checks pass. No selection or tuning based on reference/test outcomes.',
      scoring='Keep old retained mR/loss, mean/worst-request retrain KL, prediction disagreement, MIA and target thresholds unchanged. Also read existing forget retrieval R1/R5/R10/mR as association-erasure outcomes. A lower forget recall is not a retraining-equivalence or privacy guarantee. Failed targets are not ranked by energy-to-success.',
      limitations='Single seed; objectives differ: association suppression can increase retrain KL because retraining may retain generic associations. FU retention CE adds supervised-retain work within the existing FU budget; report this as part of the changed objective, not a transport-only improvement.')
    path=OUT/'protocol.json'
    if path.exists():protocol=json.loads(path.read_text())
    else:write('protocol.json',protocol);write('requests.json',requests)
    for name,digest in protocol['source_hashes'].items():assert core.sha(ROOT/name)==digest,name
    for name,digest in protocol['evidence_sha256'].items():assert core.sha(HERE/'evidence'/name)==digest,name
    assert json.loads((OUT/'requests.json').read_text())==requests
    return plan,requests

def request_gradient(engine,theta,fids,rids,step,teacher):
    import torch
    fcids=[c for i in fids for c in engine.data.caption_ids[i]]
    owner=[j for j,i in enumerate(fids) for _ in engine.data.caption_ids[i]]
    variable=theta.detach().requires_grad_(True)
    im,tx,_=engine.features(variable,fids,fcids)
    fm=forget_loss(im,tx,owner)
    gf=torch.autograd.grad(fm,variable)[0].detach();gf[-1]=0
    rcids=engine.data.training_caption_ids(rids,21+step)
    old=teacher.features(rids,rcids)
    variable=theta.detach().requires_grad_(True)
    ri,rt,scale=engine.features(variable,rids,rcids)
    rm,cm=retain_losses(ri,rt,*old,scale)
    preservation=WEIGHTS['retention']*rm+WEIGHTS['consistency']*cm
    gr=torch.autograd.grad(preservation,variable)[0].detach();gr[-1]=0
    g=WEIGHTS['forgetting']*gf+gr
    assert torch.isfinite(g).all()
    return g,dict(forget_loss=float(fm.detach()),preservation_loss=float(preservation.detach()),
      FM=float(fm.detach()),RM=float(rm.detach()),CM=float(cm.detach()),
      objective=float((WEIGHTS['forgetting']*fm+preservation).detach()),
      forget_gradient_norm=float(gf.norm()),weighted_forget_gradient_norm=float((WEIGHTS['forgetting']*gf).norm()),
      preserve_gradient_norm=float(gr.norm()),FU_gradient_norm=float(g.norm()),
      forget_retain_gradient_cosine=float(torch.dot(gf,gr)/(gf.norm()*gr.norm()+1e-30)),
      images=len(fids),forget_captions=len(fcids),retained_images=rids,retained_captions=rcids)

def vectors(engine,theta,plan,queue,t,fl,kind,ids,proxy,teachers,crossmodal):
    d=len(theta)-1;l=np.zeros((10,d));u=np.zeros_like(l);stats=[]
    pools=stream.retained_pools(plan,queue,t)
    if kind in ['fl','joint'] and fl<FL_STEPS:
        k=21+fl;lr=core.lr_at(k);total=sum(map(len,pools))
        for user,pool in enumerate(pools):
            batch=np.random.default_rng(730000+k*10+user).choice(pool,min(16,len(pool)),replace=False).astype(int).tolist()
            g,_=engine.gradient(theta,batch,k)
            l[user]=(-lr*len(pool)/total*g[:-1]).cpu().numpy()
    total=sum(len(queue.items[i]['images']) for i in ids)
    for rid in ids:
        req=queue.items[rid];user=req['user'];step=req['steps'];fids=req['images']
        if rid not in teachers:teachers[rid]=core.Teacher(engine,theta)
        seed=910000+req['burst']*1000+user*10+step
        rids=np.random.default_rng(seed).choice(pools[user],min(16,len(pools[user])),replace=False).astype(int).tolist()
        g,meta=request_gradient(engine,theta,fids,rids,step,teachers[rid])
        u[user]+=(-.0035*len(fids)/total*g[:-1]).cpu().numpy()
        stats.append(dict(request=rid,step=step+1,**meta))
    return l,u,stats

def main():
    plan,requests=prepare()
    if '--prepare' in sys.argv:
        write('cpu_checks.json',self_check());print('PREPARED_CPU_PASSED',flush=True);return
    own=(OUT/'launch.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'started.json').exists(),'Duplicate campaign'
    assert json.loads((SOURCE/'finished.json').read_text())['status']=='all_registered_arms_finished_or_reported_incomplete'
    gpu=(ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a');fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied'
    assert json.loads((OUT/'cpu_checks.json').read_text())['passed']
    write('started.json',dict(unix=time.time(),identity=stream.base.identity(os.getpid())))
    status(stage='loading_model')
    from train import Engine
    from dataset import RetrievalData
    engine=Engine(RetrievalData(),dict(plan));channels=np.load(core.SOURCE/'channels.npy')
    p=ROOT/'results/overnight_energy_v2/proxy_objective_v2/proxies/analog'
    proxy_theta=core.load_theta(p/'round_020.pt',engine.device)
    prefix=json.loads((SOURCE/'common_prefix_cost.json').read_text());write('common_prefix_cost.json',prefix)
    theta=core.load_theta(core.SOURCE/'proposed/round_020.pt',engine.device)
    q=Queue(requests,True,0);ids=[r['id'] for r in q.pending(0)];assert ids
    status(stage='actual_training_update_preflight')
    proxy=core.Teacher(engine,proxy_theta)
    l,u,local=vectors(engine,theta,plan,q,0,0,'joint',ids,proxy,{},True)
    old_l,old_u,_=ORIGINAL_VECTORS(engine,theta,plan,q,0,0,'joint',ids,proxy,{},True)
    assert np.array_equal(l,old_l),'FL changed'
    assert np.isfinite(u).all() and np.linalg.norm(u)>0
    vpart=engine.model.parts(theta)[0].numel()
    assert np.linalg.norm(u[:,:vpart])>0 and np.linalg.norm(u[:,vpart:])>0,'Both modality prompt blocks must receive gradients'
    np.savez_compressed(OUT/'fixed_training_updates.npz',FL=l,FU=u,old_FU=old_u,channels=channels)
    recovered,radio=stream.radio.analog(l,u,channels,plan['config'],seed=720000)
    assert recovered is not None and radio['status']=='feasible'
    assert radio['conditional_rmse']<=radio['epsilon']*(1+1e-5)
    assert radio['max_power_w']<=plan['config']['power_limit_w']*(1+1e-5)
    write('preflight.json',dict(passed=True,FL_exactly_unchanged=True,dimension=int(l.shape[1]),
      visual_FU_norm=float(np.linalg.norm(u[:,:vpart])),text_FU_norm=float(np.linalg.norm(u[:,vpart:])),
      old_FU_norm=float(np.linalg.norm(old_u)),new_FU_norm=float(np.linalg.norm(u)),radio=radio,local=local,
      data='First arrived requests and retained training samples only. No final reference/test/validation used to choose weights.'))
    # Reporting oracle loaded only after source, weights and all preflight gates are frozen.
    ref_theta=core.load_theta(core.SOURCE/'retrain/round_200.pt',engine.device)
    ref_eval,ref_probs=engine.evaluate(ref_theta,None);ref_mia,_=engine.mia_scores(ref_theta)
    threshold=.95*ref_eval['retained']['mR']
    old_ref=json.loads((SOURCE/'reference_metrics.json').read_text())
    assert abs(threshold-old_ref['retained_mR_threshold'])<1e-7
    write('reference_metrics.json',dict(metrics=ref_eval,retained_mR_threshold=threshold,reporting_only=True));write('reference_mia.json',ref_mia)
    stream.OUT=OUT;stream.ARMS=ARMS;stream.vectors=vectors
    stream.report()
    for name,opt in ARMS.items():
        stream.run_arm(engine,plan,requests,proxy_theta,ref_probs,threshold,channels,name,opt,prefix)
        path=OUT/name/'manifest.json';manifest=json.loads(path.read_text())
        if manifest['status']=='completed_200FL_and_requests':
            e=manifest['evaluation']
            manifest['final_joint_quality_pass']=bool(manifest['requests']['procedure_completed']==45 and e['retained']['mR']>=threshold and e['forget']['KL_retrain_to_candidate']<=.01 and e['worst_request_KL']<=.01)
            manifest['objective_note']='Association-erasure objective: inspect forget recall alongside unchanged retrain-KL criterion. Low recall alone does not certify sample unlearning.'
            core.write(path,manifest)
    prepare();stream.OUT=OUT;stream.report()
    write('finished.json',dict(unix=time.time(),status='registered_proposed_candidate_finished_or_reported_incomplete'))
    status(stage='completed')

if __name__=='__main__':
    try:main()
    except Exception as e:OUT.mkdir(parents=True,exist_ok=True);status(stage='failed',error=repr(e));traceback.print_exc();raise
    finally:stream.radio.base.close()
