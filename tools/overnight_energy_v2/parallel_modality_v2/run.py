"""Modality-wise FU control; reuse three completed matched transport arms.

Custom controlled ablation, not a reproduction of a published FU algorithm.
No cross-modal similarity matrix is formed by its FU or preservation losses.
The unchanged retained FL objective still trains the same CLIP retrieval task.
"""
from pathlib import Path
import sys,os,json,time,fcntl,subprocess,importlib.util,traceback
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD=ROOT/'results/overnight_energy_v2/parallel_aircomp_v1'
OUT=ROOT/'results/overnight_energy_v2/parallel_modality_v2'
sys.path.insert(0,str(ROOT/'tools/overnight_energy_v2/parallel_aircomp_v1'))
spec=importlib.util.spec_from_file_location('parallel_base',ROOT/'tools/overnight_energy_v2/parallel_aircomp_v1/run.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
core=base.core;radio=base.radio
import numpy as np
METHOD='modalitywise_fu'
LABEL='Parallel AirComp + modality-wise FU (intra-modal relational KL)'
REUSE=['proposed','separate_aircomp','aircomp_oma']
original_updates=core.new_updates

def intra_loss(images,texts,old_images,old_texts,scale):
    import torch
    import torch.nn.functional as F
    def one(features,teacher):
        n=len(features)
        if n<2:return features.sum()*0.
        mask=~torch.eye(n,dtype=torch.bool,device=features.device)
        logits=(scale.exp()*features@features.T)[mask].reshape(n,n-1)
        old=(scale.detach().exp()*teacher.detach()@teacher.detach().T)[mask].reshape(n,n-1)
        return F.kl_div(F.log_softmax(logits,1),F.softmax(old,1),reduction='batchmean')
    return .5*(one(images,old_images)+one(texts,old_texts))

def gradient(engine,theta,ids,cids,old):
    import torch
    if len(ids)<2:return torch.zeros_like(theta),0.
    variable=theta.detach().requires_grad_(True)
    im,tx,scale=engine.features(variable,ids,cids)
    loss=intra_loss(im,tx,*old,scale)
    grad=torch.autograd.grad(loss,variable)[0].detach();grad[-1]=0
    assert torch.isfinite(grad).all() and torch.isfinite(loss)
    return grad,float(loss.detach())

def updates(engine,theta,rec,k,forget,teacher,crossmodal=True):
    lr=engine.plan['learning_rate'];n=len(rec['retained']);d=len(theta)-1
    l=np.zeros((n,d));u=np.zeros_like(l);stats=[]
    count=sum(map(len,rec['retained']));nf=sum(map(len,forget))
    for user,ids in enumerate(rec['batches']):
        g,loss=engine.gradient(theta,ids,k)
        l[user]=(-lr*len(rec['retained'][user])/count*g[:-1]).cpu().numpy()
        row=dict(user=user,retained_loss=loss)
        if forget[user] and k in [21,22,23,24,25]:
            fids=np.random.default_rng(9000+k*10+user).choice(forget[user],min(16,len(forget[user])),replace=False).astype(int).tolist()
            fcids=[c for i in fids for c in engine.data.caption_ids[i]]
            fg,fl=gradient(engine,theta,fids,fcids,teacher.features(fids,fcids))
            rcids=engine.data.training_caption_ids(ids,k)
            rg,reg=gradient(engine,theta,ids,rcids,engine.teacher_features(ids,rcids))
            u[user]=(-lr*len(forget[user])/nf*(fg+rg)[:-1]).cpu().numpy()
            row.update(forget_intra_modal_KL=fl,retained_intra_modal_KL=reg,
                       forget_gradient_norm=float(fg.norm()),preserve_gradient_norm=float(rg.norm()),
                       forget_images=len(fids),forget_captions=len(fcids),cross_modal_FU_matrix=False)
        stats.append(row)
    return l,u,stats

def transport(l,u,channels,c,mode,seed):
    if np.any(l) or np.any(u):return radio.communicate(l,u,channels,c,mode,seed)
    return np.zeros(l.shape[1]),dict(status='feasible',physical_mode=mode,access='zero_update_silence',
        head=None,energy_bound_j=0.,expected_energy_j=0.,uplink_s=0.,frame_count=0,
        network_tx_cap_j=radio.CAP,max_user_energy_j=0.,max_power_w=0.,
        error_bound=0.,epsilon=c['absolute_tolerance_floor'],conditional_rmse=0.,
        quantization_error_bound=0.,requests_in_update=0,realized_error_to_raw=0.,
        optimizer_wall_s=0.,phases=[],zero_update_audit=True)

def cpu_checks():
    import torch
    import torch.nn.functional as F
    torch.manual_seed(81127)
    im=F.normalize(torch.randn(5,12,dtype=torch.float64),dim=-1)
    tx=F.normalize(torch.randn(15,12,dtype=torch.float64),dim=-1)
    s=torch.tensor(1.,dtype=torch.float64)
    assert abs(float(intra_loss(im,tx,im,tx,s)))<1e-12
    x=(im+.05*torch.randn_like(im)).requires_grad_(True)
    y=(tx+.05*torch.randn_like(tx)).requires_grad_(True)
    v=intra_loss(x,y,im,tx,s);gx,gy=torch.autograd.grad(v,(x,y))
    assert gx.norm()>0 and gy.norm()>0
    assert intra_loss(x-.001*gx,y-.001*gy,im,tx,s)<v
    pi=torch.randperm(len(im));pt=torch.randperm(len(tx))
    assert torch.allclose(v,intra_loss(x[pi],y[pt],im[pi],tx[pt],s),atol=1e-12)
    # Within-modality Gram distributions cannot see relative V/T rotations.
    rot=torch.linalg.qr(torch.randn(12,12,dtype=torch.float64)).Q
    assert abs(float(intra_loss(im@rot,tx,im,tx,s)))<1e-12
    assert float(core.proxy_loss(im@rot,tx,im,tx,s))>1e-5
    z=np.zeros((10,20480));c={'absolute_tolerance_floor':1e-8}
    vector,meta=transport(z,z,np.ones((10,10)),c,'joint',0)
    assert not np.any(vector) and meta['energy_bound_j']==meta['uplink_s']==0
    return dict(identity_zero=True,finite_nonzero_gradients_both_modalities=True,
        descent_sign=True,independent_modality_permutation_invariance=True,
        relative_rotation_blindness_verified=True,zero_update_silence=True)

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    plan,parent=base.prepare()
    assert json.loads((OLD/'finished.json').read_text())['status']=='completed'
    required=[OLD/'protocol.json',OLD/'summary.json',OLD/'finished.json',
              ROOT/'tools/overnight_energy_v2/proxy_objective_v1/run.py',
              ROOT/'tools/overnight_energy_v2/parallel_aircomp_v1/run.py',
              ROOT/'tools/overnight_energy_v2/parallel_aircomp_v1/radio_parallel.py',Path(__file__)]
    required += [OLD/m/'manifest.json' for m in REUSE]
    hashes={str(p.relative_to(ROOT)):core.sha(p) for p in required}
    protocol=dict(version='parallel_modality_v2',registered_unix=time.time(),seed=0,max_rounds=200,
        main_methods={'proposed':'Parallel AirComp + cross-modal FU',
                      'separate_aircomp':'Separate AirComp + identical cross-modal FU',
                      METHOD:LABEL,'aircomp_oma':'AirComp + actual 128-tone OFDMA + identical cross-modal FU'},
        reused_completed_200_arms=REUSE,new_trajectories=[METHOD],
        authorization='User agreed to the revised four-way comparison and asked whether it was already running; launch only the missing modality-wise FU arm.',
        FU='Replace BOTH cross-modal proxy FU KL and cross-modal retained preservation KL by equally weighted image-image and text-text KL. Each compares row-softmax similarity distributions with the same teacher, excluding self diagonal. No image-text similarity matrix in either FU term.',
        teachers='Forget targets: existing retained-only proxy20. Retain targets: existing pre-request round20 model. Same fixed logit scale, loss weights 1+1, same sample IDs and caption counts as parent; no loss-dependent tuning.',
        unchanged='Retained FL remains the same cross-modal CLIP retrieval loss. Architecture, initialization, proxy, channels, seed, minibatches, local steps, weights, power/energy limits, delay objective and target thresholds exactly match parent.',
        limitations='Custom controlled ablation, not an existing paper reproduction and not a claim that AirComp prevents cross-modal FU. Intra-modal relational KL does not identify relative embedding orientation, by construction. Single seed and one FU episode21..25; no universal conclusion about modality-wise unlearning. Different gradient magnitudes are recorded, not retuned against outcomes.',
        request_schedule=parent['requests'],radio_parent=parent['analog_solver'],
        protocol_parent=str((OLD/'protocol.json').relative_to(ROOT)),
        evaluation='All feasible rounds through200; scheduled metrics/MIA only for reporting, never coefficient/teacher selection. Same original goals, prefix/proxy costs and zero-LR endpoint silence.',
        supplementary='Old no-retained-relation-KL ablation remains in original folder; it is not the new modality-wise FU comparator.',source_hashes=hashes)
    p=OUT/'protocol.json'
    if p.exists():protocol=json.loads(p.read_text())
    else:core.write(p,protocol)
    for name,h in protocol['source_hashes'].items():assert core.sha(ROOT/name)==h,name
    return plan,protocol

def report(plan):
    old=json.loads((OLD/'summary.json').read_text())
    kept={x['method']:dict(x,reused_from='parallel_aircomp_v1',independent_rerun=False) for x in old['methods'] if x['method'] in REUSE}
    d=OUT/METHOD;rows=[json.loads(p.read_text()) for p in sorted((d/'rounds').glob('*.json'))]
    evs=[r for r in rows if 'evaluation' in r];hits=[];target=None
    for r in evs:
        e=r['evaluation'];worst=max(x['KL_retrain_to_candidate'] for x in e['per_request'].values() if x['images'])
        ok=r['round']>=25 and e['retained']['mR']>=.29304036883016427 and e['forget']['KL_retrain_to_candidate']<=.01 and worst<=.01 and r['request_algorithm_completed']
        hits=hits+[r] if ok else []
        if len(hits)>=3 and target is None:target=dict(entry_round=hits[0]['round'],confirmation_round=r['round'],energy_j=r['cumulative_tx_energy_bound_j'],uplink_s=r['cumulative_uplink_s'])
    manifest=json.loads((d/'manifest.json').read_text()) if (d/'manifest.json').exists() else {'status':'pending_or_running'}
    entry=dict(method=METHOD,label=LABEL,status=manifest['status'],completed_round=rows[-1]['round'] if rows else 20,joint_target=target)
    if evs:
        r=evs[-1];e=r['evaluation'];entry['last_evaluation']=dict(round=r['round'],retained=e['retained'],forget=e['forget'],per_request=e['per_request'],worst_request_KL=max(x['KL_retrain_to_candidate'] for x in e['per_request'].values() if x['images']))
    if rows:entry['costs']={k:rows[-1][k] for k in ['cumulative_tx_energy_bound_j','cumulative_uplink_s']}
    ms=sorted(d.glob('mia_*.json'))
    if ms:entry['mia']=json.loads(ms[-1].read_text())
    core.write(OUT/'summary.json',dict(methods=[kept['proposed'],kept['separate_aircomp'],entry,kept['aircomp_oma']],target=old['target'],
        limitations='Three unchanged completed trajectories reused; only modality-wise FU is new. No superiority conclusion from unequal horizons. Single-seed custom ablation. No deletion guarantee from KL/MIA. Pure uplink time and TX energy upper bounds. Exact-inversion family, not unrestricted transceiver optimum.'))

def main():
    plan,protocol=prepare()
    core.write(OUT/'cpu_checks.json',cpu_checks())
    if '--prepare' in sys.argv:print('PREPARED_CPU_CHECKED');return
    own=(OUT/'launch.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'started.json').exists(),'Duplicate launch'
    lock=(ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied'
    core.OUT=OUT;core.METHODS=[METHOD];core.mode=lambda method:'joint';core.prefix=base.prefix;core.report=report
    core.write(OUT/'started.json',dict(unix=time.time(),identity=base.identity(os.getpid())))
    core.status(stage='loading_model',campaign='parallel_modality_v2')
    from train import Engine
    from dataset import RetrievalData
    engine=Engine(RetrievalData(),dict(plan));records=json.loads((core.SOURCE/'records.json').read_text());forget=records[20]['forget'];channels=np.load(core.SOURCE/'channels.npy')
    p=ROOT/'results/overnight_energy_v2/proxy_objective_v2/proxies/analog';meta=json.loads((p/'complete.json').read_text());assert meta['status']=='completed'
    proxy=core.load_theta(p/'round_020.pt',engine.device)
    theta=core.load_theta(core.SOURCE/'proposed/round_020.pt',engine.device);engine.set_teacher(theta)
    teacher=core.Teacher(engine,proxy);engine.plan['learning_rate']=core.lr_at(21)
    core.status(stage='fixed_initial_update_validation')
    l,u,stats=updates(engine,theta,records[20],21,forget,teacher)
    ol,ou,os_=original_updates(engine,theta,records[20],21,forget,teacher,True)
    assert np.allclose(l,ol,rtol=0,atol=0),'FL changed'
    for row in stats:
        if forget[row['user']]:assert row['forget_gradient_norm']>1e-10,'Zero FU gradient'
    vector,rm=transport(l,u,channels,plan['config'],'joint',100021)
    assert vector is not None and rm['status']=='feasible'
    assert rm['conditional_rmse']<=rm['epsilon']*(1+2e-5)
    assert rm['max_power_w']<=plan['config']['power_limit_w']*(1+2e-6)
    core.write(OUT/'preflight.json',dict(passed=True,FL_exactly_unchanged=True,
        new_local_statistics=stats,original_local_statistics=os_,new_FU_norm=float(np.linalg.norm(u)),
        original_FU_norm=float(np.linalg.norm(ou)),radio=rm,
        source='Initial round20 checkpoint and training batches only; no final reference or validation/test evaluation. No parameter selection.',
        preprocessing_cost_note='One fixed-update numerical validation, not applied to training; normal trajectory restarts at shared round20.'))
    core.new_updates=updates
    proxies={'analog':(proxy,meta)}
    for stop in [40,200]:
        core.physical(engine,plan,records,forget,proxies,transport,channels,METHOD,stop)
        core.write(OUT/f'stage_{stop}_finished.json',dict(unix=time.time(),settings_unchanged=True))
    prepare();report(plan)
    final=json.loads((OUT/METHOD/'manifest.json').read_text())
    core.write(OUT/'finished.json',dict(unix=time.time(),status=final['status']))
    core.status(stage='completed' if final['status']=='completed_200' else 'incomplete',method=METHOD)

if __name__=='__main__':
    try:main()
    except Exception as e:
        OUT.mkdir(parents=True,exist_ok=True);core.write(OUT/'status.json',dict(unix=time.time(),pid=os.getpid(),stage='failed',error=repr(e)));traceback.print_exc();raise
    finally:radio.close()
