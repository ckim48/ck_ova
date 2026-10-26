"""Frozen streaming run_arm with only a physical-duration safety bound repair.
The old 2000*one-frame guard falsely aborts valid slow digital trajectories.
No learning, scheduling, radio or evaluation behavior is changed here.
"""
import os,time,json
from prepare_campaign import clip,core,OUT,ARMS,write
Queue=clip.stream.Queue
FRAME_S=clip.stream.FRAME_S
HORIZON=clip.stream.HORIZON
FL_STEPS=clip.stream.FL_STEPS
evaluation=clip.stream.evaluation
def status(**obj):write('status.json',dict(unix=time.time(),pid=os.getpid(),**obj))
def report():
    from metrics import report as refresh
    refresh()
vectors=None
radio=None

def run_arm(engine,plan,requests,proxy_theta,ref_probs,threshold,channels,name,opt,prefix_cost):
    from train import update_theta
    import torch
    dest=OUT/name;dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'started.json').exists(),'Duplicate arm'
    core.write(dest/'started.json',dict(unix=time.time(),policy=opt))
    theta=core.load_theta(core.SOURCE/'proposed/round_020.pt',engine.device)
    q=Queue(requests,opt['parallel'],opt['period']);teachers={};proxy=core.Teacher(engine,proxy_theta)
    t=occupied=energy=0.;fl=seq=0;horizon_saved=False;target=None;hits=[]
    last_eval=None;last_eval_fl=-1
    def point(at,ev=None):
        return dict(sequence=seq,completed_FL_rounds=20+fl,adaptation_elapsed_service_s=at,
            adaptation_uplink_occupied_s=occupied,adaptation_tx_energy_bound_j=energy,
            cumulative_uplink_occupied_s=prefix_cost['uplink_s']+occupied,
            cumulative_elapsed_service_s=prefix_cost['uplink_s']+at,
            cumulative_tx_energy_bound_j=prefix_cost['energy_j']+energy,
            requests=q.metrics(at),evaluation=ev,joint_target=target)
    for _ in range(2000):
        kind,ids=q.decision(t,fl)
        if kind=='done':break
        if kind=='wait':
            if not horizon_saved and t<=HORIZON<ids:
                ev=evaluation(engine,theta,ref_probs,q,HORIZON)
                core.write(dest/'common_horizon.json',point(HORIZON,ev));horizon_saved=True
            t=ids;continue
        status(stage='training',method=name,transaction=seq+1,FL_round=20+fl,clock_s=t,requests=q.metrics(t))
        l,u,local=vectors(engine,theta,plan,q,t,fl,kind,ids,proxy,teachers,opt['crossmodal'])
        seed=500000+seq
        vector,rm=(radio.digital_fu(u,channels,plan['config']) if kind=='fu' and opt['digital']
                   else radio.analog(l,u,channels,plan['config'],seed))
        if vector is None:
            core.write(dest/'manifest.json',dict(status='incomplete_no_feasible_radio_design',kind=kind,radio=rm,**point(t)))
            report();return
        assert rm['conditional_rmse']<=rm['epsilon']*(1+2e-5)+1e-10
        start=t;end=t+rm['uplink_s']
        # Model update becomes visible only after the whole transmission completes.
        if not horizon_saved and start<=HORIZON<end-1e-10:
            ev=evaluation(engine,theta,ref_probs,q,HORIZON)
            snap=point(HORIZON,ev)
            snap['inflight']=dict(kind=kind,start_s=start,end_s=end,
                 note='Model update not yet applied. Airtime is exact; energy is an interval because within-transaction power varies by block. Do not rank a point estimate as measured accrued energy.')
            snap['adaptation_uplink_occupied_s']+=HORIZON-start
            snap['cumulative_uplink_occupied_s']+=HORIZON-start
            snap['adaptation_energy_interval_j']=[energy,energy+rm['energy_bound_j']]
            snap['cumulative_energy_interval_j']=[prefix_cost['energy_j']+energy,prefix_cost['energy_j']+energy+rm['energy_bound_j']]
            snap['adaptation_tx_energy_bound_j']=None
            snap['cumulative_tx_energy_bound_j']=None
            core.write(dest/'common_horizon.json',snap);horizon_saved=True
        theta=update_theta(theta,vector)
        if kind in ['fl','joint'] and fl<FL_STEPS:fl+=1
        q.commit(ids,start,end);t=end;occupied+=rm['uplink_s'];energy+=rm['energy_bound_j'];seq+=1
        completed_now=any(q.items[i]['completed_s']==end for i in ids)
        due=(fl!=last_eval_fl and fl%5==0) or completed_now or (not horizon_saved and t>=HORIZON-1e-10)
        if due:
            last_eval=evaluation(engine,theta,ref_probs,q,t);last_eval_fl=fl
            done=q.metrics(t)['procedure_completed']==len(requests)
            ok=done and last_eval['retained']['mR']>=threshold and last_eval['forget']['KL_retrain_to_candidate']<=.01 and last_eval['worst_request_KL']<=.01
            hits=hits+[dict(fl_round=20+fl,time_s=t)] if ok else []
            if len(hits)>=3 and target is None:target=dict(confirmation_FL_round=20+fl,elapsed_service_s=prefix_cost['uplink_s']+t,uplink_s=prefix_cost['uplink_s']+occupied,energy_j=prefix_cost['energy_j']+energy,first_pass=hits[0])
            core.write(dest/f'evaluations/{seq:04d}.json',point(t,last_eval))
            core.save_theta(dest/f'checkpoints/{seq:04d}.pt',theta,20+fl)
            if not horizon_saved and t>=HORIZON-1e-10:
                core.write(dest/'common_horizon.json',point(HORIZON,last_eval));horizon_saved=True
        row=dict(**point(t,last_eval if due else None),kind=kind,FU_request_ids=ids,radio=rm,local=local)
        core.write(dest/f'transactions/{seq:04d}.json',row);core.write(dest/'latest.json',row)
        if due:report()
        if t>2000*plan['config']['round_uplink_deadline_s']+max(r['arrival_s'] for r in requests)+10*FRAME_S:raise RuntimeError('Service clock exceeds transaction safety bound')
    else:raise RuntimeError('Transaction safety limit')
    final=evaluation(engine,theta,ref_probs,q,t)
    if not horizon_saved:core.write(dest/'common_horizon.json',dict(**point(HORIZON,final),note='Workload finished before common horizon; no extra training.'))
    mia,_=engine.mia_scores(theta);refmia=json.loads((OUT/'reference_mia.json').read_text())
    mia['auc_gap_vs_retrain']=abs(mia['loss_score_auc']-refmia['loss_score_auc'])
    core.write(dest/'mia.json',mia);core.save_theta(dest/'final.pt',theta,20+fl)
    core.write(dest/'request_records.json',list(q.items.values()))
    core.write(dest/'manifest.json',dict(status='completed_200FL_and_requests',**point(t,final)))
    report()
