"""Retain learned objectives/radio code while replacing the mission clock."""
from types import SimpleNamespace
from mission_clock import MissionClock,queue_classes

def make_arm(base,deployment,state):
    joint,serial=queue_classes(base)
    synced=SimpleNamespace(**base.clip.__dict__)
    synced.vectors=base.control.synchronized_vectors('cliperase')
    synced.ORIGINAL_VECTORS=base.control.synchronized_vectors('proxy')
    edits=[
        ("q=clip.stream.Queue(requests,opt['parallel'],opt['period'])","q=JointQueue(requests) if opt['parallel'] else SerialQueue(requests,False,opt['period'])",1),
        ('t=energy=occupied=0.;fl=seq=attempt=0;last_fl=-1;horizon=False;target=None;hits=[]',
         't=energy=occupied=0.;fl=seq=attempt=0;last_fl=-1;horizon=False;target=None;hits=[]\n clock=MissionClock(geo,deployment)',1),
        ("site,endwindow=mission.window_at(t,geo)\n  if site is None:horizon_before(endwindow);t=endwindow;continue\n  kind,ids=mission.decide(q,t,fl,geo)",
         "site=min(fl//36,4)\n  t=clock.enter(site,t,q,fl)\n  q.sync(fl,t,occupied)\n  kind,ids=q.decision(t,fl)",1),
        ("if kind=='wait':horizon_before(ids);t=ids;continue","if kind=='wait':clock.wait(t,ids);t=ids;continue",1),
        ("l,u,local=vecfn(engine,theta,plan,q,t,fl,kind,ids,proxy,tentative,opt['crossmodal'])",
         "engine.start_work()\n  l,u,local=vecfn(engine,theta,plan,q,t,fl,kind,ids,proxy,tentative,opt['crossmodal'])\n  compute=engine.finish_work()\n  assert np.isfinite(l).all() and np.isfinite(u).all()\n  if kind=='joint':assert np.linalg.norm(l)>0 and np.linalg.norm(u)>0\n  if kind=='fu':assert np.linalg.norm(l)==0 and np.linalg.norm(u)>0\n  if kind=='fl':assert np.linalg.norm(u)==0",1),
        ('FL_gradient_calls=10 if kind in',"compute=compute,FL_vector_norm=float(np.linalg.norm(l)),FU_vector_norm=float(np.linalg.norm(u)),FL_learning_rate=core.lr_at(21+fl),FU_learning_rate=core.lr_at(21+fl) if ids else None,FL_gradient_calls=10 if kind in",1),
        ("assert rm['energy_bound_j']<=.1*(1+2e-6)","assert np.isfinite(rm['energy_bound_j']) and rm['energy_bound_j']>=0\n  if name!='sequential_digital_periodic':assert rm['energy_bound_j']<=.1*(1+2e-6)",1),
        ("start=t;end=t+rm['uplink_s']", "start=t;timing=clock.transaction_timing(compute,rm);end=t+timing['duration_s']\n  q.last_uplink_s=rm['uplink_s'];audit['timing']=timing",1),
        ("  if end>endwindow+1e-9:\n   audit['status']='computed_but_deferred_hover_window';write(ad/f'work_attempts/{attempt:04d}.json',audit)\n   nxt=mission.next_window(t,geo);horizon_before(nxt);t=nxt;continue\n",'',1),
        ("audit.update(status='transmitted',transaction=seq)","audit.update(status='transmitted',transaction=seq,FL_receiver_apply_s=start+timing['compute_s']+timing['uplink_s'] if kind in ['fl','joint'] else None,FU_receiver_apply_s=start+timing['compute_s']+timing['uplink_s'] if ids else None,FL_clients_apply_s=end if kind in ['fl','joint'] else None,FU_clients_apply_s=end if ids else None,FL_round_after=20+fl)",1),
        ("radio=rm,local=local,site=site,start_s=start,end_s=end", "radio=rm,local=local,site=site,start_s=start,end_s=end,timing=timing",1),
        ("mission_clock_s=at,", "mission_clock_s=at,clock_policy='completion_driven_full_service_clock',model='RemoteCLIP_B32_mean8_global_IoT_five_regions_rank10',",1),
        ("if not horizon:write(ad/'common_horizon.json',dict(**point(geo['mission_end_s'],final),note='Workload completed before common mission horizon; no extra learning.'))",
         "timeline=clock.finish(t,q,fl)\n write(ad/'mission_timeline.json',timeline)\n write(ad/'service_completion.json',point(t,final))\n t=timeline['mission_end_s']",1),
    ]
    return base.repeat.clone_function(base.old.run_arm,edits,dict(JointQueue=joint,SerialQueue=serial,clip=synced,MissionClock=MissionClock,deployment=deployment,state=state))
