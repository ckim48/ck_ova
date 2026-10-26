"""Queued request-observable VLM update experiment. Frozen sources are read-only."""
from pathlib import Path
import os,sys,json,time,fcntl,traceback
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'tools/overnight_energy_v2/proxy_objective_v1'))
import importlib.util
spec=importlib.util.spec_from_file_location('proxy_core_v1',ROOT/'tools/overnight_energy_v2/proxy_objective_v1/run.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
sys.path.insert(0,str(HERE))
import controller
import numpy as np
OUT=ROOT/'results/overnight_energy_v2/request_projection_v4'
PRIOR=ROOT/'results/overnight_energy_v2/proxy_objective_v2'
SOURCE=core.SOURCE
ARMS=['sum','aggregate_project','request_project','local_project']
METHODS=['proposed','aircomp_only','aircomp_oma','oma_only','hybrid_no_crossmodal','aircomp_local_project']
LABELS={**core.LABELS,'aircomp_local_project':'AirComp with local request projection'}


def write(path,x):core.write(path,x)
def status(**x):write(OUT/'status.json',dict(pid=os.getpid(),unix=time.time(),**x))


def canonical(l,u,c):
    from phase_split_grid import make_inputs
    inputs=make_inputs(l,u,c)
    q=inputs[2].reshape(len(u),-1)[:,:u.shape[1]]
    eps=max(c['absolute_tolerance_floor'],c['relative_tolerance']*float(np.linalg.norm(inputs[5].sum(0))))
    return q,eps


def controlled(l,u,q,kind,received=None):
    if kind=='sum':
        d=(l+q).sum(0) if received is None else received
        info=dict(status='sum',correction_ratio=0.)
    elif kind=='local_project':
        local,ds=controller.local_project(l,q)
        d=local.sum(0)
        info=dict(status='local_projection',local=ds,
                  correction_ratio=float(np.linalg.norm(d-(l+q).sum(0))/max(np.linalg.norm((l+q).sum(0)),1e-14)))
    else:
        lh=l.sum(0) if received is None else received-q.sum(0)
        d,info=controller.global_project(lh,q,aggregate=kind=='aggregate_project')
    # True signals are used only to audit; they are not inputs to receiver control.
    true=np.vstack([l.sum(0),u[np.linalg.norm(u,axis=1)>1e-14]])
    denom=np.maximum(np.linalg.norm(true,axis=1)*max(np.linalg.norm(d),1e-14),1e-30)
    info.update(true_objective_update_cosines=(true @ d/denom).tolist(),
                true_objectives_harmed_first_order=int(np.sum(true@d < -1e-10)),
                stalled=bool(np.linalg.norm(d)<1e-10))
    return d,info


def protocol():
    return dict(version='request_projection_v4',seed=0,
      objective='Unchanged proxy_objective_v1: retain CE + retain-only-proxy bidirectional forget KL + pre-request retained crossmodal KL.',
      direction_scope='Each decoded U direction is for the combined request objective (proxy forget KL plus retained relation KL), not pure forget KL. Deletion quality itself is assessed separately; projection need not decrease pure forget KL.',
      change='Replace final unconditional sum by nearest update d to d0 subject to <l_sum,d>>=0 and <qU_i,d>>=0 for every requesting UAV. Zero margin, no tuned penalty. A zero step is feasible and reported as a stall.',
      physical_information='Receiver uses received learning aggregate = delivered sum minus decoded Q8 request vectors. Original unquantized vectors are audit-only. Same controller available to NOMA+AirComp, OMA+AirComp and OMA-only.',
      aircomp='Original single-sum AirComp retained. Additional strong baseline applies same projection locally at each UAV before one analog sum; no free server access to individual request vectors.',
      ideal_arms=ARMS,physical_methods=METHODS,
      schedule='Same checkpoints20, data, local batches, five deletion steps21..25; pilot through40, evaluation25/30/35/40. If gate passes all six physical arms through40 then200.',
      pilot_gate='At both25 and40 request projection must reduce worst-request reference KL by at least1% vs sum, not increase aggregate forget KL by more than0.1%+1e-6, retain mR within0.01 and loss within0.02; controller must actually change an update. Gate is development allocation only, no test tuning.',
      accounting='Reuse matching prior prefix and paid20-round proxy costs per independent arm. Same Q8 canonical target, power and deadline, pre-controller absolute epsilon policy. Log realized error after controller separately; no post-controller MSE guarantee asserted.',
      quality='Report global and worst per-request KL, top1 differences, retained loss/mR, MIA gap, all5 request steps, energy-to-joint-target and uplink-time-to-joint-target. Original mean-KL target plus stricter all-request-KL<=0.01 target, each confirmed on3 consecutive evaluations.',
      delay_scope='TX uplink occupation only as requested; excludes compute, receiver/SIC, downlink, flight. Computational experiment runtime is separate.',
      provenance='New exploratory zero-margin projection; inspired by gradient-conflict literature, not an exact reproduction or proven federated unlearning guarantee.',
      references=[{'title':'Conflict-Averse Gradient Descent for Multi-task Learning, NeurIPS2021','url':'https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html'},
                  {'title':'Federated Unlearning with Gradient Descent and Conflict Mitigation, AAAI2025','url':'https://ojs.aaai.org/index.php/AAAI/article/download/34181/36336'}],
      no_claim='No guarantee of NOMA advantage. OMA also preserves individual information; local projection may let AirComp perform equally well. Test set unused.',
      sources={str(p.relative_to(ROOT)):core.sha(p) for p in [Path(__file__),HERE/'controller.py',ROOT/'tools/overnight_energy_v2/proxy_objective_v1/run.py']})


def worst(e):return max(x['KL_retrain_to_candidate'] for x in e['per_request'].values() if x['images'])


def pilot(engine,plan,records,forget):
    from train import update_theta
    initial=core.load_theta(SOURCE/'proposed/round_020.pt',engine.device)
    engine.set_teacher(initial)
    proxy=core.load_theta(PRIOR/'proxies/ideal/round_020.pt',engine.device)
    teacher=core.Teacher(engine,proxy)
    results={};checks={};changed=False
    for arm in ARMS:
        theta=initial.clone();results[arm]={}
        for k in range(21,41):
            t0=time.time();status(stage='pilot',arm=arm,round=k)
            engine.plan['learning_rate']=core.lr_at(k)
            l,u,stats=core.new_updates(engine,theta,records[k-1],k,forget,teacher)
            q,_=canonical(l,u,plan['config'])
            d,diag=controlled(l,u,q,arm)
            if arm=='request_project' and diag['correction_ratio']>1e-4:changed=True
            if k==21:
                np.savez_compressed(OUT/'pilot'/arm/'common_updates.npz',learning=l,unlearning=u,canonical_unlearning=q)
            theta=update_theta(theta,d)
            row=dict(round=k,arm=arm,controller=diag,local=stats)
            if k%5==0:
                e,_=core.evaluate(engine,theta,k,forget);row['evaluation']=e
                results[arm][str(k)]=e
            if k in [25,40]:core.save_theta(OUT/'pilot'/arm/f'round_{k:03d}.pt',theta,k)
            row['wall_s']=time.time()-t0;write(OUT/'pilot'/arm/f'round_{k:03d}.json',row)
            print(json.dumps(dict(stage='pilot',arm=arm,round=k,controller=diag,wall_s=row['wall_s'])),flush=True)
    for k in ['25','40']:
        a,b=results['sum'][k],results['request_project'][k]
        checks[k]=dict(worst_request_KL_improved=worst(b)<=.99*worst(a),
            mean_forget_not_degraded=b['forget']['KL_retrain_to_candidate']<=1.001*a['forget']['KL_retrain_to_candidate']+1e-6,
            retained_mR_preserved=b['retained']['mR']>=a['retained']['mR']-.01,
            retained_loss_preserved=b['retained']['loss']<=a['retained']['loss']+.02)
    gate=dict(advance=changed and all(all(c.values()) for c in checks.values()),changed=changed,checks=checks,results=results,
              conclusion='Algorithm pilot only; ideal transmissions have no energy ranking. No deletion guarantee.')
    write(OUT/'pilot/gate.json',gate)
    findings()
    return gate


def prefix_method(method):
    return 'aircomp_only' if method=='aircomp_local_project' else method


def physical(engine,plan,records,forget,transport,channels,method,stop):
    from train import update_theta
    dest=OUT/method;dest.mkdir(exist_ok=True)
    if (dest/'manifest.json').exists() and json.loads((dest/'manifest.json').read_text())['status'].startswith('incomplete'):return
    src,prefix=core.prefix(prefix_method(method));pk='oma' if method=='oma_only' else 'analog'
    cost=json.loads((PRIOR/'proxies'/pk/'complete.json').read_text())
    teacher=core.Teacher(engine,core.load_theta(PRIOR/'proxies'/pk/'round_020.pt',engine.device))
    initial=core.load_theta(src/'round_020.pt',engine.device);engine.set_teacher(initial)
    prior=sorted((dest/'rounds').glob('*.json'))
    if prior:
        row=json.loads(prior[-1].read_text());k0=row['round'];theta=core.load_theta(dest/f'round_{k0:03d}.pt',engine.device)
        e=row['cumulative_tx_energy_bound_j'];ee=row['cumulative_tx_energy_expected_j'];tt=row['cumulative_uplink_s']
    else:
        k0=20;theta=initial;last=prefix[-1]
        e=last['cumulative_tx_energy_bound_j']+cost['energy_bound_j']
        ee=last['cumulative_tx_energy_expected_j']+cost['expected_energy_j'];tt=last['cumulative_uplink_s']+cost['uplink_s']
        write(dest/'prefix_and_proxy_cost.json',dict(prefix=last,proxy=cost,prefix_sha256=core.sha(src/'round_020.pt'),paid_per_arm=True))
    for k in range(k0+1,stop+1):
        t0=time.time();status(stage='physical',method=method,round=k,stage_stop=stop)
        engine.plan['learning_rate']=core.lr_at(k)
        l,u,stats=core.new_updates(engine,theta,records[k-1],k,forget,teacher,method!='hybrid_no_crossmodal')
        q,eps=canonical(l,u,plan['config'])
        c=dict(plan['config'],absolute_tolerance_floor=eps,relative_tolerance=0.)
        if method=='aircomp_local_project':
            local,ds=controller.local_project(l,q)
            delivered,radio=transport(local,np.zeros_like(u),channels,c,'aircomp_only',100000+k)
            nominal=local.sum(0);kind='local_project'
        else:
            delivered,radio=transport(l,u,channels,c,core.mode(method),100000+k)
            nominal=(l+q).sum(0);kind='sum' if method=='aircomp_only' else 'request_project'
        if delivered is None:
            write(dest/'manifest.json',dict(status='incomplete_no_feasible_design_found',round=k,radio=radio));return
        e+=radio['energy_bound_j'];ee+=radio['expected_energy_j'];tt+=radio['uplink_s']
        if kind=='local_project':
            vector=delivered;ideal,diag=controlled(l,u,q,kind)
            true=np.vstack([l.sum(0),u[np.linalg.norm(u,axis=1)>1e-14]])
            diag['true_objectives_harmed_first_order_after_radio']=int(np.sum(true@vector < -1e-10))
        else:
            # Only separately decoded digital modes can use q and the learning residual.
            assert kind=='sum' or radio['individual_request_vectors_recovered']
            vector,diag=controlled(l,u,q,kind,received=delivered)
            ideal,_=controlled(l,u,q,kind)
        diag['realized_postcontroller_error']=float(np.linalg.norm(vector-ideal))
        diag['raw_channel_error']=float(np.linalg.norm(delivered-nominal))
        theta=update_theta(theta,vector)
        row=dict(round=k,method=method,radio=radio,controller=diag,local=stats,
            cumulative_tx_energy_bound_j=e,cumulative_tx_energy_expected_j=ee,cumulative_uplink_s=tt,
            request_steps_executed=k>=25,logical_requests=5 if k>=21 else 0)
        if k==21 or k%5==0:
            ev,prob=core.evaluate(engine,theta,k,forget);row['evaluation']=ev
            np.savez_compressed(dest/f'probabilities_{k:03d}.npz',log_image=prob[0].numpy(),log_text=prob[1].numpy())
        core.save_theta(dest/f'round_{k:03d}.pt',theta,k)
        row['wall_s']=time.time()-t0;write(dest/'rounds'/f'{k:03d}.json',row)
        if 'evaluation' in row:report()
        print(json.dumps(dict(stage='physical',method=method,round=k,wall_s=row['wall_s'])),flush=True)
    mia,_=engine.mia_scores(theta);reference=json.loads((SOURCE/'retrain/mia.json').read_text())
    mia['auc_gap_vs_retrain']=abs(mia['loss_score_auc']-reference['loss_score_auc'])
    write(dest/f'mia_{stop:03d}.json',mia)
    write(dest/'manifest.json',dict(status='completed_200' if stop==200 else 'checkpoint40',round=stop));report()


def report():
    target=.95*json.loads((SOURCE/'retrain/rounds/200.json').read_text())['evaluation']['retained']['mR']
    result=[]
    for m in METHODS:
        dest=OUT/m;rows=[json.loads(p.read_text()) for p in sorted((dest/'rounds').glob('*.json'))]
        evs=[r for r in rows if 'evaluation' in r];out=dict(method=m,label=LABELS[m],completed_rounds=rows[-1]['round'] if rows else 20)
        out['status']=json.loads((dest/'manifest.json').read_text())['status'] if (dest/'manifest.json').exists() else 'pending_or_running'
        def point(r):
            ev=r['evaluation'];return dict(round=r['round'],retained_mR=ev['retained']['mR'],retained_loss=ev['retained']['loss'],
              forget_KL=ev['forget']['KL_retrain_to_candidate'],worst_request_KL=worst(ev),energy_bound_j=r['cumulative_tx_energy_bound_j'],
              energy_expected_j=r['cumulative_tx_energy_expected_j'],uplink_s=r['cumulative_uplink_s'])
        if evs:out['last']=point(evs[-1])
        for metric,thresholds in [('mR',[.25,.29,.30]),('loss',[4.1,4.,3.9])]:
            out['utility_only_'+metric+'_targets']={}
            for threshold in thresholds:
                hit=next((r for r in evs if r['round']>=25 and
                  (r['evaluation']['retained']['mR']>=threshold if metric=='mR' else r['evaluation']['retained']['loss']<=threshold)),None)
                out['utility_only_'+metric+'_targets'][str(threshold)]=point(hit) if hit else None
        for strict in [False,True]:
            key='all_request_joint_target' if strict else 'mean_joint_target';hits=[];out[key]=None
            for r in evs:
                e=r['evaluation'];kl=worst(e) if strict else e['forget']['KL_retrain_to_candidate']
                ok=r['round']>=25 and e['retained']['mR']>=target and kl<=.01
                hits=hits+[r] if ok else []
                if len(hits)>=3:out[key]=dict(entry_round=hits[0]['round'],confirmation=point(r));break
        mia=sorted(dest.glob('mia_*.json'))
        if mia:out['mia']=json.loads(mia[-1].read_text())
        result.append(out)
    write(OUT/'summary.json',dict(methods=result,retained_mR_target=target,forget_KL_target=.01,
        paired_additive_control=str(PRIOR/'summary.json'),
        ranking_policy='Only quality-confirmed completed request trajectories; no pending,failed,ideal or missed-target energy winners.',
        scope='Single seed development. TX energy and uplink only. Post-controller error is audited, not bounded by the raw-radio MSE constraint.'))
    findings()


def findings():
    lines=['요청별 수신 후 조정 실험: request_projection_v4',
           '기존 목적함수/데이터를 유지하고 최종 합산 연산만 조정하는 탐색 실험.',
           '기울기 1차 조건이며 실제 삭제 보장은 아님. 단일seed 개발 결과, 테스트셋 미사용.']
    p=OUT/'pilot/gate.json'
    if p.exists():
        gate=json.loads(p.read_text())
        lines+=['','이상적 링크 알고리즘 검사 (무선 에너지 순위에 포함하지 않음)',
                '알고리즘 | 라운드 | 보존mR | 보존loss | 평균forgetKL | 최악요청KL']
        for arm,evs in gate['results'].items():
            for k in ['25','40']:
                e=evs[k]
                lines.append(f"{arm} | {k} | {e['retained']['mR']:.6f} | {e['retained']['loss']:.6f} | {e['forget']['KL_retrain_to_candidate']:.6f} | {worst(e):.6f}")
        lines+=['실제 조정 발생: '+str(gate['changed']),'장기 비교 확장 기준 통과: '+str(gate['advance']),
                '세부 기준: '+json.dumps(gate['checks'],ensure_ascii=False)]
        if not gate['advance']:lines+=['이 조건에서 사전 기준을 만족하는 개선이 확인되지 않아 장기 무선 비교를 자동 추가하지 않았음.']
    p=OUT/'summary.json'
    if p.exists():
        rows=json.loads(p.read_text())['methods']
        lines+=['','실제 무선 비교 (누적 비용에 prefix와 proxy 생성 포함)',
                '방식 | 상태 | 라운드 | 보존mR | 평균KL | 최악요청KL | TX상한J | uplink초 | 모든요청목표확정라운드']
        for row in rows:
            x=row.get('last');hit=row.get('all_request_joint_target')
            if not x:lines.append(row['label']+' | 대기');continue
            lines.append(f"{row['label']} | {row['status']} | {x['round']} | {x['retained_mR']:.6f} | {x['forget_KL']:.6f} | {x['worst_request_KL']:.6f} | {x['energy_bound_j']:.6f} | {x['uplink_s']:.6f} | "+str(hit['confirmation']['round'] if hit else '미달/미확정'))
        prop=next(r for r in rows if r['method']=='proposed')
        a=prop.get('all_request_joint_target')
        lines+=['','모든 요청의 삭제/보존 공동 목표 달성 비용 비교:']
        for row in rows:
            if row['method'] in ['proposed','hybrid_no_crossmodal']:continue
            b=row.get('all_request_joint_target')
            if not a or not b:
                lines.append(row['label']+': 두 방식 모두의 목표 달성이 확인되지 않아 비용 순위 미확정.');continue
            aa,bb=a['confirmation'],b['confirmation']
            ej=100*(1-aa['energy_bound_j']/bb['energy_bound_j'])
            ts=100*(1-aa['uplink_s']/bb['uplink_s'])
            lines.append(f"{row['label']} 대비 제안법 TX상한 절감률 {ej:.3f}%, 시간 절감률 {ts:.3f}% (음수는 제안법 불리)")
    lines+=['','실패/미완료/목표미달과 이상적 링크를 에너지 우위로 취급하지 않음.',
            '시간은 uplink 점유시간이며 계산·수신/SIC·다운링크·비행 비용을 제외함.']
    (OUT/'findings_ko.txt').write_text('\n'.join(lines)+'\n')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    claim=(OUT/'queue.lock').open('a')
    fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'started.json').exists(),'Refuse duplicate experiment'
    write(OUT/'protocol.json',protocol());write(OUT/'cpu_checks.json',controller.checks())
    if '--cpu-only' in sys.argv:print('CPU_CHECKS_PASSED');return
    assert json.loads((OUT/'transport_checks.json').read_text())['passed']
    status(stage='queued_waiting_for_existing_gpu_run')
    lock=(ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    # Explicit user priority: the four main arms and all proxies are complete;
    # the remaining old ablation was checkpointed before releasing the GPU.
    pause=json.loads((PRIOR/'priority_pause.json').read_text())
    assert pause['next_version']=='request_projection_v4'
    assert Path(pause['checkpoint']).exists()
    for method in ['proposed','aircomp_only','aircomp_oma','oma_only']:
        assert json.loads((PRIOR/method/'manifest.json').read_text())['status']=='completed_200',method
    for mode in ['ideal','analog','oma']:
        assert (PRIOR/'proxies'/mode/'complete.json').exists(),mode
    import subprocess
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert not gpu,'Another GPU process is running: '+gpu
    write(OUT/'started.json',dict(pid=os.getpid(),unix=time.time()))
    plan=json.loads((SOURCE/'plan.json').read_text())
    for name,digest in plan['sources'].items():assert core.sha(ROOT/name)==digest,name
    for name,digest in plan['frozen_engine_hashes'].items():assert core.sha(core.FROZEN/name)==digest,name
    for name,digest in plan['runner_hashes'].items():assert core.sha(core.LONG/name)==digest,name
    import torch
    from train import Engine
    from dataset import RetrievalData
    status(stage='loading_model')
    engine=Engine(RetrievalData(),dict(plan))
    records=json.loads((SOURCE/'records.json').read_text());forget=records[20]['forget']
    for arm in ARMS:(OUT/'pilot'/arm).mkdir(parents=True,exist_ok=True)
    gate=pilot(engine,plan,records,forget)
    if gate['advance']:
        sys.path.insert(0,str(ROOT/'tools/overnight_energy_v2/proxy_objective_v2'))
        import parallel_heads
        module=core.load_module('request_projection_transport',core.LONG/'radio_partial.py')
        import radio
        radio.optimize_head=parallel_heads.parallel;module.optimize_head=parallel_heads.parallel
        channels=np.load(SOURCE/'channels.npy')
        try:
            for stop in [40,200]:
                for method in METHODS:physical(engine,plan,records,forget,module.communicate,channels,method,stop)
        finally:parallel_heads.close()
    for name,digest in plan['frozen_engine_hashes'].items():assert core.sha(core.FROZEN/name)==digest,name
    write(OUT/'finished.json',dict(unix=time.time(),physical_launched=gate['advance']))
    status(stage='completed' if gate['advance'] else 'completed_pilot_no_measured_benefit',physical_launched=gate['advance'])

if __name__=='__main__':
    try:main()
    except Exception as exc:
        status(stage='failed',error=repr(exc));traceback.print_exc();raise
