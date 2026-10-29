"""Forgetting audit: fixed models, fresh matched retraining, paired FU controls.

No fitting to evaluation outcomes. Exact-aggregation controls retain the original
transaction/request order and are not new physical energy/delay measurements.
"""
import argparse
import copy
import fcntl
import json
import math
from pathlib import Path
import time

import numpy as np
import proposed_pks_report as current
import pks_balance_v1 as matched
import learning_quality_v3 as quality

ROOT, read, write, sha = current.ROOT, current.read, current.write, current.sha
OUT = ROOT/'results/forgetting_quality_v1_20261030'
METHODS, DATASETS = current.METHODS, current.DATASETS
POLICIES = dict(gate_select16='priority', iscc='equal', afl='equal', salun='equal')


def register():
    path = OUT/'protocol.json'
    if path.exists():
        return read(path)
    inputs = {}
    counts = {}
    for ds in DATASETS:
        anchor = current.source('baseline', ds, 'gate_select16')
        plan = read(anchor.parent/'plan.json')
        assert set(plan['forget_probe']) == set(plan['forgotten_images'])
        assert set(plan['forgotten_images']).isdisjoint(plan['validation_images'])
        counts[ds] = dict(forget_videos=len(plan['forgotten_images']),
                         retained_validation_videos=len(plan['validation_images']),
                         gallery_videos=len(plan['gallery_images']))
        for method in METHODS:
            arm = current.source('baseline', ds, method)
            for name in ['plan.json', 'requests.json', 'initial20/final.pt', 'proxy20/final.pt',
                         'feature_cache/features.pt', 'reference200/final.pt']:
                assert sha(arm.parent/name) == sha(anchor.parent/name), (ds, method, name)
                inputs[str((arm.parent/name).relative_to(ROOT))] = sha(arm.parent/name)
            for file in [arm/'final.pt', arm/'request_records.json', *sorted((arm/'transactions').glob('*.json'))]:
                inputs[str(file.relative_to(ROOT))] = sha(file)
    p = dict(registered_unix=time.time(), methods=METHODS, datasets=DATASETS, counts=counts,
        fixed='Current paper Proposed is P-KS/gate_select16. Preserve all four original final200 models. Same 45 deletion requests, 225 FU steps, cached native features, gallery and seed0. No model selection or retuning.',
        cohort='All 247 CapERA / 358 AirScape deleted videos and all their native captions; all 256 retained validation videos in each dataset. Identical complete gallery for every comparison. No selected examples.',
        retrieval='Six recalls R@1/5/10 in video-to-text and text-to-video, averaged as mR. Multiple positive captions accepted. Forget mR alone is not a deletion certificate and its ideal value is not necessarily zero.',
        retraining='Fresh theta0 adapters, unchanged frozen pretrained RemoteCLIP encoders and logit scale. Entire deletion union excluded from every round. Original common reference uses all original reference_records; policy-matched references replay its retained-only rounds1..20 then each original sampler/weighting for21..200. Reproduce saved reference tensors exactly. This is adapter retraining from initialization, NOT backbone pretraining from scratch.',
        matched_policy=POLICIES,
        controls='For each original method, rerun from the same actual warm20 checkpoint through its original FL/FU transaction order with exact aggregation, both with its unchanged FU rule and with FU updates disabled. Deleted data excluded from FL at the same request arrivals in both. No-FU commits are bookkeeping only, not completed deletion procedures. Same LR, seeds, sample policy and 200 FL rounds. These isolate FU conditional on exact transport; do not directly attribute a physical-FU vs exact-no-FU difference solely to FU.',
        transport='Exact controls use the full local update sum and make no energy/delay/feasibility claim. Preserve original scheduled transactions and arrivals; no radio optimization or noise. AFL-OTA original records must have all10 ready and zero staleness. Digital exact-FU controls must reproduce final tensors bitwise. Proposed gate uses its original fixed retention coefficient1; original logs must show no radio repair.',
        association='Compute mean positive-pair cosine, symmetric positive-minus-mean-negative cosine margin, correct-pair probability and ranks for every deleted video. Paired FU-on/off differences averaged over videos. Save before-first/after-last prescribed FU step for each request for both controls. These intervals can contain concurrent FL; only paired controls support an FU intervention comparison.',
        KL='D_KL(P_retrain || P_candidate), symmetric mean over the two retrieval directions and native positive captions. Publish common and policy-matched references together, per-video values and per-request means/worst. Fixed existing0.01 tolerance is a diagnostic, not a certified forgetting threshold.',
        limits='One existing development training seed. Any video bootstrap intervals are conditional on these fitted models, not independent training-run uncertainty. Frozen pretrained encoders may retain prior knowledge; no raw-data/cache erasure or formal unlearning guarantee. C9 controls aggregation perturbation relative to error-free updates, not equivalence to retraining.',
        input_sha256=inputs,
        references=[dict(title='CLIPErase, ACL2025',url='https://aclanthology.org/2025.acl-long.1469/'),
                    dict(title='SalUn, ICLR2024',url='https://openreview.net/forum?id=gn0mIhQGNM')])
    write(path,p)
    return p


def runtime(ds):
    import torch
    quality.infra.lib.infra.extra.Data = quality.infra.lib.infra.VideoData
    event, base = quality.infra.lib.infra.prior.load_base()
    base.repeat.clone_function = quality.infra.cached_clone
    import accounting
    from prepare_extra import native_engine, native_base
    quality.infra.lib.infra.ORIGINAL_INSTRUMENTED = accounting.instrumented_engine
    data = quality.infra.lib.infra.VideoData(ds) if ds == 'AirScape' else event.recover.Data(ds)
    if ds == 'AirScape':
        base = native_base(base, 1)
    parent = native_engine(base, len(data.caption_ids[0])) if ds == 'AirScape' else base.Engine
    anchor = current.source('baseline',ds,'gate_select16')
    plan = read(anchor.parent/'plan.json')
    payload = torch.load(anchor.parent/'feature_cache/features.pt',map_location='cpu',weights_only=False)
    return base, data, parent, plan, payload, anchor


def configure(method, base):
    import learning_selection_fix_v2 as fix
    if method == 'gate_select16':
        quality.CONFIG = quality.VARIANTS['gate_select16']
        quality.ACTIVE = 'gate_select16'
        quality.infra.lib.ACTIVE = quality.PROPOSED
        adaptive = quality.infra.lib.adaptive
        if not getattr(adaptive,'forgetting_audit_gate_installed',False):
            adaptive.request_gradient = quality.infra.cached_clone(adaptive.request_gradient,[
                ('weight, smooth, signal = feedback(old_weight,old_ema,dr,df)',
                 'weight, smooth, signal = 1., 0., 0.',1),
                ('3*fk+3*ck','3*torch.relu(fk-.01)+3*ck',1),
                ('objective=float((3*fk+weight*rm+3*ck).detach())',
                 'objective=float((3*torch.relu(fk-.01)+weight*rm+3*ck).detach())',1)],{})
            adaptive.forgetting_audit_gate_installed = True
        return fix.vectors(base,'cliperase')
    quality.CONFIG = quality.VARIANTS['salun'] if method=='salun' else dict(quality.VARIANTS['gaft'],equal=True)
    quality.ACTIVE = method
    quality.infra.lib.ACTIVE = 'fedavg_clip_gaft'
    return quality.vectors(base,'cliperase')


def per_video(engine,theta,plan):
    """All directions and captions, float64 reductions for stable diagnostics."""
    import torch
    with torch.no_grad():
        logits = engine.score(theta,plan['gallery_images']).double().cpu()
        n = len(plan['gallery_images']); c = len(engine.data.caption_ids[0])
        assert logits.shape == (n,n*c)
        indices = torch.arange(n)
        owner = indices.repeat_interleave(c)
        ci = indices[:,None]*c+torch.arange(c)
        logi, logt = logits.log_softmax(1), logits.T.log_softmax(1)
        ri, rt = logits.argsort(1,descending=True), logits.T.argsort(1,descending=True)
        metrics = {}
        for k in [1,5,10]:
            metrics['i2t_R'+str(k)] = (owner[ri[:,:k]]==indices[:,None]).any(1).double().numpy()
            metrics['t2i_R'+str(k)] = (rt[:,:k]==owner[:,None]).any(1).double().reshape(n,c).mean(1).numpy()
        metrics['mR'] = np.mean(list(metrics.values()),axis=0)
        metrics['loss'] = (-torch.logsumexp(logi[indices[:,None],ci],1)-logt[ci,indices[:,None]].mean(1)).numpy()/2
        cosine = logits/float(theta[-1].exp())
        positive = cosine[indices[:,None],ci].mean(1)
        negative_i = (cosine.sum(1)-positive*c)/(n*c-c)
        negative_t = ((cosine.sum(0)[ci]-cosine[indices[:,None],ci])/(n-1)).mean(1)
        metrics['positive_cosine'] = positive.numpy()
        metrics['association_margin'] = (positive-(negative_i+negative_t)/2).numpy()
        metrics['correct_pair_probability'] = .5*(logi[indices[:,None],ci].exp().sum(1)+logt[ci,indices[:,None]].exp().mean(1)).numpy()
        ranks = rt.argsort(1)+1
        metrics['t2i_mean_rank'] = ranks[ci,indices[:,None]].double().mean(1).numpy()
    return metrics,(logi,logt)


def kl_per_video(reference,candidate,count):
    import torch
    oi,ot=reference;ni,nt=candidate
    a=(oi.exp()*(oi-ni)).sum(1)
    b=(ot.exp()*(ot-nt)).sum(1).reshape(-1,count).mean(1)
    return torch.clamp((a+b)/2,min=0).numpy()


def indices(plan,group):
    values = plan['forgotten_images'] if group=='forget' else plan['validation_images']
    return np.array([plan['gallery_images'].index(v) for v in values])


def summarize(values,plan):
    return {g:{k:float(np.mean(v[indices(plan,g)])) for k,v in values.items()} for g in ['forget','retained']}


def store_eval(engine,theta,plan,dest,refs=None):
    values,prob = per_video(engine,theta,plan)
    if refs:
        count=len(engine.data.caption_ids[0])
        for name,ref in refs.items():
            values[name+'_KL']=kl_per_video(ref,prob,count)
    dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'per_video.npz',video_ids=np.array(plan['gallery_images']),**values)
    summary=summarize(values,plan)
    write(dest/'metrics.json',summary)
    return summary,prob


def retrain(ds):
    import torch
    from train import update_theta
    base,data,parent,plan,payload,anchor=runtime(ds)
    records=read((anchor.parent/'reference200').resolve().parent/'reference_records.json')
    assert len(records)==200
    deleted=set(plan['forgotten_images'])
    pools=[sorted(set(p)-deleted) for p in plan['partitions']]
    total=sum(map(len,pools)); rows=[]
    for policy in ['common','priority','equal']:
        dest=OUT/ds.lower()/'references'/policy
        if (dest/'finished.json').exists():
            rows.append(read(dest/'finished.json'));continue
        matched.initialize_variant('pk_uniform')
        engine=matched.engine_wrapper(parent)(data,plan,payload)
        theta=engine.theta0.clone()
        t0=time.time()
        for k in range(1,201):
            engine.start_work()
            if policy=='common' or k<=20:
                rec=records[k-1]
                assert all(not(set(b)&deleted) for b in rec['batches'])
                engine.plan['learning_rate']=base.core.lr_at(k)
                l,_,_=engine.updates(theta,rec,k,'retrain',[[] for _ in range(10)])
            else:
                matched.initialize_variant('control' if policy=='priority' else 'pk_uniform')
                l=np.zeros((10,len(theta)-1))
                for user,pool in enumerate(pools):
                    batch=matched.choose_batch(engine,theta,pool,k,user,np.random.default_rng(730000+k*10+user))
                    assert not set(batch)&deleted
                    g,_=engine.gradient(theta,batch,k)
                    weight=len(pool)/total if policy=='priority' else 1/10
                    l[user]=(-base.core.lr_at(k)*weight*g[:-1]).cpu().numpy()
            theta=update_theta(theta,l.sum(0));engine.finish_work()
        expected=anchor.parent/'reference200/final.pt' if policy=='common' else ROOT/'results/pks_evaluation_v1_20261028'/(ds.lower()+'_seed0')/'references'/policy/'200.pt'
        equal=torch.equal(theta,base.core.load_theta(expected,engine.device))
        assert equal,(ds,policy,'fresh retrain must reproduce frozen reference')
        base.core.save_theta(dest/'final.pt',theta,200)
        summary,_=store_eval(engine,theta,plan,dest)
        result=dict(dataset=ds,policy=policy,fresh_rounds=200,wall_s=time.time()-t0,
                    original_reference_tensor_exact=equal,checkpoint_sha256=sha(dest/'final.pt'),metrics=summary)
        write(dest/'finished.json',result);rows.append(result)
        print(dict(stage='retrain',dataset=ds,policy=policy,wall_s=round(time.time()-t0,1),exact=True),flush=True)
    return rows


def control(ds,method,fu):
    import torch
    from train import update_theta
    base,data,parent,plan,payload,anchor=runtime(ds)
    arm=current.source('baseline',ds,method)
    name='exact_fu' if fu else 'exact_no_fu'
    dest=OUT/ds.lower()/method/name
    if (dest/'finished.json').exists():return read(dest/'finished.json')
    dest.mkdir(parents=True,exist_ok=True)
    write(dest/'started.json',dict(unix=time.time(),source_sha256=sha(Path(__file__))))
    vectors=configure(method,base)
    engine=quality.engine_wrapper(parent)(data,plan,payload)
    warm=base.core.load_theta(arm.parent/'initial20/final.pt',engine.device)
    engine.set_teacher(warm);theta=warm.clone()
    proxy=base.core.Teacher(engine,base.core.load_theta(arm.parent/'proxy20/final.pt',engine.device))
    queue=base.RoundQueue(read(arm.parent/'requests.json'));teachers={}
    txs=[read(p) for p in sorted((arm/'transactions').glob('*.json'))]
    fl=0;t0=time.time();curves=[];events={};fu_count=0;finalprob=None
    for record in txs:
        start,end=record['start_s'],record['end_s'];kind=record['kind'];ids=record['FU_request_ids']
        queue.sync(fl,start,0.)
        new=[rid for rid in ids if queue.items[rid]['steps']==0]
        if new:
            before,_=per_video(engine,theta,plan)
            for rid in new:
                req=queue.items[rid];idx=[plan['gallery_images'].index(i) for i in req['images']]
                events[rid]=dict(request=rid,user=req['user'],videos=len(idx),first_FL_round=20+fl,
                    before={k:float(v[idx].mean()) for k,v in before.items()})
        engine.start_work()
        if method=='gate_select16':
            assert not record['radio'].get('adaptive_radio_repaired',False)
            l,u,stats=vectors(engine,theta,plan,queue,start,fl,kind,ids if fu else [],proxy,teachers,False)
            for item in engine.adaptive_pending.values():
                teacher=item['teacher'];teacher.adaptive_weight=item['desired'];teacher.adaptive_ema=item['ema'];teacher.adaptive_initial_forget=item['initial_forget']
            update=(l+u).sum(0)
        elif kind=='fl':
            if method=='afl':
                assert len(record['radio']['selected_job_ids'])==10
                assert not any(record['radio']['staleness'].values())
            pools=base.clip.stream.retained_pools(plan,queue,start);k=21+fl;gradients=[]
            for user,pool in enumerate(pools):
                batch=np.random.default_rng(730000+k*10+user).choice(pool,min(16,len(pool)),replace=False).astype(int).tolist()
                g,_=engine.gradient(theta,batch,k)
                gradients.append(g[:-1].detach().cpu().numpy().copy())
            update=np.stack([-base.core.lr_at(k)*g/10 for g in gradients]).sum(0)
        else:
            _,u,stats=vectors(engine,theta,plan,queue,start,fl,'fu',ids if fu else [],None,{},False)
            update=u.sum(0)
        theta=update_theta(theta,update);engine.finish_work()
        queue.commit(ids,start,end);fu_count+=len(ids) if fu else 0
        fl+=int(kind in ['fl','joint'])
        done=[rid for rid in ids if queue.items[rid]['steps']==queue.items[rid]['required_steps']]
        if done:
            after,_=per_video(engine,theta,plan)
            for rid in done:
                idx=[plan['gallery_images'].index(i) for i in queue.items[rid]['images']]
                events[rid].update(last_FL_round=20+fl,after={k:float(v[idx].mean()) for k,v in after.items()})
        if (fl%20==0 and kind in ['fl','joint']) or record is txs[-1]:
            vals,finalprob=per_video(engine,theta,plan)
            curves.append(dict(FL_round=20+fl,FU_steps=fu_count,metrics=summarize(vals,plan)))
        assert 20+fl==record['completed_FL_rounds']
    assert fl==180 and len(events)==45 and all('after' in r for r in events.values())
    assert fu_count==(225 if fu else 0)
    exact=None
    if fu and method in ['iscc','salun']:
        exact=torch.equal(theta,base.core.load_theta(arm/'final.pt',engine.device))
        assert exact,(ds,method,'digital control must reproduce original final tensor')
    refs={}
    for key,policy in [('common','common'),('matched',POLICIES[method])]:
        ref=base.core.load_theta(OUT/ds.lower()/'references'/policy/'final.pt',engine.device)
        _,refs[key]=per_video(engine,ref,plan)
    base.core.save_theta(dest/'final.pt',theta,200)
    summary,_=store_eval(engine,theta,plan,dest,refs)
    write(dest/'request_events.json',list(events.values()));write(dest/'curves.json',curves)
    result=dict(dataset=ds,method=method,condition=name,FL_rounds=200,FU_steps=fu_count,
                wall_s=time.time()-t0,metrics=summary,digital_original_tensor_exact=exact,
                checkpoint_sha256=sha(dest/'final.pt'))
    write(dest/'finished.json',result)
    print(dict(stage=name,dataset=ds,method=method,wall_s=round(time.time()-t0,1),forget_mR=100*summary['forget']['mR']),flush=True)
    return result


def evaluate(ds):
    import torch
    base,data,parent,plan,payload,anchor=runtime(ds)
    engine=parent(data,plan,payload)
    refprobs={}
    for policy in ['common','priority','equal']:
        theta=base.core.load_theta(OUT/ds.lower()/'references'/policy/'final.pt',engine.device)
        _,refprobs[policy]=per_video(engine,theta,plan)
    rows=[]
    for method in METHODS:
        arm=current.source('baseline',ds,method)
        theta=base.core.load_theta(arm/'final.pt',engine.device)
        exact=base.core.load_theta(OUT/ds.lower()/method/'exact_fu/final.pt',engine.device)
        _,exactprob=per_video(engine,exact,plan)
        refs={'common':refprobs['common'],'matched':refprobs[POLICIES[method]],'exact_FU_to_wireless':exactprob}
        summary,prob=store_eval(engine,theta,plan,OUT/ds.lower()/method/'actual',refs)
        expected=read(arm/'service_completion.json')['evaluation']
        for cohort in ['forget','retained']:
            for metric in ['mR','loss','i2t_R1','i2t_R5','i2t_R10','t2i_R1','t2i_R5','t2i_R10']:
                assert math.isclose(summary[cohort][metric],expected[cohort][metric],abs_tol=2e-6),(ds,method,metric)
        assert math.isclose(summary['forget']['common_KL'],expected['forget']['KL_retrain_to_candidate'],abs_tol=2e-7)
        per=np.load(OUT/ds.lower()/method/'actual/per_video.npz')
        requests=[]
        for req in read(arm/'request_records.json'):
            idx=[plan['gallery_images'].index(i) for i in req['images']]
            requests.append(dict(request=req['id'],videos=len(idx),common_KL=float(per['common_KL'][idx].mean()),
                                 matched_KL=float(per['matched_KL'][idx].mean())))
        write(OUT/ds.lower()/method/'actual/request_KL.json',requests)
        rows.append(dict(dataset=ds,method=method,label=METHODS[method],metrics=summary,
                    worst_request_common_KL=max(x['common_KL'] for x in requests),
                    worst_request_matched_KL=max(x['matched_KL'] for x in requests),
                    original_metrics_reproduced=True))
    for name,theta in [('initialization',engine.theta0),('warm20',base.core.load_theta(anchor.parent/'initial20/final.pt',engine.device))]:
        store_eval(engine,theta,plan,OUT/ds.lower()/name,{'common':refprobs['common']})
    write(OUT/ds.lower()/'summary.json',rows)
    print(dict(stage='evaluated_frozen_models',dataset=ds,original_metrics_verified=True),flush=True)
    return rows


def run(ds):
    import torch
    register();assert torch.cuda.is_available();torch.set_num_threads(1)
    with (ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        # Load dependencies before importing dynamically discovered train module.
        runtime(ds)
        refs=retrain(ds)
        controls=[]
        for method in METHODS:
            for fu in [False,True]:controls.append(control(ds,method,fu))
        rows=evaluate(ds)
        write(OUT/ds.lower()/'finished.json',dict(passed=True,reference_runs=len(refs),control_runs=len(controls),
                frozen_models_evaluated=len(rows),source_sha256=sha(Path(__file__)),finished_unix=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['register','run'])
    parser.add_argument('--dataset',choices=DATASETS);args=parser.parse_args()
    if args.action=='register':register()
    else:
        for dataset in [args.dataset] if args.dataset else DATASETS:run(dataset)
