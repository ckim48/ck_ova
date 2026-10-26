"""CPU post-processing of completed ledgers/probabilities; never trains/tunes."""
from pathlib import Path
import os,json,time,hashlib,math
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'results/overnight_energy_v2/full_system_baselines_v1'
DEST=BASE/'fedmemprune_rho040'
REF=BASE/'fedmemprune_matched_reference'
OUT=ROOT/'results/overnight_energy_v2/final_metrics_audit_v1'
SOURCE=ROOT/'results/overnight_energy_v2/retrieval_long200_v3'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,d):
    OUT.mkdir(parents=True,exist_ok=True);p=OUT/name;t=p.with_suffix('.tmp')
    t.write_text(json.dumps(d,indent=2,allow_nan=False));t.replace(p)
def probabilities(p):
    z=np.load(p);a=[z['log_image'].astype(np.float64),z['log_text'].astype(np.float64)]
    for x in a:
        assert np.isfinite(x).all() and np.max(np.abs(np.exp(x).sum(1)-1))<2e-5
    return a
def quality(candidate,reference,indices):
    ix=np.asarray(indices,dtype=int);ci=(5*ix[:,None]+np.arange(5)).reshape(-1)
    out={};ks=[]
    for name,idx,c,r in zip(['i2t','t2i'],[ix,ci],candidate,reference):
        k=float((np.exp(r[idx])*(r[idx]-c[idx])).sum(1).mean())
        assert k>=-2e-5
        ks.append(max(0.,k));out[name+'_top1_disagreement']=float((r[idx].argmax(1)!=c[idx].argmax(1)).mean())
    return dict(images=len(ix),KL_retrain_to_candidate=sum(ks)/2,**out)
def auc(scores,n):
    a=scores[:n,None];b=scores[None,n:]
    return float(((a>b)+.5*(a==b)).mean())
def target(rows):
    consecutive=[]
    for r in rows:
        ok=(r['round']>=25 and r['retained']['mR']>=.29304036883016427
            and r['forget']['KL_retrain_to_candidate']<=.01 and r['worst_request_KL']<=.01
            and r['requests_handled'])
        if consecutive and r['round']!=consecutive[-1]['round']+5:consecutive=[]
        consecutive=consecutive+[r] if ok else []
        if len(consecutive)==3:
            return dict(entry_round=consecutive[0]['round'],confirmation_round=r['round'],
                        cumulative_tx_energy_bound_j=r['energy_j'],cumulative_uplink_s=r['uplink_s'])
    return None
def self_check():
    a=np.log(np.array([[.2,.8],[.6,.4]]));b=np.tile(a,(5,1))
    assert quality([a,b],[a,b],[0,1])['KL_retrain_to_candidate']==0
    assert auc(np.array([3.,4.,1.,2.]),2)==1 and auc(np.zeros(4),2)==.5
    r=dict(retained={'mR':.3},forget={'KL_retrain_to_candidate':0.},worst_request_KL=0.,
           requests_handled=True,energy_j=1.,uplink_s=2.)
    assert target([dict(r,round=k) for k in [25,30,35]])['confirmation_round']==35
    assert target([dict(r,round=k) for k in [25,35,40]]) is None
    bad=[dict(r,round=k) for k in [25,30,35]];bad[1]['worst_request_KL']=.02
    assert target(bad) is None
    return dict(identity_KL=True,AUC_ties_and_order=True,three_consecutive_five_round_checks=True,worst_request_gate=True)


def main():
    tests=self_check();plan=read(SOURCE/'plan.json');recs=read(SOURCE/'records.json')
    prefix=read(DEST/'prefix_rounds.json');paths=sorted((DEST/'rounds').glob('*.json'))
    rows=[read(p) for p in paths];allrows=prefix+rows
    assert [r['round'] for r in allrows]==list(range(1,len(allrows)+1))
    energy=delay=0.;max_ediff=max_tdiff=0.;steps=0
    for r in allrows:
        x=r['radio'];assert x['status']=='feasible'
        energy+=x['energy_bound_j'];delay+=x['uplink_s']
        max_ediff=max(max_ediff,abs(energy-r['cumulative_tx_energy_bound_j']))
        max_tdiff=max(max_tdiff,abs(delay-r['cumulative_uplink_s']))
        steps+=sum(x.get('steps',0) for x in r.get('local',[]))
        assert x['maximum_digital_power_w']<=plan['config']['power_limit_w']*(1+1e-5)
        assert x['error_bound']<=x['epsilon']*(1+1e-5)
        assert x['rate_margin_bits']>=-1e-7
    assert max_ediff<1e-8 and max_tdiff<1e-8
    forget=recs[20]['forget'];forbidden=set(plan['forgotten_images'])
    requested={u:set(f) for u,f in enumerate(forget) if f}
    assert set().union(*requested.values())==forbidden
    assert forbidden.isdisjoint(plan['validation_images']+plan['official_test_images'])
    for rec in recs[20:]:assert all(forbidden.isdisjoint(ids) for ids in rec['retained'])
    reset=read(DEST/'pruning.json');assert reset['selected_count']==int(.4*20480)
    handled=all(r['radio']['status']=='feasible' for r in rows) and any(r['round']==21 for r in rows)
    complete=(DEST/'finished.json').exists() and read(DEST/'finished.json').get('rounds')==200 and len(allrows)==200
    reference_ready=(REF/'finished.json').exists() and read(REF/'finished.json').get('rounds')==200
    position={i:j for j,i in enumerate(plan['gallery_images'])}
    groups={str(u):[position[i] for i in sorted(f&set(plan['forget_probe']))] for u,f in requested.items()}
    assert all(groups.values()),'Some requests have no evaluation samples'
    groups={'forget':[position[i] for i in plan['forget_probe']],**groups}
    final_ref=probabilities(REF/'probabilities_200.npz') if reference_ready else None
    metrics=[];horizon=[]
    for r in rows:
        k=r['round'];p=DEST/f'probabilities_{k:03d}.npz'
        if k%5 or 'evaluation' not in r or not p.exists():continue
        cp=probabilities(p)
        bases=[]
        if final_ref is not None:bases.append(('final',final_ref))
        hp=REF/f'probabilities_{k:03d}.npz'
        if hp.exists() and (REF/'rounds'/f'{k:03d}.json').exists():bases.append(('horizon',probabilities(hp)))
        for name,ref in bases:
            q={g:quality(cp,ref,ix) for g,ix in groups.items()}
            rr=dict(round=k,forget=q.pop('forget'),per_request=q,
                worst_request_KL=max(x['KL_retrain_to_candidate'] for x in q.values()),
                retained={v:r['evaluation']['retained'][v] for v in ['mR','loss']},
                requests_handled=handled,energy_j=r['cumulative_tx_energy_bound_j'],uplink_s=r['cumulative_uplink_s'])
            (metrics if name=='final' else horizon).append(rr)
    mia=None
    if complete and reference_ready:
        n=len(plan['mia_member_images']);m=len(plan['mia_nonmember_images'])
        a=np.load(DEST/'mia_scores.npy');b=np.load(REF/'mia_scores.npy')
        assert a.shape==b.shape==(n+m,) and np.isfinite(a).all() and np.isfinite(b).all()
        ca=auc(a,n);ra=auc(b,n);rng=np.random.default_rng(823);draws=[]
        assert abs(ca-read(DEST/'mia.json')['loss_score_auc'])<1e-7
        assert abs(ra-read(REF/'mia.json')['loss_score_auc'])<1e-7
        for _ in range(1000):
            ids=np.r_[rng.integers(n,size=n),n+rng.integers(m,size=m)]
            aa=auc(a[ids],n);bb=auc(b[ids],n)
            draws.append(max(aa,1-aa)-max(bb,1-bb))
        mia=dict(candidate_auc=ca,reference_auc=ra,candidate_orientation_free_auc=max(ca,1-ca),
            reference_orientation_free_auc=max(ra,1-ra),raw_auc_gap=ca-ra,
            orientation_free_auc_gap=max(ca,1-ca)-max(ra,1-ra),
            paired_image_bootstrap_95_interval=np.quantile(draws,[.025,.975]).tolist(),members=n,nonmembers=m,
            caveat='One fixed loss-score attack; interval reflects these probe images, not training-seed uncertainty or a deletion guarantee.')
    latest=next((r for r in reversed(rows) if 'evaluation' in r),None)
    report=dict(unix=time.time(),method='FedMemPrune VLM/OFDMA adaptation',training_complete=complete,
        reference_complete=reference_ready,completed_rounds=len(allrows),planned_rounds=200,
        requests=dict(total=len(requested),excluded_from_recovery=len(requested),reset_transaction_completed=handled,
                      erasure_quality_evaluated=reference_ready,successful_erasure_not_inferred_from_processing=True),
        cost=dict(total_tx_energy_bound_j=energy,uplink_occupation_s=delay,prefix_tx_energy_bound_j=prefix[-1]['cumulative_tx_energy_bound_j'],
                  prefix_uplink_s=prefix[-1]['cumulative_uplink_s'],post_request_local_optimizer_steps=steps,
                  local_compute_and_receiver_time_excluded=True,max_energy_ledger_difference=max_ediff,max_time_ledger_difference=max_tdiff),
        final_matched_reference=metrics,matched_horizon_reference=horizon,
        joint_target=target(metrics) if complete and reference_ready else None,target_status='assessed' if complete and reference_ready else 'pending',
        MIA=mia,latest_utility=None if latest is None else dict(round=latest['round'],mR=latest['evaluation']['retained']['mR'],loss=latest['evaluation']['retained']['loss']),
        checks=dict(**tests,contiguous_ledgers=True,stored_power_rate_error_constraints=True,deleted_data_excluded=True),
        limitations=['Different local training work from one-step SGD proposal; same round is not equal computation.',
          'Reference uses ideal radio and optimizer/aggregation matching; not an ensemble of retraining seeds or exact deletion certificate.',
          'Common-reference KL is not used here to declare external-method failure. No setting is tuned from these outputs.',
          'Missing or unfinished references stay pending, never zero-KL or zero-energy successful results.'],
        hashes={str(p.relative_to(ROOT)):sha(p) for p in [SOURCE/'plan.json',SOURCE/'records.json',DEST/'protocol.json',Path(__file__)]})
    write('report.json',report)
    print(json.dumps({k:report[k] for k in ['training_complete','reference_complete','completed_rounds','cost','target_status','latest_utility','checks']}),flush=True)

if __name__=='__main__':main()
