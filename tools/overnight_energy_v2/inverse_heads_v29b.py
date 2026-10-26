"""Enumerate every head for all61 fixed-block-budget refinement cases."""
import argparse
from datetime import datetime,timezone,timedelta
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--plan-only',action='store_true')
    a=p.parse_args();root=a.project.resolve();f=root/'results/overnight_energy_v2';prior=f/'inverse_budget_v29'
    sys.path[:0]=[str(root),str(root/'tools/overnight_energy_v2/inverse_budget_v29')]
    from solver import refine
    from certificate import check
    from wireless import quantize
    assert json.loads((prior/'status.json').read_text())['status']=='completed_fixed_block_budget_convex_refinement_no_gpu_promotion'
    registered=json.loads((prior/'plan.json').read_text());seeds=json.loads((prior/'results.json').read_text());assert len(seeds)==registered['cases']==61
    for n,h in registered['code_sha256'].items():assert sha(root/'tools/overnight_energy_v2/inverse_budget_v29'/n)==h
    for n,h in registered['core_sha256'].items():assert sha(root/n)==h
    source=f/'public_history_fu_v19/screen';cfg=json.loads((source/'plan.json').read_text())['config'];epsilon=registered['epsilon']
    with np.load(source/'vectors.npz') as d:
        learning,unlearning,channels=[d[k].copy() for k in ['learning','unlearning','channels']]
        bases={k:d[f'rank{k}_basis'].copy() for k in [1,3]};public_scale=float(d['public_scale'])
    assert sha(source/'vectors.npz')==registered['input_vector_sha256']
    width=cfg['block_dimension'];dimension=learning.shape[1]
    block=lambda x:np.pad(x,((0,0),(0,(-dimension)%width))).reshape(len(x),-1,width)
    out=f/'inverse_heads_v29b';out.mkdir(exist_ok=True)
    def write(n,v):
        target=out/n;tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(target)
    plan=dict(version='inverse_heads_v29b',code_sha256=sha(__file__),prior_plan_sha256=sha(prior/'plan.json'),prior_results_sha256=sha(prior/'results.json'),
        prior_code_sha256=registered['code_sha256'],core_sha256=registered['core_sha256'],cases=61,heads_per_case=10,max_head_solves=610,epsilon=epsilon,
        limits=['Same61 completedoriginal0.2 case block-budget vectors, each enumerates all10heads with direct convex TX/RX solve. No favorable-method-only refinement.',
            'Hold codec/history/error-report option, fixedbit setting and phase fraction/boundary rule. Recompute head-owned payloads, exact paidcontroltime/RF, physical symbols/rates and individual batteries for eachhead.',
            'Original fixedblockbudget vectors shared across head candidates within a case. This is discretehead+convexinner optimization for these budgets, not global optimization over blockbudget allocation or fulltrajectory.',
            'Allselected61 V29seed incumbents remain fallbacks; allsuccessful candidates receive independent physical certificate. Newheadsolvercompute/circuit/header energy unmeasured.',
            'No task/test/model inference/GPU/currentqueue/source changes orautomaticpromotion. No newheadsolve after noon; partialsweep notarchitecture ranking.'])
    if (out/'plan.json').exists():assert json.loads((out/'plan.json').read_text())==plan
    else:write('plan.json',plan)
    if a.plan_only:print(json.dumps(dict(cases=61,heads=610)));return
    assert not (out/'results.json').exists()
    deadline=datetime(2026,10,21,12,tzinfo=timezone(timedelta(hours=9)));rows=[]
    for i,(case,previous) in enumerate(zip(registered['case_inputs'],seeds)):
        assert previous['case']==i and previous['status']=='certified_nonincreasing_design'
        raw=learning+unlearning if case['mode']=='mixed_aircomp' else learning
        if case['rank']:
            basis=bases[case['rank']];coeff=((raw@basis)/public_scale).astype(np.float16).astype(float)*public_scale;raw=raw-coeff@basis.T
        analog=block(raw);digital=block(np.zeros_like(unlearning) if case['mode']=='mixed_aircomp' else unlearning)
        decoded,delta,payload=quantize(digital,case['bits'],cfg['scale_metadata_bits'])
        if case['reports']:
            actual=np.linalg.norm(decoded-digital,axis=-1);r=actual.astype(np.float32);delta=np.where(r.astype(float)<actual,np.nextafter(r,np.float32(np.inf)),r).astype(float)
        rho=np.linalg.norm(analog,axis=-1);target_budget=np.asarray(previous['seed_certificate']['block_error_bound'])
        best_path=root/previous['design_path'];assert sha(best_path)==previous['design_sha256']
        with np.load(best_path) as d:best={k:d[k].copy() for k in d.files}
        bestcert=previous['candidate_certificate'];heads=[];started=time.monotonic()
        opts=dict(reports=case['reports'],rank=case['rank'],basis=bases.get(case['rank']),public_scale=public_scale)
        for head in range(cfg['users']):
            if datetime.now(timezone(timedelta(hours=9)))>=deadline:
                write('status.json',dict(status='partial_noon_cutoff_no_ranking',completed_cases=len(rows),case=i,head=head));return
            write('status.json',dict(status='solving_convex_head',case=i,total_cases=61,head=head,mode=case['mode'],at_kst=datetime.now().isoformat(timespec='seconds')))
            users=np.delete(np.arange(cfg['users']),head);h=channels[users,head,None]*np.ones((1,len(target_budget)));reporters=np.any(digital[users]!=0,axis=(1,2))
            ncontrol=64*len(target_budget) if case['reports'] else 32*case['rank']
            if case['rank']:reporters=np.ones(len(users),dtype=bool)
            pc=np.where(reporters,(2**.5-1)/channels[users,head]**2,0.) if ncontrol else np.zeros(len(users));ec=pc*ncontrol/cfg['symbol_rate_hz']
            control_total=int(reporters.sum())*ncontrol;total=np.full(len(target_budget),width*cfg['symbols_per_coordinate'])
            debit=np.full(len(total),control_total//len(total));debit[:control_total%len(total)]+=1;main=total-debit
            na=main.copy();nd=main.copy();active=np.any(payload[users]>0,axis=0);fraction=case['row']['analog_fraction']
            if case['mode']=='phase_grid':
                if fraction=='minimum_dimension_boundary':na=np.where(active,width,main);nd=np.where(active,main-width,main)
                else:na=np.where(active,np.floor(main*fraction).astype(int),main);nd=np.where(active,main-na,main)
            item=dict(head=head)
            if np.any(na<width) or np.any(nd<=0) or np.max(pc,initial=0)>cfg['power_limit_w']*(1+1e-6):
                item['status']='necessary_symbol_or_controlpower_failed';heads.append(item);continue
            ds=delta[users].sum(axis=0);rs=rho[users].sum(axis=0);noise=np.sqrt(width/(2*na))
            zero=target_budget>=rs+ds
            if np.any((target_budget<=ds)&~zero):item['status']='fixed_block_quantization_floor_failed';heads.append(item);continue
            receiver=np.where(zero,0.,np.maximum(target_budget-ds,0)/noise)
            amplitude=np.divide(rho[users],h*receiver,out=np.zeros_like(h),where=(h*receiver)>0)
            seed=dict(head=np.asarray(head),users=users,analog_rms=amplitude,digital_sqrt_power=np.zeros_like(amplitude),receiver=receiver,
                normalized_channels=h,rho=rho[users],quantization_bound=delta[users],symbols=na,dimensions=np.full(len(na),width),chi=np.asarray(0.),
                mode=np.asarray('orthogonal' if case['mode']=='phase_grid' else case['mode']),digital_symbols=nd,control_power=pc,control_energy=ec)
            try:
                new,stats=refine(seed,payload[users],cfg)
                cert=check(learning,unlearning,channels,cfg,case['bits'],case['mode'],epsilon,new,**opts)
                assert np.all(np.asarray(cert['block_error_bound'])<=target_budget+np.maximum(1e-12,target_budget*1e-5))
                item.update(status='independently_certified_feasible',energy_j=cert['energy_upper_j'],solver_status=stats['status'])
                if cert['energy_upper_j']<bestcert['energy_upper_j']:best,bestcert=new,cert
            except Exception as exc:item.update(status='no_certified_design_from_convex_solve',error_type=type(exc).__name__,error=str(exc))
            heads.append(item)
        best['energy_j']=np.asarray(bestcert['energy_upper_j']-bestcert['extra_control_energy_j']);best['error_bound']=np.asarray(bestcert['root_error_bound'])
        path=out/f'case_{i:03d}_design.npz';np.savez_compressed(path,**best)
        row=dict(case=i,family=case['family'],mode=case['mode'],rank=case['rank'],bits=case['bits'],reports=case['reports'],analog_fraction=fraction,
            baseline_v29_energy_j=previous['refined_energy_j'],refined_energy_j=bestcert['energy_upper_j'],head=int(best['head']),heads=heads,
            design_path=str(path.relative_to(root)),design_sha256=sha(path),certificate=bestcert,original_block_budget=target_budget.tolist(),seconds=time.monotonic()-started)
        rows.append(row);write('results.json',rows)
        print('CASE',json.dumps({k:v for k,v in row.items() if k not in ['heads','certificate','original_block_budget']}),flush=True)
    selected=[min([r for r in rows if r['mode']==mode],key=lambda r:r['refined_energy_j']) for mode in ['hybrid','mixed_aircomp','phase_grid']]
    write('summary.json',dict(selected=selected,cases=len(rows),head_candidates=sum(len(r['heads']) for r in rows),
        certified_heads=sum(x['status']=='independently_certified_feasible' for r in rows for x in r['heads']),limits=plan['limits']))
    write('status.json',dict(status='completed61_budget_vectors_all610head_checks_no_gpu_promotion',cases=len(rows),at_kst=datetime.now().isoformat(timespec='seconds')))


if __name__=='__main__':main()
