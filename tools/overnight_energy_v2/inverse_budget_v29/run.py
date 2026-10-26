"""Systematic paired refinement of all saved original0.2 FU61 incumbents."""
import argparse
from datetime import datetime,timezone,timedelta
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from solver import refine
from certificate import check


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--plan-only',action='store_true')
    args=p.parse_args();root=args.project.resolve();family=root/'results/overnight_energy_v2';source=family/'public_history_fu_v19/screen'
    parent=json.loads((source/'plan.json').read_text());cfg=parent['config'];assert all(sha(root/n)==v for n,v in parent['core_sha256'].items())
    with np.load(source/'vectors.npz') as data:
        learning,unlearning,channels=[data[k].copy() for k in ['learning','unlearning','channels']]
        bases={k:data[f'rank{k}_basis'].copy() for k in [1,3]};public_scale=float(data['public_scale'])
    dimension=learning.shape[1];width=cfg['block_dimension'];block=lambda x:np.pad(x,((0,0),(0,(-dimension)%width))).reshape(len(x),-1,width)
    epsilon=float(.2*np.linalg.norm(np.linalg.norm(block(learning),axis=-1).sum(axis=0)))
    families=['public_history_fu_v19','public_history_fu_v19b','fu_precision_v24','certified_error_v27'];cases=[];inputs={}
    for name in families:
        directory=family/name/('radio' if name=='public_history_fu_v19' else '')
        assert json.loads((directory/'status.json').read_text())['status'].startswith('completed')
        rows=json.loads((directory/'results.json').read_text());inputs[str((directory/'results.json').relative_to(root))]=sha(directory/'results.json')
        for row in rows:
            if row['status']!='feasible_radio_only' or row['relative_tolerance']!=.2:continue
            rank=row['rank'];bits=row.get('quantization_bits',8);mode=row['mode'];fraction=row['analog_fraction']
            filename=f'r0.2_rank{rank}_{mode}_f{fraction}_design.npz' if name.startswith('public_history') else f'q{bits}_{mode}_f{fraction}_design.npz'
            path=directory/filename
            cases.append(dict(family=name,row=row,path=str(path.relative_to(root)),sha256=sha(path),rank=rank,bits=bits,mode=mode,reports=name=='certified_error_v27'))
    out=family/'inverse_budget_v29';out.mkdir(exist_ok=True)
    def write(name,value):
        target=out/name;tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(target)
    plan=dict(version='inverse_budget_v29',epsilon=epsilon,cases=len(cases),case_inputs=cases,input_result_sha256=inputs,
        input_vector_sha256=sha(source/'vectors.npz'),core_sha256=parent['core_sha256'],
        code_sha256={n:sha(Path(__file__).with_name(n)) for n in ['solver.py','certificate.py','test_solver.py','run.py']},
        limits=['All saved feasible original0.2 incumbents from the complete58history,24low-bit and27paid-report grids; no favorable method-only selection. This is NOT all unsaved heads or an outer-budget/global optimum.',
            'Fixed head, codec, packet/analog/digital durations and original ideal update. PerblockRMS budgets equal each seed existing bound, so a certified new design preserves global original epsilon.',
            'ZeroSICresidual only. Eliminate minimum digitalpowers and use inverse receive magnitude. Convex fixed-block-budget inner problem; numerical results still independently verified in physicalW/J units.',
            'All coefficient/error-reportRF,time,power,battery costs stay paid. New solverhead-computation energy unmeasured. No fullUAVenergy/NOMA/deletion guarantee.',
            'Analytic single-link optimum test and explicit zero-receiver/nonzero-SIC handling precede real fixtures. Every seed and candidate gets an independent quantizer/physical certificate; retain failures and unchanged cases.',
            'No test/task-output-based selection, new model inference, GPU/queue/source edits or automatic full-training promotion. No new solve starts after noon; partial sweep not architecture ranking.'])
    if (out/'plan.json').exists():assert json.loads((out/'plan.json').read_text())==plan
    else:write('plan.json',plan)
    if args.plan_only:print(json.dumps(dict(cases=len(cases),epsilon=epsilon)));return
    assert not (out/'results.json').exists(),'Do not duplicate this sweep.'
    note='''Inverse-receiver fixed-block-budget formulation (isolated v29, zero SIC residual)
Use normalized channel H, analog RMS amplitude a, receive magnitude s, quantization norm bounds delta.
Let r_m be the seed design block RMS upper bound, held fixed. n_m=sqrt(d_m/(2L_A,m)).
For s>0 set t=1/s. The error inequality is exactly
  sum_u |H_um a_um-rho_um t_m| + n_m <= (r_m-sum_u delta_um) t_m.
The left side is convex and the right side affine. For Hybrid, the componentwise-minimum digital amplitude is
  kappa_um * norm([1,sum_i H_im a_im]),
where kappa is obtained by backward perfect-SIC power recursion for the fixed payload/order/symbol counts.
For phase separation it is constant kappa because analog interference is absent; for Mixed it is zero.
The energy bound, power caps and per-user batteries are convex nondecreasing functions of these amplitudes.
An epigraph for that two-element norm gives an SOCP optimizing TX and RX together for fixed r/head/codec/time.
Zero-receiver seed blocks are handled explicitly by fixing a=0,s=0 and minimum digital power; inverse infinity is not approximated.
This is a global convex INNER formulation only. The original global RMS budget allocation and head search remain outside it.
Each seed is a feasible reference, apart from numerical tolerance. Report actual nonincreasing, independently certified incumbents only.
Actual head solver computation/circuit cost is unmeasured; RF upper-bound savings alone do not establish total-energy benefit.
'''
    (out/'method_note.txt').write_text(note)
    rows=[];deadline=datetime(2026,10,21,12,tzinfo=timezone(timedelta(hours=9)))
    for i,case in enumerate(cases):
        if datetime.now(timezone(timedelta(hours=9)))>=deadline:
            write('status.json',dict(status='partial_noon_cutoff_no_ranking',completed=len(rows),total=len(cases)));return
        write('status.json',dict(status='refining_saved_incumbent',case=i,total=len(cases),family=case['family'],mode=case['mode'],at_kst=datetime.now().isoformat(timespec='seconds')))
        path=root/case['path'];assert sha(path)==case['sha256']
        with np.load(path) as data:seed={k:data[k].copy() for k in data.files}
        oldrow=dict(case['row']);oldrow['expected_energy_random_codewords_j']=oldrow.get('expected_energy_including_coefficients_j',oldrow['expected_energy_random_codewords_j'])
        opts=dict(reports=case['reports'],rank=case['rank'],basis=bases.get(case['rank']),public_scale=public_scale)
        oldcert=check(learning,unlearning,channels,cfg,case['bits'],case['mode'],epsilon,seed,oldrow,**opts)
        raw=np.zeros_like(unlearning) if case['mode']=='mixed_aircomp' else unlearning
        payload=np.where(np.max(np.abs(block(raw)),axis=-1)>0,width*case['bits']+cfg['scale_metadata_bits'],0)[seed['users']]
        started=time.monotonic();row=dict(case=i,family=case['family'],mode=case['mode'],rank=case['rank'],bits=case['bits'],reports=case['reports'],
            head=int(seed['head']),analog_fraction=case['row']['analog_fraction'],seed_design_path=case['path'],seed_design_sha256=case['sha256'],
            baseline_energy_j=oldcert['energy_upper_j'],seed_certificate=oldcert)
        try:
            new,stats=refine(seed,payload,cfg)
            cert=check(learning,unlearning,channels,cfg,case['bits'],case['mode'],epsilon,new,**opts)
            assert np.all(np.asarray(cert['block_error_bound'])<=np.asarray(oldcert['block_error_bound'])+np.maximum(1e-12,np.asarray(oldcert['block_error_bound'])*1e-5))
            saving=oldcert['energy_upper_j']-cert['energy_upper_j']
            row.update(solver=stats,candidate_certificate=cert,energy_saving_j=saving,energy_saving_fraction=saving/oldcert['energy_upper_j'])
            if saving>=0:
                new['energy_j']=np.asarray(cert['energy_upper_j']-cert['extra_control_energy_j']);new['error_bound']=np.asarray(cert['root_error_bound'])
                dest=out/f'case_{i:03d}_design.npz';np.savez_compressed(dest,**new)
                row.update(status='certified_nonincreasing_design',design_path=str(dest.relative_to(root)),design_sha256=sha(dest),refined_energy_j=cert['energy_upper_j'])
            else:row.update(status='retain_seed_numerical_energy_increase',refined_energy_j=oldcert['energy_upper_j'])
        except Exception as exc:
            row.update(status='retain_seed_refinement_failed',error_type=type(exc).__name__,error=str(exc),refined_energy_j=oldcert['energy_upper_j'])
        row['seconds']=time.monotonic()-started;rows.append(row);write('results.json',rows)
        print('CASE',json.dumps({k:row[k] for k in ['case','family','mode','rank','bits','status','baseline_energy_j','refined_energy_j']}),flush=True)
    summary=[]
    for mode in ['hybrid','mixed_aircomp','phase_grid']:
        eligible=[r for r in rows if r['mode']==mode];best=min(eligible,key=lambda x:x['refined_energy_j'])
        summary.append({k:v for k,v in best.items() if k not in ['solver','seed_certificate','candidate_certificate']})
    write('summary.json',dict(selected=summary,all_cases=len(rows),certified_nonincreasing=sum(r['status']=='certified_nonincreasing_design' for r in rows),
        failed_or_unchanged=[r['case'] for r in rows if r['status']!='certified_nonincreasing_design'],limits=plan['limits']))
    write('status.json',dict(status='completed_fixed_block_budget_convex_refinement_no_gpu_promotion',cases=len(rows),at_kst=datetime.now().isoformat(timespec='seconds')))


if __name__=='__main__':main()
