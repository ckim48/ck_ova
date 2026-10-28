"""Recover exact completed models at 200/300/400 kJ without changing training."""
import argparse
import inspect
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import proposed_pks_report as current
import learning_time_budget as continuation
import learning_time_report as timeline

ROOT, read, write, sha = current.ROOT, current.read, current.write, current.sha
OUT = ROOT/'results/limited_energy_checkpoints_v1_20261028'
BUDGETS = [200, 300, 400]


def original_sources():
    return {(ds, m): current.source('baseline', ds, m)
            for ds in current.DATASETS for m in current.METHODS}


def snapshot(ds, method, budget):
    return OUT/'checkpoints'/f'{budget}kJ'/ds.lower()/method


def register():
    path = OUT/'protocol.json'
    if path.exists():
        result = read(path)
        assert result['source_sha256'][str(Path(__file__).relative_to(ROOT))] == sha(Path(__file__))
        return result
    sources = original_sources()
    selected, references, hashes = [], [], {}
    for (ds, method), arm in sources.items():
        settings = read(arm.parents[1]/'settings.json')
        physical = timeline.prior.physical
        physical.PARAMETERS = settings['energy_model']
        params = physical.PARAMETERS
        tl = read(arm/'mission_timeline.json')
        works = {r['transaction']: r for r in
                 [read(p) for p in (arm/'work_attempts').glob('*.json')]}
        points = []
        energy = last_active = 0.
        cycles = 0
        for p in sorted((arm/'transactions').glob('*.json')):
            row = read(p); work = works[row['sequence']]
            active = timeline.prior.chart.active_time(tl, row['end_s'])
            energy += float(physical.compute_cost(work['compute']['device_macs'])[0].sum())
            energy += timeline.export.plots.comm(row['radio'], params['communication'])[0]
            energy += params['communication']['head_processing_w']*row['radio']['uplink_s']
            energy += 10*float(physical.propulsion_power(0))*(active-last_active)
            last_active = active; cycles += len(row['FU_request_ids'])
            points.append(dict(stage='fixed_workload', sequence=row['sequence'],
                FL_round=row['completed_FL_rounds'], FU_cycles=cycles,
                FU_arrived=row['requests']['arrived'], FU_completed=row['requests']['procedure_completed'],
                energy_kJ=energy/1000, active_s=active,
                expected_evaluation=row['evaluation']))
        curve_path = continuation.OUT/(ds.lower()+'_seed0')/method/'curve.json'
        curve = read(curve_path)
        assert math.isclose(energy, curve[0]['energy_j'], rel_tol=1e-10)
        for point in curve[1:]:
            points.append(dict(stage='retained_continuation', sequence=point['extra_FL'],
                FL_round=point['FL_round'], FU_cycles=225, FU_arrived=45, FU_completed=45,
                energy_kJ=point['energy_j']/1000, active_s=point['active_s'],
                expected_evaluation=point['evaluation']))
        assert all(a['energy_kJ'] < b['energy_kJ'] for a,b in zip(points,points[1:]))
        for budget in BUDGETS:
            candidates=[p for p in points if p['energy_kJ'] <= budget]
            chosen=candidates[-1]
            following=next(p for p in points if p['energy_kJ'] > budget)
            selected.append(dict(dataset=ds, method=method, budget_kJ=budget,
                next_completion_energy_kJ=following['energy_kJ'], **chosen))
        references.append(dict(dataset=ds, method=method, arm=str(arm.relative_to(ROOT))))
        for p in [arm/'final.pt', arm/'service_completion.json', arm.parent/'plan.json',
                  arm.parent/'feature_cache/features.pt', curve_path]:
            hashes[str(p.relative_to(ROOT))] = sha(p)
    protocol = dict(registered_unix=time.time(), budgets_kJ=BUDGETS, main_budget_kJ=300,
        methods=current.METHODS, datasets=current.DATASETS, selected_models=selected, original_sources=references,
        rule='Last completed transaction within each shared active-energy budget, including transactions without an original evaluation. No partial update, interpolation, changed learning schedule, or outcome-based stopping.',
        choice='The existing aggregate mR curves were inspected before this analysis. Report the entire fixed 200/300/400 kJ grid; 300 kJ is the middle illustrative budget. This is exploratory, not independent confirmation.',
        requests='Same prescribed 45 requests with original round-triggered releases. Exposure and completed FU counts can differ at a common budget. Not an exogenous-time arrival benchmark, and not a comparison at equal forgetting quality.',
        evaluation='Use the identical retained validation queries and complete native gallery for every method and budget. Report FL rounds, FU cycles, arrived/completed requests and actual energy. Retained-rank results do not certify unlearning.',
        accounting='Executed communication, adapter computation and active hovering; common warm20, feature extraction, travel and collection excluded as in the existing figures. No partial transaction is counted.',
        integrity='Replay full original fixed-workload arms and required continuations; require exact final parameter equality and all recorded evaluations/timing before using intermediate models. Raw checkpoints/features/logs remain local.',
        input_sha256=hashes, source_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))})
    write(path,protocol)
    return protocol


def save_selected(theta, ds, method, stage, sequence):
    import torch
    for row in ACTIVE_PROTOCOL['selected_models']:
        if (row['dataset'],row['method'],row['stage'],row['sequence']) != (ds,method,stage,sequence):
            continue
        dest=snapshot(ds,method,row['budget_kJ']);dest.mkdir(parents=True,exist_ok=True)
        torch.save(dict(theta=theta.detach().cpu(),round=row['FL_round']),dest/'model.pt')
        write(dest/'metadata.json',dict(row,checkpoint_sha256=sha(dest/'model.pt')))


def captured_write(path, value):
    write(path,value)
    path=Path(path)
    if path.parent.name != 'transactions' or 'completed_FL_rounds' not in value:
        return
    ds=next(ds for ds in current.DATASETS if ds.lower()+'_seed0' in path.parts)
    frame=inspect.currentframe().f_back
    try:
        save_selected(frame.f_locals['theta'],ds,ACTIVE_METHOD,'fixed_workload',value['sequence'])
    finally:
        del frame


def clean_runtime(value):
    if isinstance(value,dict):
        return {k:clean_runtime(v) for k,v in value.items() if k not in ['optimizer_wall_s']}
    if isinstance(value,list):
        return [clean_runtime(v) for v in value]
    return value


def verify_fixed(old,new,ds,method):
    import torch
    assert read(new/'outcome.json')['status']=='completed'
    assert torch.equal(torch.load(old/'final.pt',map_location='cpu',weights_only=True)['theta'],
                       torch.load(new/'final.pt',map_location='cpu',weights_only=True)['theta'])
    previous=sorted((old/'transactions').glob('*.json'))
    assert len(previous)==len(list((new/'transactions').glob('*.json')))
    for p in previous:
        assert clean_runtime(read(p))==clean_runtime(read(new/'transactions'/p.name)),(ds,method,p.name)
    assert read(old/'service_completion.json')['evaluation']==read(new/'service_completion.json')['evaluation']
    result=dict(passed=True,dataset=ds,method=method,transactions=len(previous),
                final_parameters_exact=True,all_transactions_exact_except_solver_wall_time=True)
    write(OUT/'checks'/f'{ds.lower()}_{method}_fixed.json',result)
    print(result,flush=True)


def replay_fixed(method):
    global ACTIVE_PROTOCOL,ACTIVE_METHOD
    ACTIVE_PROTOCOL=register();ACTIVE_METHOD=method
    source_map={(r['dataset'],r['method']):ROOT/r['arm'] for r in ACTIVE_PROTOCOL['original_sources']}
    import workload_sweeps_v3 as infra
    destination=OUT/'fixed'/method
    if method in ['gate_select16','salun']:
        import learning_quality_v3 as study
        if method=='gate_select16':
            import learning_selection_fix_v2 as selected
            selected.install()
        # Validate the original registered sources before changing output locations.
        study.register()
        original_make=study.make_arm
        original_paper=infra.paper_arm
        def make(*args):
            return infra.cached_clone(original_make(*args),[],dict(write=captured_write))
        def paper():
            return infra.cached_clone(original_paper(),[],dict(write=captured_write))
        study.OUT=destination;study.register=lambda:ACTIVE_PROTOCOL
        study.make_arm=make;infra.paper_arm=paper
        study.run(method,'baseline')
        raw_method=study.DIGITAL if method=='salun' else study.PROPOSED
        for ds in current.DATASETS:
            verify_fixed(source_map[(ds,method)],destination/method/'cases/baseline'/(ds.lower()+'_seed0')/raw_method,ds,method)
    else:
        import paper_uav_benchmarks as study
        study.register()
        study.OUT=destination;study.register=lambda:ACTIVE_PROTOCOL
        study.train_arm=infra.cached_clone(study.train_arm,[],dict(write=captured_write),live_globals=study.__dict__)
        raw_method=current.TRANSFER_IDS[method]
        study.run_case('baseline',raw_method)
        for ds in current.DATASETS:
            verify_fixed(source_map[(ds,method)],destination/'cases/baseline'/(ds.lower()+'_seed0')/raw_method,ds,method)


def replay_continuation(ds):
    global ACTIVE_PROTOCOL
    ACTIVE_PROTOCOL=register()
    import torch
    methods={m:current.METHODS[m] for m in ['gate_select16','afl']}
    fn=continuation.quality.infra.cached_clone(continuation.run,[
        ('   extra=0;cached={};current_cost=np.zeros(10);job_no=0;all_jobs=[]',
         '   extra=0;cached={};current_cost=np.zeros(10);job_no=0;all_jobs=[]\n   save_selected(theta,ds,m,"retained_continuation",extra)',1),
        ("    evaluations.append(dict(**point,evaluation=ev));write(ad/'evaluations'/f'{extra:04d}.json',evaluations[-1])",
         "    evaluations.append(dict(**point,evaluation=ev));write(ad/'evaluations'/f'{extra:04d}.json',evaluations[-1])\n    save_selected(theta,ds,m,\"retained_continuation\",extra)",1)],
        dict(OUT=OUT/'continuation',METHODS=methods,register=lambda:ACTIVE_PROTOCOL,save_selected=save_selected))
    fn(ds)
    for method in methods:
        old=continuation.OUT/(ds.lower()+'_seed0')/method
        new=OUT/'continuation'/(ds.lower()+'_seed0')/method
        assert read(old/'curve.json')==read(new/'curve.json')
        assert torch.equal(torch.load(old/'final.pt',map_location='cpu',weights_only=True)['theta'],
                           torch.load(new/'final.pt',map_location='cpu',weights_only=True)['theta'])
        write(OUT/'checks'/f'{ds.lower()}_{method}_continuation.json',dict(passed=True,
            dataset=ds,method=method,original_curve_and_final_model_exact=True))


def run():
    protocol=register()
    for method in current.METHODS:
        subprocess.run([sys.executable,__file__,'fixed','--method',method],cwd=ROOT,check=True)
    for ds in current.DATASETS:
        subprocess.run([sys.executable,__file__,'continuation','--dataset',ds],cwd=ROOT,check=True)
    for row in protocol['selected_models']:
        dest=snapshot(row['dataset'],row['method'],row['budget_kJ'])
        assert sha(dest/'model.pt')==read(dest/'metadata.json')['checkpoint_sha256']
    write(OUT/'finished.json',dict(passed=True,checkpoints=len(protocol['selected_models']),
        budgets_kJ=BUDGETS,all_original_trajectories_reproduced=True))
    print(dict(passed=True,checkpoints=len(protocol['selected_models'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['register','fixed','continuation','run'])
    p.add_argument('--method',choices=list(current.METHODS))
    p.add_argument('--dataset',choices=current.DATASETS)
    a=p.parse_args()
    if a.action=='register':
        result=register()
        print([{k:v for k,v in r.items() if k not in ['expected_evaluation']} for r in result['selected_models']])
    elif a.action=='fixed':replay_fixed(a.method)
    elif a.action=='continuation':replay_continuation(a.dataset)
    else:run()
