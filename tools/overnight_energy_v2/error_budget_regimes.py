"""Map energy/error tradeoffs with an equally optimized phase comparator."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--bits',type=int,nargs='+',default=[8,6])
    parser.add_argument('--relative-tolerances',type=float,nargs='+',default=[0.6,0.8,0.4,0.2,0.1])
    parser.add_argument('--analog-fractions',type=float,nargs='+',default=[0.125,0.25,0.375,0.5,0.625,0.75,0.875])
    args=parser.parse_args();root=args.project.resolve();sys.path.insert(0,str(root))
    from phase_split_grid import make_inputs,optimize_fraction
    from adaptive_precision import exact_statistics
    from transport import design_transport,energy_accounting
    source=root/'results/failed_round_diagnostic'
    config=json.loads((source/'metadata.json').read_text())['config']
    with np.load(source/'updates.npz') as data:
        l,u,channels=[data[k].copy() for k in ('learning','unlearning','channels')]
    source_hash={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ('wireless.py','transport.py')}
    input_hash=hashlib.sha256((source/'updates.npz').read_bytes()).hexdigest()
    # Reuse compatible, already-computed cases; no repeated baseline solves.
    cache={}
    old=root/'results/energy_quality_radio_v1'
    old_manifest=json.loads((old/'manifest.json').read_text())
    if old_manifest['source_sha256']==source_hash and old_manifest['updates_sha256']==input_hash:
        with (old/'radio_audit.csv').open() as f:
            for r in csv.DictReader(f):
                if int(r['receive_initializations'])!=config['receive_initializations']:
                    continue
                method=r['method'];fraction=0.5 if method=='orthogonal_equal_time' else None
                mode='orthogonal' if method=='orthogonal_equal_time' else method
                row={'status':r['status'],'cached_from':str(old/'radio_audit.csv')}
                mapping={'energy_bound_j':'energy_bound_j','expected_energy_j':'expected_energy_random_codewords_j',
                         'uplink_s':'uplink_s','head':'head','error_bound':'error_bound','analytic_total_rmse':'analytic_total_rmse'}
                for key,target in mapping.items():
                    if r.get(key)!='':row[target]=float(r[key])
                cache[(8,float(r['relative_tolerance']),mode,fraction)]=row
    phase=root/'results/overnight_energy_v2/phase_split_grid'
    phase_manifest=json.loads((phase/'manifest.json').read_text())
    if phase_manifest['source_sha256']==source_hash and phase_manifest['updates_sha256']==input_hash:
        for r in json.loads((phase/'results.json').read_text()):
            cache[(r['bits'],phase_manifest['relative_tolerance'],'orthogonal',r['analog_fraction'])]={
                **{k:v for k,v in r.items() if k not in ('heads','bits','analog_fraction','seconds')},'cached_from':str(phase/'results.json')}
    precision=json.loads((old/'precision_design_audit.json').read_text())
    if precision['source_sha256']==source_hash and precision['updates_sha256']==input_hash:
        for r in precision['rows']:
            cache[(r['bits'],config['relative_tolerance'],'hybrid',None)]={
                **{k:v for k,v in r.items() if k not in ('bits','seconds')},'cached_from':str(old/'precision_design_audit.json')}
    output=root/'results/overnight_energy_v2/error_budget_regimes';output.mkdir(parents=True,exist_ok=True)
    manifest={'bits':args.bits,'relative_tolerances':args.relative_tolerances,'analog_fractions':args.analog_fractions,
        'source_sha256':source_hash,'updates_sha256':input_hash,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'interpretation':'Exploratory same-update energy/error regime map. Same channel, time and power constraints across methods at each error budget; all listed budgets are retained. Large error budgets may harm VLM utility/forgetting and are not comparable as equal-task-quality until separately validated. Do not select a favorable loose budget using test data. Phase split is grid-optimized; digital users retain NOMA.'}
    f=output/'manifest.json'
    if f.exists() and json.loads(f.read_text())!=manifest:raise ValueError('Existing regime experiment differs.')
    f.write_text(json.dumps(manifest,indent=2)+'\n')
    f=output/'results.json';rows=json.loads(f.read_text()) if f.exists() else []
    done={(r['bits'],r['relative_tolerance'],r['mode'],r.get('analog_fraction')) for r in rows}
    for bits in args.bits:
        for relative in args.relative_tolerances:
            c=dict(config,quantization_bits=bits,relative_tolerance=relative)
            blocked=make_inputs(l,u,c)
            epsilon=max(c['absolute_tolerance_floor'],relative*np.linalg.norm(blocked[5].sum(axis=0)))
            total_time=blocked[-1].sum()/c['symbol_rate_hz']
            cases=[('hybrid',None),('mixed_aircomp',None)]+[('orthogonal',a) for a in args.analog_fractions]
            for mode,fraction in cases:
                key=(bits,relative,mode,fraction)
                if key in done:continue
                began=time.monotonic()
                row={'bits':bits,'relative_tolerance':relative,'mode':mode,'analog_fraction':fraction,'epsilon':float(epsilon)}
                if key in cache:
                    row.update(cache[key])
                elif mode=='mixed_aircomp' and (8,relative,mode,None) in cache:
                    row.update(cache[(8,relative,mode,None)])
                    row['precision_irrelevant_for_mixed']=True
                else:
                    if mode=='orthogonal':
                        design,inputs,heads=optimize_fraction(l,u,channels,c,epsilon,fraction)
                        bl,bu,decoded=inputs[:3]
                        row['feasible_heads']=sum(x['status']=='feasible' for x in heads)
                    else:
                        design,bl,bu,decoded,count=design_transport(l,u,channels,c,mode,epsilon)
                        row['feasible_heads']=count
                    row['status']='feasible' if design is not None else 'no_feasible_design_found'
                    if design is not None:
                        row.update(energy_accounting(design,c))
                        row.update(exact_statistics(design,bl,bu,decoded,l.shape[1]))
                        if mode=='mixed_aircomp':row['unlearning_component_error']=None
                        cache[key]={k:v for k,v in row.items() if k not in ('bits','relative_tolerance','mode','analog_fraction')}
                if row['status']=='feasible':
                    assert np.isclose(row['uplink_s'],total_time)
                    assert row['error_bound']<=epsilon*(1+1e-5)
                    if row.get('analytic_total_rmse') is not None:
                        assert row['analytic_total_rmse']<=row['error_bound']*(1+1e-4)+1e-12
                row['seconds']=time.monotonic()-began
                rows.append(row)
                tmp=f.with_suffix('.partial.json');tmp.write_text(json.dumps(rows,indent=2)+'\n');tmp.replace(f)
                print(json.dumps(row),flush=True)
    comparisons=[]
    for bits in args.bits:
        for relative in sorted(args.relative_tolerances):
            group=[r for r in rows if r['bits']==bits and r['relative_tolerance']==relative and r['status']=='feasible']
            hybrid=next((r for r in group if r['mode']=='hybrid'),None)
            separated=[r for r in group if r['mode']=='orthogonal']
            phase_best=min(separated,key=lambda r:r['energy_bound_j']) if separated else None
            mixed=next((r for r in group if r['mode']=='mixed_aircomp'),None)
            comparisons.append({'bits':bits,'relative_tolerance':relative,
                'hybrid_energy_j':hybrid['energy_bound_j'] if hybrid else None,
                'phase_grid_best_energy_j':phase_best['energy_bound_j'] if phase_best else None,
                'phase_grid_best_analog_fraction':phase_best['analog_fraction'] if phase_best else None,
                'mixed_energy_j':mixed['energy_bound_j'] if mixed else None,
                'hybrid_lower_than_phase_at_same_update_error_budget':bool(hybrid and phase_best and hybrid['energy_bound_j']<phase_best['energy_bound_j'])})
    (output/'comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    print('COMPARISONS',json.dumps(comparisons),flush=True)


if __name__=='__main__':main()
