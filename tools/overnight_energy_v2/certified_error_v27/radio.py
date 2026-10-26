"""Paid a-posteriori quantization certificates at fixed actualFU61 updates."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from metadata import report_bounds, reserve
from certificate import check


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True); parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args(); root = args.project.resolve()
    sys.path[:0] = [str(root), str(root/'tools/overnight_energy_v2'), str(root/'tools/overnight_energy_v2/initialization_v11')]
    from phase_split_grid import make_inputs, split_symbols, necessary_power
    from wireless_init_v11 import optimize_head
    from transport import energy_accounting
    from adaptive_precision import exact_statistics
    source = root/'results/overnight_energy_v2/public_history_fu_v19/screen'; parent = json.loads((source/'plan.json').read_text())
    cfg = dict(parent['config'], max_ao_iterations=30)
    assert all(sha(root/n) == value for n,value in parent['core_sha256'].items())
    solver = root/'tools/overnight_energy_v2/initialization_v11/wireless_init_v11.py'; assert sha(solver) == parent['parent_solver_sha256']
    with np.load(source/'vectors.npz') as data: learning, unlearning, channels = [data[k].copy() for k in ['learning','unlearning','channels']]
    base = make_inputs(learning, unlearning, cfg); epsilon = float(.2*np.linalg.norm(base[5].sum(axis=0)))
    out = root/'results/overnight_energy_v2/certified_error_v27'; out.mkdir(exist_ok=True)
    def write(name,value):
        target=out/name;tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(target)
    plan = dict(version='certified_error_v27', actual_round=61, bits=[4,6,8], relative_tolerance=.2, epsilon=epsilon,
        modes=['hybrid','phase_grid'], phase_fractions=parent['phase_fractions']+['minimum_dimension_boundary'], cases=27,
        config=cfg, core_sha256=parent['core_sha256'], input_vector_sha256=sha(source/'vectors.npz'), solver_sha256=sha(solver),
        code_sha256={n:sha(Path(__file__).with_name(n)) for n in ['radio.py','metadata.py','certificate.py']},
        extra_stats_bits_per_remote_requester=32*len(base[-1]), extra_stats_rate_bits_per_complex_symbol=.5,
        reason='In actualFU61,4bit worst-case quantizer bound permits only some heads, while computed decoded-minus-input norms permit all10 before RF noise. Tighter valid local certificates may reduce analog power. Preview uses fixed private updates/codec arithmetic, no model outputs/tests.',
        limits=['New information/control protocol: UAV locally computes perblock norm of quantized-minus-originalU and reports32bit floats rounded UP. Valid conditional upper bound, no changed ideal update or epsilon.',
            'Statistics are additional ideal-coded orthogonal uplink packets:0.5bit/symbol,one2624-symbol packet per remote requestingUAV. Ownhead statistics require noRF. Reserve these exact symbols from original167936symbol frame, pay packetRF/power/battery/coherence costs. No free new channel/private gradient/history.',
            'AllHybrid andphase codecs/head/fraction cases get the same reporting option. Both original worstbound protocols remain eligible; Mixed needs no quantization-report packet and retains its full-frame comparator. No weakened baseline.',
            'Strong7fractionphase plus actual per-head minimum analogL_A=d boundary. All10heads,convex+4starts/AO30. Availablebattery conservatively reserves max remote packet energy before solving and exact user batteries are checked afterwards.',
            'Original scalar metadata already carries quantizer scale; these are ADDITIONAL32bit error reports. Fixed per-case bitdepth requires no uncounted adaptive depth header. Public choice of protocol assumed in existing fixed control configuration.',
            'Real report measurement/quantizer computation/coding/circuit/header energy unmeasured. Report conditional communication RF/upper-bound gain only, with remaining extra-cost allowance. No total UAV energy/NOMA/deletion/global-optimality claim.',
            '2bit actual norm floor still exceeds original bound for everyhead, so4/6/8bit targeted. No task/test-output selection,newGPU orautomaticlongtraining. Stop starting newhead solves at noon; partialgrid excluded from ranking.'])
    if (out/'plan.json').exists():assert json.loads((out/'plan.json').read_text())==plan
    else:write('plan.json',plan)
    if args.plan_only:print(json.dumps(dict(cases=27,epsilon=epsilon,extra_bits_per_remote_requester=plan['extra_stats_bits_per_remote_requester'])));return
    assert not (out/'results.json').exists(),'Do not duplicate this diagnostic.'
    # Check both unmodified known references, power rejection, upward reporting,
    # and the physically different head-owned/remote request slot counts.
    old=root/'results/overnight_energy_v2/public_history_fu_v19/radio';old_rows=json.loads((old/'results.json').read_text())
    tests=[]
    for mode,fraction in [('hybrid',None),('phase_grid',.125)]:
        row=next(x for x in old_rows if x['relative_tolerance']==.2 and x['rank']==0 and x['mode']==mode and x['analog_fraction']==fraction)
        with np.load(old/f'r0.2_rank0_{mode}_f{fraction}_design.npz') as data:d={k:data[k].copy() for k in data.files}
        tests.append(check(learning,unlearning,channels,cfg,8,mode,epsilon,d,row,reports=False))
        bad={k:v.copy() for k,v in d.items()};bad['analog_rms'][0,0]+=1000.
        try:check(learning,unlearning,channels,cfg,8,mode,epsilon,bad,row,reports=False)
        except AssertionError:pass
        else:raise AssertionError('Invalid power accepted.')
    bounds=report_bounds(base[1],base[2]);assert np.all(bounds>=np.linalg.norm(base[2]-base[1],axis=-1))
    for head,count in [(5,1),(0,2)]:
        main,pc,ec,reporters,n=reserve(channels,base[1],base[-1],head,cfg['symbol_rate_hz'])
        assert reporters.sum()==count and int(main.sum())+count*n==int(base[-1].sum())
        assert n==2624 and np.all(main==(4096-64*count))
    write('startup_checks.json',dict(status='known8bitH_phase_power_rejections_upward_rounding_head_owned_packet_counts_passed',references=tests))
    deadline=datetime(2026,10,21,12,tzinfo=timezone(timedelta(hours=9)));rows=[];certificates=[]
    for bits in plan['bits']:
        c=dict(cfg,quantization_bits=bits);l,u,decoded,old_delta,payload,rho,dims,total=make_inputs(learning,unlearning,c)
        delta=report_bounds(u,decoded);assert np.all(delta<=old_delta*(1+1e-5))
        for mode in plan['modes']:
            for fraction in plan['phase_fractions'] if mode=='phase_grid' else [None]:
                started=time.monotonic();heads=[];best=None;selected=None;best_arrays=None
                for head in range(c['users']):
                    if datetime.now(timezone(timedelta(hours=9)))>=deadline:
                        write('status.json',dict(status='partial_noon_cutoff_no_ranking',completed_cases=len(rows)));return
                    write('status.json',dict(status='solving_cpu_head',quantization_bits=bits,mode=mode,analog_fraction=fraction,head=head,completed_cases=len(rows),at_kst=datetime.now().isoformat(timespec='seconds')))
                    users=np.delete(np.arange(c['users']),head);main,pc,ec,reporters,ncontrol=reserve(channels,u,total,head,c['symbol_rate_hz'])
                    item=dict(head=head,remote_reporters=int(reporters.sum()),report_rf_j=float(ec.sum()))
                    if np.max(pc,initial=0)>c['power_limit_w']*(1+1e-6):item['status']='report_power_failed';heads.append(item);continue
                    if np.linalg.norm(delta[users].sum(axis=0))>=epsilon:item['status']='quantization_bound_floor_exceeds_budget';heads.append(item);continue
                    active=np.any(payload[users]>0,axis=0);na=main.copy();nd=main.copy()
                    if mode=='phase_grid':
                        if fraction=='minimum_dimension_boundary':na=np.where(active,dims,main);nd=np.where(active,main-dims,main)
                        else:na,nd,active=split_symbols(main,payload,head,fraction)
                    if np.any(na<dims):item['status']='symbol_dimension_condition_failed';heads.append(item);continue
                    floor=necessary_power(channels,payload,head,nd,c)
                    if floor is None or np.max(floor,initial=0)>c['power_limit_w']*(1+1e-6):item['status']='necessary_digital_power_failed';heads.append(item);continue
                    reserve_cfg=dict(c,available_energy_j=c['available_energy_j']-float(ec.max(initial=0)))
                    d=optimize_head(head,channels,rho,delta,payload,dims,na,epsilon,reserve_cfg,c['sic_residual_fraction'],'orthogonal' if mode=='phase_grid' else 'hybrid',digital_symbols=nd)
                    item['status']='no_feasible_design_found'
                    if d is not None:
                        stats=energy_accounting(d,c);stats.update(exact_statistics(d,l,u,decoded,learning.shape[1]));data_upper=stats['energy_bound_j'];data_expected=stats['expected_energy_random_codewords_j']
                        stats.update(data_energy_upper_j=data_upper,data_expected_energy_j=data_expected,report_rf_energy_j=float(ec.sum()),report_payload_bits=int(reporters.sum()*32*len(total)),
                            report_control_symbols=int(reporters.sum()*ncontrol),energy_bound_j=data_upper+float(ec.sum()),expected_energy_random_codewords_j=data_expected+float(ec.sum()),total_uplink_s=float(total.sum()/c['symbol_rate_hz']))
                        item.update(status='feasible_radio_only',energy_bound_j=stats['energy_bound_j'],history=d.history)
                        if best is None or stats['energy_bound_j']<selected['energy_bound_j']:
                            best,selected=d,stats;best_arrays={k:getattr(d,k) for k in d.__dataclass_fields__ if k!='history'}
                            best_arrays.update(control_power=pc,control_energy=ec)
                    heads.append(item)
                row=dict(quantization_bits=bits,mode=mode,rank=0,relative_tolerance=.2,analog_fraction=fraction,epsilon=epsilon,
                    status='feasible_radio_only' if best is not None else 'no_feasible_design_found',heads=heads,seconds=time.monotonic()-started)
                if best is not None:
                    row.update(selected);certificate=check(learning,unlearning,channels,c,bits,mode,epsilon,best_arrays,row,reports=True)
                    path=out/f'q{bits}_{mode}_f{fraction}_design.npz';np.savez_compressed(path,**best_arrays)
                    certificate.update(quantization_bits=bits,mode=mode,analog_fraction=fraction,design_sha256=sha(path));certificates.append(certificate);write('certificates.json',certificates)
                rows.append(row);write('results.json',rows);print('CASE',json.dumps({k:v for k,v in row.items() if k!='heads'}),flush=True)
    write('status.json',dict(status='completed27_paid_error_certificate_radio_cases_no_gpu_promotion',cases=len(rows),saved_feasible_candidates=len(certificates),at_kst=datetime.now().isoformat(timespec='seconds')))


if __name__=='__main__':main()
