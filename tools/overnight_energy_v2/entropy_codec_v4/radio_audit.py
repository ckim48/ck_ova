"""CPU radio screen for lossless coding, with explicit unmeasured overhead."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

import codec


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    args=parser.parse_args();root=args.project.resolve()
    sys.path[:0]=[str(root),str(root/'tools/overnight_energy_v2'),str(root/'tools/overnight_energy_v2/joint_power_v3')]
    from phase_split_grid import make_inputs,necessary_power,split_symbols
    from adaptive_precision import exact_statistics
    from transport import energy_accounting
    from wireless_joint_v3 import optimize_head
    source=root/'results/failed_round_diagnostic';output=root/'results/overnight_energy_v2/entropy_codec_v4'
    c=json.loads((source/'metadata.json').read_text())['config']
    with np.load(source/'updates.npz') as data:
        learning,unlearning,channels=[data[k].copy() for k in ['learning','unlearning','channels']]
    inputs=make_inputs(learning,unlearning,c)
    l,u,decoded,delta,original_payload,rho,dims,symbols=inputs
    with np.load(output/'bits8_payloads.npz') as data:
        payload=data['payload'].copy()
        np.testing.assert_array_equal(delta,data['delta'])
        np.testing.assert_array_equal(original_payload,data['original_payload'])
    screen=json.loads((output/'payload_screen.json').read_text())
    assert screen['updates_sha256']==sha(source/'updates.npz') and screen['codec_sha256']==sha(codec.__file__)
    for client in range(c['users']):
        if not np.any(u[client]):continue
        packet=(output/f'bits8_client{client}.bin').read_bytes()
        np.testing.assert_array_equal(codec.decode(packet,len(dims),l.shape[-1]),decoded[client])
        np.testing.assert_array_equal(codec.fragment_bits(packet,len(dims)),payload[client])
    # Before head/power selection, each active client broadcasts its 32-bit length.
    # Ideal coding, maximum permitted RF power, and the weakest peer link give
    # a feasible conservative signaling airtime/TX energy in this channel model.
    control_symbols=np.zeros(c['users'],dtype=int)
    for client in np.flatnonzero(np.any(payload>0,axis=1)):
        weakest=np.min(np.delete(channels[client],client))
        capacity=np.log2(1+weakest**2*c['power_limit_w'])
        control_symbols[client]=int(np.ceil(32/capacity))
    control_time=control_symbols.sum()/c['symbol_rate_hz']
    control_tx=c['power_limit_w']*control_symbols/c['symbol_rate_hz']
    radio_config=dict(c,available_energy_j=c['available_energy_j']-float(control_tx.max()))
    # This reserve covers the actual length-control TX energy of every UAV.
    # Extra RX circuitry and codec compute remain explicitly unmeasured.
    manifest={'bits':8,'relative_tolerances':[.2,.1,.6],
        'analog_fractions':[.125,.25,.375,.5,.625,.75,.875],
        'updates_sha256':sha(source/'updates.npz'),'payload_screen_sha256':sha(output/'payload_screen.json'),
        'source_sha256':{str(Path(n)):sha(root/n) for n in ['wireless.py','transport.py',
            'tools/overnight_energy_v2/phase_split_grid.py','tools/overnight_energy_v2/adaptive_precision.py',
            'tools/overnight_energy_v2/joint_power_v3/wireless_joint_v3.py',
            'tools/overnight_energy_v2/entropy_codec_v4/codec.py',
            'tools/overnight_energy_v2/entropy_codec_v4/radio_audit.py']},
        'length_control_symbols_per_client':control_symbols.tolist(),
        'length_control_tx_j_per_client':control_tx.tolist(),
        'length_control_s':float(control_time),
        'unmeasured_costs':'Codec computation and extra control-RX circuitry. Do not rank as an end-to-end energy winner. Report a break-even allowance for those costs instead.',
        'protocol_difference':'Lossless client-wide quantized U stream, balanced byte fragments across blocks, reconstructed after all blocks. Original quantization and RMS envelope retained. Phase and Hybrid use identical codec. Mixed has no digital stream and pays no unnecessary length signaling.'}
    p=output/'radio_manifest.json'
    if p.exists() and json.loads(p.read_text())!=manifest:raise ValueError('Different entropy radio experiment exists.')
    p.write_text(json.dumps(manifest,indent=2)+'\n')
    uncoded=json.loads((root/'results/overnight_energy_v2/joint_power_v3/results.json').read_text())
    map_rows=json.loads((root/'results/overnight_energy_v2/error_budget_regimes/results.json').read_text())
    p=output/'radio_results.json';rows=json.loads(p.read_text()) if p.exists() else []
    done={(r['relative_tolerance'],r['mode'],r['analog_fraction']) for r in rows}
    for relative in manifest['relative_tolerances']:
        epsilon=max(c['absolute_tolerance_floor'],relative*np.linalg.norm(rho.sum(axis=0)))
        for mode,fraction in [('hybrid',None)]+[('orthogonal',f) for f in manifest['analog_fractions']]:
            if (relative,mode,fraction) in done:continue
            started=time.monotonic();best=None;heads=[]
            for head in range(c['users']):
                analog_symbols,digital_symbols=symbols.copy(),symbols.copy()
                if mode=='orthogonal':analog_symbols,digital_symbols,_=split_symbols(symbols,payload,head,fraction)
                item={'head':head,'status':'no_feasible_design_found'}
                if np.any(analog_symbols<dims):
                    item['status']='symbol_dimension_condition_failed';heads.append(item);continue
                floor=necessary_power(channels,payload,head,digital_symbols,radio_config)
                if floor is None or np.max(floor,initial=0)>c['power_limit_w']*(1+1e-6):
                    item['status']='necessary_digital_power_condition_failed';heads.append(item);continue
                design=optimize_head(head,channels,rho,delta,payload,dims,analog_symbols,
                    epsilon,radio_config,c['sic_residual_fraction'],mode,digital_symbols=digital_symbols)
                if design is not None:
                    item.update(status='feasible',data_energy_bound_j=design.energy_j)
                    if best is None or design.energy_j<best.energy_j:best=design
                heads.append(item)
            row={'relative_tolerance':relative,'mode':mode,'analog_fraction':fraction,
                'epsilon':float(epsilon),'status':'no_feasible_design_found','heads':heads}
            if best is not None:
                stats=energy_accounting(best,c)
                assert stats['error_bound']<=epsilon*(1+1e-5)
                total_time=stats['uplink_s']+control_time
                assert total_time<=c['round_uplink_deadline_s']
                row.update(status='radio_feasible_codec_rx_overhead_unmeasured',
                    data_energy_bound_j=stats['energy_bound_j'],data_expected_energy_j=stats['expected_energy_random_codewords_j'],
                    radio_plus_control_tx_j=stats['energy_bound_j']+float(control_tx.sum()),
                    head=best.head,error_bound=stats['error_bound'],
                    communication_s_before_codec_compute=float(total_time),
                    remaining_deadline_for_codec_s=float(c['round_uplink_deadline_s']-total_time))
                row.update(exact_statistics(best,l,u,decoded,learning.shape[1]))
                original=[r for r in uncoded if r['bits']==8 and r['relative_tolerance']==relative
                          and r['mode']==mode and r['analog_fraction']==fraction and 'selected_energy_bound_j' in r]
                if original:
                    row['same_setting_uncoded_energy_j']=original[0]['selected_energy_bound_j']
                    row['extra_codec_and_rx_energy_break_even_j']=original[0]['selected_energy_bound_j']-row['radio_plus_control_tx_j']
                mixed=[r['energy_bound_j'] for r in map_rows if r['bits']==8 and r['relative_tolerance']==relative and r['mode']=='mixed_aircomp' and r['status']=='feasible']
                mixed += [r['selected_energy_bound_j'] for r in uncoded if r['bits']==8 and r['relative_tolerance']==relative and r['mode']=='mixed_aircomp' and 'selected_energy_bound_j' in r]
                row['strongest_saved_mixed_energy_j']=min(mixed) if mixed else None
                name=f'r{relative:g}_{mode}_f{fraction}'
                np.savez_compressed(output/(name+'_design.npz'),**{k:getattr(best,k) for k in best.__dataclass_fields__ if k!='history'})
            row['seconds']=time.monotonic()-started;rows.append(row)
            temp=p.with_suffix('.partial.json');temp.write_text(json.dumps(rows,indent=2)+'\n');temp.replace(p)
            print(json.dumps({k:v for k,v in row.items() if k!='heads'}),flush=True)
    best_rows=[]
    for relative in manifest['relative_tolerances']:
        for mode in ['hybrid','orthogonal']:
            feasible=[r for r in rows if r['relative_tolerance']==relative and r['mode']==mode and 'radio_plus_control_tx_j' in r]
            if feasible:best_rows.append({k:v for k,v in min(feasible,key=lambda r:r['radio_plus_control_tx_j']).items() if k!='heads'})
    (output/'radio_best_observed.json').write_text(json.dumps(best_rows,indent=2)+'\n')


if __name__=='__main__':
    main()
