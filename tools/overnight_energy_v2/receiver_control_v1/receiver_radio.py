"""Fair fixed-frame controller transport diagnostic, including paid sketch feedback."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from receiver_control import control, project_rows
from receiver_pilot import sha, write


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--project',type=Path,required=True)
    a=p.parse_args(); root=a.project.resolve(); scripts=Path(__file__).parent
    sys.path[:0]=[str(root),str(root/'tools/overnight_energy_v2'),str(root/'tools/overnight_energy_v2/initialization_v11')]
    from wireless_init_v11 import optimize_head
    from transport import recover, energy_accounting
    from phase_split_grid import make_inputs, split_symbols, necessary_power
    out=root/'results/overnight_energy_v2/receiver_control_v1'
    plan=json.loads((out/'plan.json').read_text())
    assert all(sha(root/n)==v for n,v in plan['core_sha256'].items())
    assert all(sha(scripts/n)==v for n,v in plan['code_sha256'].items())
    c=plan['config']; c=dict(c,max_ao_iterations=30)
    with np.load(out/'prepared.npz') as z:
        learning,quantized,channels,g,active,ideal=[z[k].copy() for k in ['learning','quantized','channels','gradient','active','control_alpha1']]
    geometry=json.loads((out/'geometry.json').read_text()); eps=geometry['application_rmse_budget']; lip=geometry['controller_lipschitz_bound']
    l,u,decoded,delta,payload,rho,dims,total=make_inputs(learning,quantized,c)
    # Requests are canonical algorithm inputs, including at the head. They are
    # transmitted losslessly as their pre-existing 8-bit integer codes + scale.
    assert np.array_equal(decoded,u), 'Canonical codec must be idempotent.'
    delta=np.zeros_like(delta)
    n,d=learning.shape; blocks=len(dims); W=c['symbol_rate_hz']; full_time=float(total.sum()/W)
    def block(x):
        return np.pad(x,((0,0),(0,(-x.shape[1])%c['block_dimension']))).reshape(n,blocks,-1)
    zero=np.zeros_like(l)
    radio_plan=dict(primary_alpha=1., frame_symbols=int(total.sum()), frame_s=full_time,
        application_rmse_budget=eps, hybrid_oma_analog_budget=eps/lip,
        fractions=[.125,.25,.375,.5,.625,.75,.875], sketch_ranks=[64,128,256], sketch_bits=16,
        sketch_seed=1281061, metadata_scale_bits=32, norm_bits=32, feedback_header_bits=32,
        feedback_weight_bits=32, packet_header_bits=32, statistics_rate_bits_per_symbol=.5,
        feedback_rate_bits_per_symbol=.5, added_circuit_w_per_active_tx_or_rx=.01,
        feedback_budget_allocation=dict(sketch_fraction=.5,analog_fraction=.5),
        source_sha256=plan['code_sha256'],prepared_sha256=sha(out/'prepared.npz'),
        solver_sha256=sha(root/'tools/overnight_energy_v2/initialization_v11/wireless_init_v11.py'),
        limits=['Same total frame, channels, .32W power and .1J per UAV; all10 candidate heads and same strong initialization/AO30.',
            'OMA separates analog learning and digital requests; digital users still NOMA. Seven time splits, not a proven global time optimum or full per-user TDMA.',
            'Original paper common channel/update norm reports treated available for ALL methods; NEW sketch/feedback/optional weighted norm reports explicitly paid.',
            'Added .01W circuitry is a design choice, not measured. Report zero-extra-circuit sensitivity too. Coding/compute/flight energy unmeasured.',
            'Sketch approximation is ex-post checked against half the application budget; not an online hard certificate.',
            'Genie Mixed uses free exact receiver-derived weights as an optimistic reference, not an implementable competitor/global lower bound.',
            'No claim of superiority over untested per-request analog-stream schemes or larger adaptive sketches.'])
    if (out/'radio_plan.json').exists(): assert json.loads((out/'radio_plan.json').read_text())==radio_plan
    else: write(out/'radio_plan.json',radio_plan)
    rowsfile=out/'radio_results.json'; rows=json.loads(rowsfile.read_text()) if rowsfile.exists() else []
    done={x['name'] for x in rows}; physical={}
    physfile=out/'physical_updates.npz'
    if physfile.exists():
        with np.load(physfile) as z: physical={k:z[k].copy() for k in z.files}
    def status(value,**extra):
        write(out/'radio_status.json',dict(status=value,at_kst=datetime.now().isoformat(timespec='seconds'),**extra))
    def solve(name,mode,lb,rb,epsilon,extra=None,fractions=None,bias=0.,correction=None,controller=False,role='physical_model'):
        if name in done:return
        began=time.monotonic(); best=None; diagnostics=[]
        for head in range(n):
            ext=extra(head) if extra else dict(symbols=0,energy=np.zeros(n),rf=np.zeros(n),circuit=np.zeros(n))
            if ext is None:
                diagnostics.append(dict(head=head,status='side_information_power_or_coherence_failed'));continue
            remaining=int(total.sum())-ext['symbols']
            available=np.full(blocks,remaining//blocks,dtype=int); available[:remaining%blocks]+=1
            if np.any(available<dims):
                diagnostics.append(dict(head=head,status='dimension_or_frame_failed'));continue
            for fraction in fractions or [None]:
                analog=available.copy();digital=available.copy()
                pl=payload if controller else np.zeros_like(payload)
                de=delta if controller else np.zeros_like(delta)
                if fraction is not None:analog,digital,_=split_symbols(available,pl,head,fraction)
                if np.any(analog<dims):continue
                if controller:
                    floor=necessary_power(channels,pl,head,digital,c)
                    if floor is None or np.max(floor,initial=0)>c['power_limit_w']*(1+1e-6):
                        diagnostics.append(dict(head=head,fraction=fraction,status='necessary_digital_power_failed'));continue
                status('optimizing',candidate=name,head=head,fraction=fraction)
                design=optimize_head(head,channels,rb,de,pl,dims,analog,epsilon,c,0.,mode,digital_symbols=digital)
                record=dict(head=head,fraction=fraction,status='no_feasible_design_found')
                if design is not None:
                    base=c['base_overhead_j']+np.eye(1,n,head).reshape(-1)*c['head_overhead_j']
                    if mode=='orthogonal':
                        energies=(analog/W*design.analog_rms**2+digital/W*design.digital_sqrt_power**2).sum(axis=1)
                    else:energies=(analog/W*(design.analog_rms+design.digital_sqrt_power)**2).sum(axis=1)
                    base[design.users]+=energies
                    if np.any(base+ext['energy']>c['available_energy_j']*(1+1e-6)):
                        record['status']='additional_control_battery_failed'
                    else:
                        value=design.energy_j+float(ext['energy'].sum())
                        record.update(status='feasible',energy_bound_j=value)
                        if best is None or value<best[0]:best=value,design,ext,fraction
                diagnostics.append(record)
        row=dict(name=name,role=role,status='no_feasible_design_found',heads=diagnostics,
            application_rmse_budget=eps,analog_rmse_budget=epsilon,sketch_bias_norm=float(bias))
        if best is not None:
            value,design,ext,fraction=best; stats=energy_accounting(design,c)
            row.update(stats,energy_bound_j=value,radio_energy_bound_j=stats['energy_bound_j'],
                extra_control_energy_j=float(ext['energy'].sum()),extra_control_rf_j=float(ext['rf'].sum()),
                extra_control_circuit_j=float(ext['circuit'].sum()),
                zero_extra_circuit_energy_j=value-float(ext['circuit'].sum()),
                extra_control_symbols=ext['symbols'],total_frame_s=stats['uplink_s']+ext['symbols']/W,
                analog_fraction=fraction,status='feasible')
            assert abs(row['total_frame_s']-full_time)<1e-10
            errors=[]; directional=[]
            # Monte Carlo in update space is cheap. Do not reinterpret 128 draws
            # as independent training seeds or actual retained/forget quality.
            for draw in range(128):
                seed=plan['common_noise_seeds'][draw] if draw<3 else 610000+draw
                recovered=recover(design,lb,zero,zero,np.random.default_rng(seed))[:d]
                if controller:
                    recovered,info=control(recovered,quantized[active],g,1.)
                    directional.append(info['request_directional_progress'])
                else:recovered+=correction
                errors.append(float(np.linalg.norm(recovered-ideal)))
                if draw<3 and role=='physical_model':physical[name+'_noise'+str(seed)]=recovered
            row.update(application_rmse_mc=float(np.sqrt(np.mean(np.square(errors)))),
                application_error_mc_max=float(max(errors)),noise_draws=128,
                application_bound=lip*design.error_bound if controller else design.error_bound+bias,
                sketch_half_budget_pass=bool(bias<=eps/2) if extra else None)
            row['eligible_same_application_budget']=bool(row['application_bound']<=eps*(1+1e-5))
            arrays={k:np.asarray(getattr(design,k)) for k in ['users','analog_rms','digital_sqrt_power','receiver','normalized_channels','rho','quantization_bound','symbols','dimensions','digital_symbols']}
            np.savez_compressed(out/(name+'_design.npz'),**arrays)
        row['elapsed_seconds']=time.monotonic()-began
        rows.append(row);write(rowsfile,rows);np.savez_compressed(physfile,**physical);done.add(name)
        print('RADIO',json.dumps({k:v for k,v in row.items() if k!='heads'}),flush=True)
    solve('hybrid','hybrid',l,rho,eps/lip,controller=True)
    solve('oma_grid','orthogonal',l,rho,eps/lip,fractions=radio_plan['fractions'],controller=True)
    weights=np.asarray(geometry['controls']['control_alpha1']['weights'])
    weighted=learning.copy();weighted[active]+=weights[:,None]*quantized[active]
    correction=ideal-weighted.sum(axis=0)
    solve('genie_mixed_reference','mixed_aircomp',block(weighted),np.linalg.norm(block(weighted),axis=-1),eps,
        correction=correction,role='optimistic_free_feedback_reference')
    for rank in radio_plan['sketch_ranks']:
        generator=np.random.default_rng(radio_plan['sketch_seed'])
        projection=(generator.integers(0,2,size=(rank,d),dtype=np.int8)*2-1).astype(float)/np.sqrt(rank)
        ls=learning@projection.T; us=quantized@projection.T; gs=g@projection.T
        def encode_sketch(x):
            scale=(np.max(np.abs(x),axis=-1)/32767).astype(np.float32).astype(float)
            return np.rint(x/np.where(scale>0,scale,1)[:,None])*scale[:,None]
        ldecode=encode_sketch(ls);udecode=encode_sketch(us)
        for norm_mode in ['safe_norm','paid_norm']:
            name=f'feedback_mixed_r{rank}_{norm_mode}'
            if name in done:continue
            candidates={}
            for head in range(n):
                lhead=ldecode.copy();uhead=udecode.copy();lhead[head]=ls[head];uhead[head]=us[head]
                norms=np.linalg.norm(quantized[active],axis=1).astype(np.float32).astype(float)
                for j,user in enumerate(active):
                    if user==head:norms[j]=np.linalg.norm(quantized[user])
                A=lhead.sum(axis=0);center=A+uhead[active].sum(axis=0)
                N=np.vstack([uhead[active]/norms[:,None],-gs/np.linalg.norm(g)])
                lower=np.r_[norms,N[-1]@A]
                try:_,dual,_=project_rows(center,N,lower)
                except ValueError:continue
                w=(1+dual[:-1]/norms).astype(np.float32).astype(float)
                combo=learning.copy();combo[active]+=w[:,None]*quantized[active]
                corr=-dual[-1]*g/np.linalg.norm(g)
                bias=float(np.linalg.norm(combo.sum(axis=0)+corr-ideal))
                true_rho=np.linalg.norm(block(combo),axis=-1)
                if norm_mode=='paid_norm':
                    rb=np.nextafter(true_rho.astype(np.float32),np.float32(np.inf)).astype(float)
                else:
                    rb=rho.copy();rb[active]+=abs(w[:,None])*np.linalg.norm(u[active],axis=-1)
                candidates[head]=(combo,corr,bias,rb)
            # A design's input depends on the head's exact local sketch. Use the
            # same solver for every head rather than pretending inputs are global.
            incumbent=None; head_rows=[]
            for selected in range(n):
                if selected not in candidates:continue
                combo,corr,bias,rb=candidates[selected]
                def paid(head):
                    if head!=selected:return None
                    rf=np.zeros(n);circuit=np.zeros(n);symbols=0
                    for user in range(n):
                        if user==head:continue
                        bits=rank*16+32+32
                        if user in active:bits+=rank*16+32+32
                        if norm_mode=='paid_norm':bits+=blocks*32+32
                        count=int(np.ceil(bits/.5));duration=count/W
                        power=np.expm1(np.log(2)*.5)/channels[user,head]**2
                        if power>c['power_limit_w'] or duration>c['coherence_time_s']:return None
                        rf[user]+=power*duration;circuit[user]+=.01*duration;circuit[head]+=.01*duration;symbols+=count
                    count=int(np.ceil((32+32*len(active))/.5));duration=count/W
                    remote=np.delete(np.arange(n),head)
                    power=np.expm1(np.log(2)*.5)/np.min(channels[head,remote])**2
                    if power>c['power_limit_w'] or duration>c['coherence_time_s']:return None
                    rf[head]+=power*duration;circuit[head]+=.01*duration;circuit[remote]+=.01*duration;symbols+=count
                    return dict(symbols=symbols,rf=rf,circuit=circuit,energy=rf+circuit)
                # Keep each head trial reviewable, then select by ENERGY ONLY.
                trial=name+'_head'+str(selected)
                solve(trial,'mixed_aircomp',block(combo),rb,eps/2,extra=paid,bias=bias,correction=corr,
                    role='head_trial_physical')
                result=next(x for x in rows if x['name']==trial)
                head_rows.append(result['name'])
                if result['status']=='feasible' and (incumbent is None or result['energy_bound_j']<incumbent['energy_bound_j']):
                    incumbent=result
            if incumbent is None:
                row=dict(name=name,status='no_feasible_design_found',head_trials=head_rows,role='physical_model')
            else:
                row=dict(incumbent,name=name,role='physical_model',selected_trial=incumbent['name'],head_trials=head_rows)
                h=incumbent['head'];combo,corr,bias,rb=candidates[h]
                # Reconstruct only the selected head design; no second optimization.
                from wireless_init_v11 import Design
                with np.load(out/(incumbent['name']+'_design.npz')) as z:fields={k:z[k].copy() for k in z.files}
                des=Design(head=h,energy_j=incumbent['radio_energy_bound_j'],error_bound=incumbent['error_bound'],history=[],chi=0.,mode='mixed_aircomp',**fields)
                for seed in plan['common_noise_seeds']:
                    physical[name+'_noise'+str(seed)]=recover(des,block(combo),zero,zero,np.random.default_rng(seed))[:d]+corr
            rows.append(row);write(rowsfile,rows);np.savez_compressed(physfile,**physical);done.add(name)
            print('FEEDBACK',json.dumps({k:v for k,v in row.items() if k!='heads'}),flush=True)
    status('completed',candidates=len(rows)); print('COMPLETED_RADIO',flush=True)


if __name__=='__main__':main()
