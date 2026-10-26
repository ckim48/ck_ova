"""Strengthened partial-overlap audit; exact overlap solver plus fixed tail."""
from run import ROOT,SRC,OUT,write,sha,make_inputs,np,json,time,os
from fast_solver import optimize_head
from overlap_solver import received_coefficients,audit

def main():
    with (OUT/'convex_started.json').open('x') as f:json.dump(dict(pid=os.getpid(),unix=time.time()),f)
    c=dict(json.loads((SRC/'plan.json').read_text())['config'],symbols_per_coordinate=32)
    z=np.load(SRC/'fixed_updates.npz');channels=z['channels']
    l,u,decoded,delta,payload,rho,dims,total=make_inputs(z['learning'],z['unlearning'],c)
    eps=max(c['absolute_tolerance_floor'],c['relative_tolerance']*float(np.linalg.norm(rho.sum(0))))
    la=2*dims
    cases=[('partial_positive_shorter',total-la),('partial_positive_full_allowance',total)]
    grid=[0.,.025,.05,.1,1/7,.25]
    # Peak limits imply the total battery limit for all phases in these cases.
    assert max(float(ld.sum()/c['symbol_rate_hz']*c['power_limit_w']) for _,ld in cases)<=c['available_energy_j']
    write('convex_plan.json',dict(created_unix=time.time(),script_sha256=sha(__file__),
        overlap_solver_sha256=sha(ROOT/'tools/overnight_energy_v2/retrieval_fourway_v2/fast_solver.py'),
        explanation='For each prescribed payload fraction the digital-only tail has exact minimum powers independent of overlap variables. Optimize overlap with the same convex feasibility initialization and AO as full-block Hybrid, then audit the combined phases. Battery constraints are redundant under these durations and peak limits.',
        payload_overlap_grid=grid,head_search='all 10',epsilon=eps,scope='Fixed updates only. No source or training change.'))
    rows=[];best={}
    for head in range(len(channels)):
        users=np.delete(np.arange(len(channels)),head);h=channels[users,head];H=h[:,None]
        for eta in grid:
            write('convex_status.json',dict(stage='running',head=head,eta=eta,pid=os.getpid(),unix=time.time()))
            d=optimize_head(head,channels,rho,np.zeros_like(delta),eta*payload,dims,la,eps,c,0.,'hybrid',digital_symbols=la)
            for name,ld in cases:
                name=name if eta else name.replace('partial_positive','zero_overlap_noma_ablation')
                r=dict(method=name,head=head,payload_overlap_fraction=eta,status='no_feasible_design_found',occupied_uplink_s=float(ld.sum()/c['symbol_rate_hz']))
                if d is not None:
                    pt=received_coefficients((1-eta)*payload[users],ld-la,h)/(H*H)
                    au=audit(d.analog_rms,d.digital_sqrt_power**2,pt,d.receiver,h,rho[users],payload[users],la,ld,eta,dims,eps,c)
                    if au['passed']:
                        r.update(au,status='feasible')
                        if name not in best or r['tx_energy_bound_j']<best[name]['tx_energy_bound_j']:
                            best[name]=r
                            np.savez_compressed(OUT/(name+'_convex_design.npz'),analog=d.analog_rms,digital_overlap=d.digital_sqrt_power**2,digital_tail=pt,receiver=d.receiver,users=users,analog_symbols=la,digital_symbols=ld)
                rows.append(r)
            write('convex_head_results.json',rows);write('convex_best.json',list(best.values()))
    write('convex_status.json',dict(stage='completed',pid=os.getpid(),unix=time.time(),designs=len(rows)))
    print(json.dumps(list(best.values())),flush=True)

if __name__=='__main__':main()
