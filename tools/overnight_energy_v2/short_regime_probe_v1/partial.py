import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import importlib.util,pathlib,json,sys,time
import numpy as np
p=pathlib.Path(__file__).with_name('run.py');s=importlib.util.spec_from_file_location('probe',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
sys.path.insert(0,str(m.ROOT/'tools/overnight_energy_v2/partial_overlap_tx_v1'))
from overlap_solver import optimize_partial
proto=json.loads((m.OUT/'protocol.json').read_text());z=np.load('results/overnight_energy_v2/retrieval_fourway_v2/fixed_updates.npz')
l=z['learning'][:,:128];u=z['unlearning'][:,:128];rho=np.linalg.norm(l,axis=1)[:,None];head=proto['head'];eps=proto['epsilon'];c=dict(proto['config'],max_ao_iterations=3,receive_initializations=1,sic_residual_fraction=0.)
results=[];start=time.time()
for db,k,L in [(10,8,256),(10,32,512),(10,128,2048),(0,8,2048),(0,128,8192),(-10,8,8192)]:
    h=z['channels']*10**(db/20);q,bits=m.encode(u,k)
    best=None;nonoverlap=None
    for la in sorted(set([128,min(256,L//2)])):
        if la>=L:continue
        warm=[np.array([eps/np.sqrt(128/(2*la))*v]) for v in [.5,.999999]]
        for positive,grid in [(False,(0.,)),(True,(.025,.1,.25,.5,.75))]:
            r=optimize_partial(head,h,rho,bits[:,None],np.array([128]),np.array([la]),np.array([L]),eps,c,split_grid=grid,warm_receivers=warm)
            if r is None:continue
            record={key:r[key] for key in ['tx_energy_bound_j','tx_energy_expected_j','error_bound','minimum_payload_margin_bits','payload_overlap_fraction','maximum_overlap_power_bound_w','maximum_tail_power_w']}
            record['analog_symbols']=la
            if positive and (best is None or record['tx_energy_bound_j']<best['tx_energy_bound_j']):best=record
            if not positive and (nonoverlap is None or record['tx_energy_bound_j']<nonoverlap['tx_energy_bound_j']):nonoverlap=record
    results.append(dict(channel_gain_db=db,topk=k,symbols=L,positive_overlap=best,no_overlap_diagnostic=nonoverlap))
    m.write('partial_candidates.json',dict(scope='Six diagnostic cells, existing partial-overlap algorithm with additional receiver starts. Zero-overlap is a diagnostic, not a NOMA simultaneous win.',runtime_s=time.time()-start,cases=results))
    print(json.dumps(results[-1]),flush=True)
print('COMPLETE',time.time()-start,flush=True)
