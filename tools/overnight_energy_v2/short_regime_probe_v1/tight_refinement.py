from pathlib import Path
import sys
exec(Path('tools/overnight_energy_v2/short_regime_probe_v1/polish.py').read_text().split('# Refine all cases')[0])
sys.path.insert(0,str(m.ROOT/'tools/overnight_energy_v2/partial_overlap_tx_v1'))
from overlap_solver import optimize_partial
h=base*10**.5;q,bits=m.encode(u,128);rho=np.linalg.norm(l,axis=1)[:,None]
c=dict(proto['config'],max_ao_iterations=3,receive_initializations=1,sic_residual_fraction=0.)
results=[];cache={}
for L in [640,768,1024,1536]:
 n=grid_solve(h,l,bits,L);best=None;positive=None;zero=None
 for la in np.unique(np.r_[128,np.linspace(128,L-1,17)]):
  dg=m.ofdma(bits[ids],h[ids,head],L-la)
  if dg is None:continue
  if la not in cache:cache[la]=grid_solve(h,l,np.zeros(10),la)
  a=cache[la]
  if a is not None:
   cand=dict(a,energy_j=a['energy_j']+dg['energy'],expected_energy_j=a['energy_j']+dg['energy'],analog_symbols=float(la),digital_symbols=float(L-la),occupied_ms=L/m.W*1000,peak_power_bound_w=max(a['peak_power_bound_w'],float(dg['power'].max())))
   if best is None or cand['energy_j']<best['energy_j']:best=cand
 for la in [128,256]:
  warm=[np.array([eps/np.sqrt(128/(2*la))*x]) for x in [.5,.999999]]
  for yes,grid in [(False,(0.,)),(True,(.025,.1,.25,.5,.75))]:
   r=optimize_partial(head,h,rho,bits[:,None],np.array([128]),np.array([la]),np.array([L]),eps,c,split_grid=grid,warm_receivers=warm)
   if r is None:continue
   rr={k:r[k] for k in ['tx_energy_bound_j','tx_energy_expected_j','payload_overlap_fraction','error_bound','minimum_payload_margin_bits']};rr['analog_symbols']=la
   if yes and (positive is None or rr['tx_energy_bound_j']<positive['tx_energy_bound_j']):positive=rr
   if not yes and (zero is None or rr['tx_energy_bound_j']<zero['tx_energy_bound_j']):zero=rr
 row=dict(channel_gain_db=10,topk=128,symbols=L,deadline_ms=L/1000,fu_codec_relative_l2_error=float(np.linalg.norm(q[ids]-u[ids])/np.linalg.norm(u[ids])),full_noma=n,air_ofdma=best,partial_noma=positive,zero_overlap_diagnostic=zero)
 results.append(row);m.write('tight_refinement.json',results);print(json.dumps(row),flush=True)
print('COMPLETE',flush=True)
