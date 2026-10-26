from pathlib import Path
import json,time
__file__='tools/overnight_energy_v2/minimum_latency_v2/run.py'
source=Path(__file__).read_text()
exec(source.split('protocol=dict')[0])
rows=json.loads((OUT/'results.json').read_text());before=json.loads(json.dumps(rows));checks=[]
for db in [0,10]:
 H=base[ids,head]*10**(db/20)
 eng=ConvexOverlap(H,learning[ids],radio_eps)
 for row in rows:
  if row['channel_gain_db']!=db or not row['partial_noma']:continue
  cap=row['energy_cap_j'];best=row['partial_noma'];oldbest=best.copy()
  la0=best['analog_symbols'];b0=int(round(best['fu_overlap_fraction']*1056))
  for la in range(max(128,la0-64),la0+65,4):
   for nb in range(max(1,b0-48),min(1055,b0+48)+1,2):
    bo=np.where(bu[ids]>0,nb,0.);bt=bu[ids]-bo
    a=eng.solve(la,bo)
    if a is None or a['energy_j']>=cap:continue
    def evaluate(L):
     d=noma_tail(H,bt,L-la)
     if d is None:return None
     return dict(a,energy_j=a['energy_j']+d['energy_j'],expected_energy_j=a['expected_energy_j']+d['energy_j'],
        peak_power_w=max(a['peak_power_w'],d['peak_power_w']),digital_tail_symbols=int(L-la),
        fu_overlap_fraction=float(nb/1056))
    x=bsearch(evaluate,cap,la+1,hi=best['symbols'])
    if x and (x['symbols'],x['energy_j'])<(best['symbols'],best['energy_j']):best=compact(x)
  row['partial_noma']=best
  row['partial_time_saving_vs_ofdma_percent']=100*(1-best['symbols']/row['air_ofdma']['symbols'])
  row['partial_time_saving_vs_separate_noma_percent']=100*(1-best['symbols']/row['separate_noma']['symbols'])
  checks.append(dict(db=db,cap_j=cap,coarse_symbols=oldbest['symbols'],refined_symbols=best['symbols'],
      analog_symbols=best['analog_symbols'],overlap_bits_per_request=round(best['fu_overlap_fraction']*1056)))
  write('refined_results.json',rows)
  print(json.dumps(checks[-1]),flush=True)
# Independent boundary and audited constraint checks for the full-overlap search.
for row in rows:
 if not row['full_noma']:continue
 H=base[ids,head]*10**(row['channel_gain_db']/20);eng=ConvexOverlap(H,learning[ids],radio_eps)
 L=row['full_noma']['symbols'];cap=row['energy_cap_j']
 x=eng.solve(L,bu[ids]);p=eng.solve(L-1,bu[ids])
 assert x and x['energy_j']<=cap*(1+1e-6)
 assert p is None or p['energy_j']>cap*(1+1e-7)
 checks.append(dict(full_noma_boundary_pass=True,db=row['channel_gain_db'],cap_j=cap,symbols=L,
                    preceding_energy_j=None if p is None else p['energy_j']))
for row in rows:
 for k in ['full_noma','partial_noma','air_ofdma','separate_noma','airair','ofdma_only']:
  x=row[k]
  if not x:continue
  assert x['energy_j']<=row['energy_cap_j']*(1+1e-6)
  assert x['total_error_bound']<=total_eps*(1+4e-6)
  assert x['peak_power_w']<=P*(1+4e-6)
write('refinement_checks.json',dict(passed=True,checks=checks,runtime_s=time.time()-t0,
    search='Local analog-length +-64 symbols /4, overlap payload +-48 bits /2 around coarse winners. Independent full-overlap preceding-symbol checks.'))
print('REFINEMENT_COMPLETE',flush=True)
