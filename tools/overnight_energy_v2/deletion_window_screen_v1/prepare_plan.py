import copy,hashlib,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
OUT=ROOT/'results/overnight_energy_v2/deletion_window_screen_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((ROOT/'results/overnight_energy_v2/convergence_energy_delay_v1/plan.json').read_text())
family=ROOT/'results/overnight_energy_v2/burst_joint_v3/hybrid'
base=json.loads((family/'schedule.json').read_text())
first=base[60];assert first['round']==61
actual=[u for u,x in enumerate(first['forget']) if x];assert actual==[5,6]
extra=np.random.default_rng(6108).permutation([u for u in range(10) if u not in actual]).tolist()[:3]
new_forgets={u:[] for u in range(10)}
for u in actual:new_forgets[u]=list(first['forget'][u])
for u in extra:
    used=set().union(*(set(r['batches'][u]) for r in base[:60]))
    deleted=set().union(*(set(r['forget'][u]) for r in base[:60]))
    available=sorted(used-deleted)
    new_forgets[u]=np.random.default_rng(610800+u).choice(available,int(len(available)*old['config']['deletion_fraction']),replace=False).astype(int).tolist()
all_new=set().union(*map(set,new_forgets.values()))
records=[]
for original in base[60:65]:
    rec=copy.deepcopy(original)
    for u in range(10):
        rec['retained'][u]=sorted(set(rec['retained'][u])-set(new_forgets[u]))
        rec['forget'][u]=new_forgets[u] if rec['round']==61 else []
        r=rec['retained'][u]
        rec['batches'][u]=np.random.default_rng(rec['batch_seeds'][u]).choice(r,min(old['config']['batch_size'],len(r)),replace=False).astype(int).tolist()
        assert not set(rec['batches'][u]).intersection(new_forgets[u])
    rec['deleted']=sorted(set(rec['deleted'])|all_new)
    records.append(rec)
assert [x['round'] for x in records]==list(range(61,66))
OUT.mkdir(exist_ok=False)
(OUT/'records.json').write_text(json.dumps(records))
old['config']['request_rounds']=[61]
old['methods']=['proposed','aircomp_oma','hybrid_no_coupling','aircomp_only','oma_only']
old['scope']='Five-round counterfactual deletion window (61-65) from the same historical round-60 observer checkpoint; K=5 at round 61, excluded from later training. Costs exclude the historical prefix. This is a screening experiment, not a new complete 100-round trajectory.'
old['warm_checkpoint']='results/overnight_energy_v2/public_history_fu_v19/observer/round_060.pt'
old['warm_checkpoint_sha256']=sha(ROOT/old['warm_checkpoint'])
old['channel_source']='results/overnight_energy_v2/burst_joint_v3/hybrid/partition.npz'
old['channel_source_sha256']=sha(ROOT/old['channel_source'])
old['records_sha256']=sha(OUT/'records.json')
old['historical_schedule_sha256']=sha(family/'schedule.json')
old['runner_sha256']={p.name:sha(p) for p in HERE.glob('*.py')}
old['screen_decision']='Inspect matched validation targets, costs, completion and request handling after all five methods. Do not extend automatically. A higher one-round energy cost requires an observed quality or feasibility benefit to justify a new longer study. Do not label mere failure to find a design as global infeasibility.'
(OUT/'plan.json').write_text(json.dumps(old,indent=2)+'\n')
print(json.dumps({'status':'prepared','window':[61,65],'active_users':sorted(actual+extra),'warm_checkpoint_sha256':old['warm_checkpoint_sha256']}))
