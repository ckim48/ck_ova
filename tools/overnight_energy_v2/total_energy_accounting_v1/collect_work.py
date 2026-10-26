"""Read-only CPU audit of logged work. Run on server; no model/GPU imports.

Work units are operations, NOT joules. No training/queue/source mutations.
Teacher cache keys reproduce the saved implementation, using image+caption-slot
identities instead of loading private caption contents.
"""
from pathlib import Path
from collections import Counter, defaultdict
import json, hashlib, time, gzip, base64, argparse

ROOT = Path('/home/wnlab/Documents/Codex/2026-10-20/files-pasted-by-the-user-standalone/outputs/ova_CK')
R = ROOT/'results/overnight_energy_v2'
NAMES = ['proposed_fu_guard_cliperase','separate_aircomp_periodic',
         'aircomp_ofdma_periodic','sequential_digital_periodic']

def read(p):
    return json.loads(p.read_text())

def collect():
    hashes = {}
    def source(p):
        raw = p.read_bytes(); hashes[str(p.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    prefix_dir = R/'retrieval_long200_v3'
    plan = source(prefix_dir/'plan.json')
    warmup = source(prefix_dir/'records.json')[:20]
    proxy_meta = source(R/'proxy_objective_v2/proxies/analog/complete.json')
    proxy_path = next(p for p in [R/'proxy_objective_v2/proxy_records.json', R/'proxy_objective_v1/proxy_records.json']
                      if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == proxy_meta['schedule_sha256'])
    proxy = source(proxy_path)
    assert len(warmup) == len(proxy) == 20
    prefix = Counter()
    for records in [warmup, proxy]:
        for rec in records:
            for ids in rec['batches']:
                if len(ids) >= 2: prefix['FL_gradient_i%d_t%d' % (len(ids), len(ids))] += 1
    reqs = source(R/'fourway_cap100_v1/requests.json')
    reqmap = {r['id']: r for r in reqs}
    assert len(reqmap) == 45
    pcost = source(R/'fourway_cap100_v1/common_prefix_cost.json')
    result = dict(version='total_energy_accounting_v1', collected_unix=time.time(),
        scope='Network-wide communication + computing; flight/hover excluded provisionally. Joules require hardware calibration.',
        prefix=dict(work_units=dict(prefix),tx_bound_j=pcost['energy_j'],uplink_s=pcost['uplink_s'],
                    FL_rounds=20,auxiliary_proxy_rounds=20), methods={})
    for name in NAMES:
        d = R/('sequential_digital_cap100_v1' if name.startswith('sequential_') else 'fourway_cap100_v1')/name
        clip = name in [NAMES[0],NAMES[3]]
        counts = Counter(prefix); teachers = defaultdict(set); rows = []; fl = 20; fu = 0; previous_time = 0.
        for p in sorted((d/'transactions').glob('*.json')):
            x = source(p); delta = Counter()
            if x['kind'] in ['fl','joint']:
                deleted = {i for r in reqs if r['arrival_s'] <= previous_time+1e-12 for i in r['images']}
                for pool in plan['partitions']:
                    n = min(16, len(set(pool)-deleted)); assert n >= 2
                    delta['FL_gradient_i%d_t%d' % (n,n)] += 1
                fl += 1
            for m in x.get('local',[]):
                rid = m['request']; req = reqmap[rid]; fids = req['images']; rids = m['retained_images']; step = m['step']
                assert len(fids) == m['images']
                tag = 'cliperase' if clip else 'safe_proxy'
                delta['FU_forget_%s_gradient_i%d_t%d' % (tag,len(fids),5*len(fids))] += 1
                delta['FU_retain_%s_gradient_i%d_t%d' % (tag,len(rids),len(rids))] += 1
                calls = [(rid,rids,[(i,(21+step+i)%5) for i in rids])]
                if not clip:
                    # Proxy is fixed. Separate caches per requesting UAV are appropriate
                    # for distributed deployment, rather than simulator-wide cache reuse.
                    user = req.get('user',req.get('uav',rid.split('_u')[-1]))
                    calls.append(('proxy_'+str(user),fids,[(i,k) for i in fids for k in range(5)]))
                for key,ims,txts in calls:
                    for modality,items in [('image',ims),('text',txts)]:
                        cache = teachers[(key,modality)]
                        missing = [v for v in items if v not in cache]
                        # Teacher.features batches text in chunks of at most64.
                        batchsize = 64 if modality=='text' else max(1,len(missing))
                        for start in range(0,len(missing),batchsize):
                            delta['teacher_%s_forward_n%d' % (modality,len(missing[start:start+batchsize]))] += 1
                        cache.update(missing)
                fu += 1
            counts.update(delta)
            assert x['completed_FL_rounds'] == fl
            rows.append(dict(sequence=x['sequence'],round=fl,kind=x['kind'],
                tx_bound_j=x['cumulative_tx_energy_bound_j'],uplink_s=x['cumulative_uplink_occupied_s'],
                work_units_delta=dict(delta), cumulative_work_units=dict(counts),FU_request_steps=fu))
            previous_time=x['adaptation_elapsed_service_s']
        manifest=d/'manifest.json'
        status = source(manifest)['status'] if manifest.exists() else 'running' if rows else 'pending'
        if status=='completed_200FL_and_requests': assert fl==200 and fu==225
        result['methods'][name]=dict(status=status,transactions=rows,
            totals=dict(FL_rounds=fl,FU_request_steps=fu,work_units=dict(counts)),
            energy_total_j=None,energy_status='requires_hardware_and_radio_circuit_calibration')
    result['input_sha256']=hashes
    result['cautions']=[
        'Counts are reconstructed workload, not measured energy or time.',
        'Gradient operation includes full prompted encoders, autodiff/checkpoint recomputation and its specified loss. Frozen encoders still execute.',
        'Counts preserve actual objective differences; do not assign equal per-FU energy without profiling.',
        'Teacher cache behavior follows current code; memory transfers and storage need calibration.',
        'Server optimizer/evaluation wall time is not UAV computation time.',
        'Common initial20 and auxiliary proxy20 are charged once to each independent deployment.',
        'Offline reference training/MIA/plotting excluded from deployed protocol energy.',
        'Existing 0.10J cap constrains TX only, not newly requested total energy.',
        'Radio RX, TX circuits/PA inefficiency, control/downlink and head processing remain unmeasured; never silently replace with zero.',
        'Current uplink request clock is unchanged; this posthoc accounting does not establish end-to-end deadlines.']
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base64',action='store_true');a=p.parse_args()
    raw=json.dumps(collect(),ensure_ascii=False,allow_nan=False).encode()
    print('EXPORT_START'+base64.b64encode(gzip.compress(raw)).decode()+'EXPORT_END' if a.base64 else raw.decode())
