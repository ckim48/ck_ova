"""Read-only export. Run on the ova_CK server; writes no training files."""
from pathlib import Path
import argparse,json,time,hashlib,gzip,base64

ORDER=['proposed_fu_guard_cliperase','separate_aircomp_periodic','aircomp_ofdma_periodic','sequential_digital_periodic']
ROOT=Path('/home/wnlab/Documents/Codex/2026-10-20/files-pasted-by-the-user-standalone/outputs/ova_CK')

def read(path):return json.loads(path.read_text())
def slim(x):
    ev=x.get('evaluation') or {};r=ev.get('retained') or {}
    return dict(sequence=x['sequence'],round=x['completed_FL_rounds'],mR=r.get('mR'),loss=r.get('loss'),
        energy_j=x['cumulative_tx_energy_bound_j'],uplink_s=x['cumulative_uplink_occupied_s'],
        service_s=x['cumulative_elapsed_service_s'],requests=x['requests']['procedure_completed'])

def export(historical=False):
    root=ROOT/'results/overnight_energy_v2';campaign=root/'fourway_cap100_v1'
    mapping={n:campaign/n for n in ORDER}
    mapping['sequential_digital_periodic']=root/'sequential_digital_cap100_v1/sequential_digital_periodic'
    if historical:
        mapping=dict(proposed_fu_guard_cliperase=root/'fu_protected_aircomp_v1/parallel_cliperase_fu_guard',
            separate_aircomp_periodic=root/'streaming_requests_v1/separate_aircomp_periodic',
            aircomp_ofdma_periodic=root/'streaming_requests_v1/aircomp_ofdma_periodic')
    payload=dict(exported_unix=time.time(),cap_j=.05 if historical else .10,
        historical_preview=historical,methods={},target_mR=.17688802083333333,
        scope='Single-seed complete-system comparison. TX upper bounds and pure uplink only. Not deletion certification.')
    for name,d in mapping.items():
        rec=dict(status='pending',source=str(d.relative_to(ROOT)),curve=[],transactions=[],files_sha256={})
        for f in sorted((d/'evaluations').glob('*.json')):
            rec['curve'].append(slim(read(f)))
            rec['files_sha256'][str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted((d/'transactions').glob('*.json')):
            x=read(f);rec['transactions'].append({k:v for k,v in slim(x).items() if k not in ['mR','loss']})
        f=d/'manifest.json'
        if f.exists():
            x=read(f);rec['status']=x['status'];rec['final']=slim(x)
            rec['files_sha256'][str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
        elif (d/'latest.json').exists():rec['status']='running';rec['latest']=slim(read(d/'latest.json'))
        if (d/'failure.json').exists():rec['status']='failed';rec['failure']=read(d/'failure.json')
        payload['methods'][name]=rec
    for n in ['status.json','queue_status.json','finished.json','launched.json']:
        if (campaign/n).exists():payload[n]=read(campaign/n)
    seq=root/'sequential_digital_cap100_v1'
    for n in ['protocol.json','preflight.json','queue_registered.json','launched.json','status.json','finished.json']:
        if (seq/n).exists():payload['sequential_'+n]=read(seq/n)
    return payload

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--historical',action='store_true');p.add_argument('--base64',action='store_true');a=p.parse_args()
    value=json.dumps(export(a.historical),ensure_ascii=False)
    if a.base64:print('EXPORT_START'+base64.b64encode(gzip.compress(value.encode())).decode()+'EXPORT_END')
    else:print(value)
