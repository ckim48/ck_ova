"""Read-only export of the separately registered RSICD experiment."""
from pathlib import Path
import argparse, json, time, hashlib, gzip, base64

ROOT = Path('/home/wnlab/Documents/Codex/2026-10-20/files-pasted-by-the-user-standalone/outputs/ova_CK')
CAMPAIGN = ROOT / 'results/overnight_energy_v2/rsicd_cap100_v1'
ORDER = ['proposed_fu_guard_cliperase', 'separate_aircomp_periodic',
         'aircomp_ofdma_periodic', 'sequential_digital_periodic']

def read(path):
    return json.loads(path.read_text())

def slim(x):
    retained = (x.get('evaluation') or {}).get('retained') or {}
    return dict(sequence=x['sequence'], round=x['completed_FL_rounds'],
                mR=retained.get('mR'), loss=retained.get('loss'),
                energy_j=x['cumulative_tx_energy_bound_j'],
                uplink_s=x['cumulative_uplink_occupied_s'],
                service_s=x['cumulative_elapsed_service_s'],
                requests=x['requests']['procedure_completed'])

def export():
    reference = CAMPAIGN / 'reference_metrics.json'
    payload = dict(exported_unix=time.time(), dataset='RSICD', cap_j=.10,
                   historical_preview=False, methods={}, target_mR=None,
                   scope='Single-seed complete-system comparison. TX energy bound and pure uplink; total communication-and-computation energy remains uncalibrated. Procedure completion is not deletion certification.')
    if reference.exists():
        payload['target_mR'] = read(reference)['retained_mR_threshold']
    for name in ORDER:
        d = CAMPAIGN / name
        rec = dict(status='pending', source=str(d.relative_to(ROOT)),
                   curve=[], transactions=[], files_sha256={})
        for f in sorted((d / 'evaluations').glob('*.json')):
            rec['curve'].append(slim(read(f)))
            rec['files_sha256'][str(f.relative_to(ROOT))] = hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted((d / 'transactions').glob('*.json')):
            rec['transactions'].append({k: v for k, v in slim(read(f)).items() if k not in ['mR', 'loss']})
            rec['files_sha256'][str(f.relative_to(ROOT))] = hashlib.sha256(f.read_bytes()).hexdigest()
        f = d / 'manifest.json'
        if f.exists():
            x = read(f)
            rec.update(status=x['status'], final=slim(x), quality=x.get('evaluation'),
                       request_metrics=x.get('requests'), joint_target=x.get('joint_target'))
            rec['files_sha256'][str(f.relative_to(ROOT))] = hashlib.sha256(f.read_bytes()).hexdigest()
        elif (d / 'latest.json').exists():
            rec.update(status='running', latest=slim(read(d / 'latest.json')))
        if (d / 'failure.json').exists():
            rec.update(status='failed', failure=read(d / 'failure.json'))
        if (d / 'mia.json').exists():
            rec['mia'] = read(d / 'mia.json')
        payload['methods'][name] = rec
    for n in ['protocol.json', 'preparation_checks.json', 'queue_registered.json',
              'launched.json', 'status.json', 'queue_status.json', 'summary.json',
              'finished.json', 'failure.json', 'common_prefix_cost.json', 'reference_metrics.json']:
        if (CAMPAIGN / n).exists():
            payload[n] = read(CAMPAIGN / n)
    return payload

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--base64', action='store_true')
    a = p.parse_args()
    value = json.dumps(export(), ensure_ascii=True)
    if a.base64:
        print('EXPORT_START' + base64.b64encode(gzip.compress(value.encode())).decode() + 'EXPORT_END')
    else:
        print(value)
