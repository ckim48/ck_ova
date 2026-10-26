"""Export the four comparable RSITMD seed0 results after three controls terminate."""
from pathlib import Path
import argparse,base64,gzip,hashlib,importlib.util,json,zipfile
ROOT=Path('/home/wnlab/Documents/Codex/2026-10-20/files-pasted-by-the-user-standalone/outputs/ova_CK')
OUT=ROOT/'results/overnight_energy_v2/korea_round_rsitmd_seed0_v3'
PILOT=ROOT/'results/overnight_energy_v2/korea_joint_seed0_pilot_v2'
PROPOSED='proposed_fu_guard_cliperase'
ORDER=[PROPOSED,'separate_aircomp_periodic','aircomp_ofdma_periodic','sequential_digital_periodic']
FAMILY='rsitmd_round_seed0_v3'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def package():
    assert (OUT/'finished.json').exists() or (OUT/'failure.json').exists(),'Terminal only'
    dest=OUT/'rsitmd_seed0';prior=PILOT/'rsitmd_seed0'
    files=list(dest.rglob('*.json'))+list(prior.glob('*.json'))+list((prior/PROPOSED).rglob('*.json'))
    files += [OUT/n for n in ['protocol.json','cpu_preflight.json','launched.json','started.json','finished.json','failure.json'] if (OUT/n).exists()]
    files=sorted(set(files));signatures={str(p.relative_to(ROOT)):sha(p) for p in files}
    fingerprint=hashlib.sha256(json.dumps(signatures,sort_keys=True).encode()).hexdigest()
    cache=OUT/'exports/rsitmd_seed0.json'
    if cache.exists():
        d=read(cache);assert d['source_fingerprint']==fingerprint,'Changed source; preserve original export'
        assert sha(ROOT/d['csv_archive']['path'])==d['csv_archive']['sha256']
        return d
    normal=load(ROOT/'tools/overnight_energy_v2/comparison_figures_two_datasets_v1/export_rsicd.py','round_normal')
    normal.CAMPAIGN=dest;normal.ORDER=ORDER[1:];d=normal.export()
    existing=read(PILOT/'exports/rsitmd_seed0.json')
    assert existing['methods'][PROPOSED]['status']=='completed_200FL_and_requests'
    for name in ['plan.json','requests.json']:
        assert read(dest/name)==read(prior/name),'Comparison inputs differ: '+name
    d['methods'][PROPOSED]=existing['methods'][PROPOSED]
    d['methods']={k:d['methods'][k] for k in ORDER}
    for method,rec in d['methods'].items():
        ad=ROOT/rec['source'];rec['details']={p.name:read(p) for p in ad.glob('*.json')}
        if (ad/'failure.json').exists():rec.update(status='failed',failure=read(ad/'failure.json'))
        if (ad/'manifest.json').exists():
            raw=read(ad/'manifest.json');rec.update(quality=raw.get('evaluation'),request_metrics=raw.get('requests'),joint_target=raw.get('joint_target'))
        tx={p.stem:read(p) for p in (ad/'transactions').glob('*.json')}
        for row in rec['transactions']:
            raw=tx.get(f"{row['sequence']:04d}",{})
            row.update(kind=raw.get('kind'),site=raw.get('site'),mission_clock_s=raw.get('mission_clock_s'))
    d.update(family=FAMILY,dataset='RSITMD',seed=0,source_fingerprint=fingerprint,
             scope='Same round-triggered requests,814 gallery,map channels,initial/proxy/reference,200FL/45requests. Proposed is the completed unchanged seed0 pilot; only three serial controls newly run. Existing per-method objectives differ. No seed SD/CI. TX upper bound and pure uplink, no calibrated total energy or old calendar deadline comparison.')
    for p in dest.glob('*.json'):d[p.name]=read(p)
    d['protocol.json']=read(OUT/'protocol.json')
    d['campaign_terminal']=read(OUT/'finished.json') if (OUT/'finished.json').exists() else read(OUT/'failure.json')
    d['reused_proposed_export_sha256']=sha(PILOT/'exports/rsitmd_seed0.json')
    pack=load(ROOT/'tools/overnight_energy_v2/korea_mission_exports_v1/package_results.py','round_csv')
    info=pack.csv_package(d);folder=ROOT/Path(info['path']).with_suffix('')
    campaign=[p for p in files if p.parent in [OUT,dest,prior]]
    pack.table(folder/'campaign_raw_documents.csv',[dict(file=str(p.relative_to(ROOT)),sha256=sha(p),json=p.read_text()) for p in campaign])
    pack.table(folder/'all_source_sha256.csv',[dict(file=f,sha256=h) for f,h in signatures.items()])
    archive=ROOT/info['path']
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(folder.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(folder))
    info.update(bytes=archive.stat().st_size,sha256=sha(archive),base64_chars=4*((archive.stat().st_size+2)//3))
    d['csv_archive']=info;cache.parent.mkdir(exist_ok=True)
    with cache.open('x') as f:json.dump(d,f,ensure_ascii=True)
    return d

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base64',action='store_true');p.add_argument('--chunk',type=int);p.add_argument('--chunk-size',type=int,default=20000);a=p.parse_args()
    if a.chunk is not None:
        archive=ROOT/f'results/overnight_energy_v2/csv_archive_v1/{FAMILY}/rsitmd_seed0.zip'
        text=base64.b64encode(archive.read_bytes()).decode()[a.chunk:a.chunk+a.chunk_size]
        print('CHUNK_START'+text+'CHUNK_END')
    else:
        text=json.dumps(package(),ensure_ascii=True)
        print('EXPORT_START'+base64.b64encode(gzip.compress(text.encode())).decode()+'EXPORT_END' if a.base64 else text)
