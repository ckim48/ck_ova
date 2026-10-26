"""Terminal pilot CSV/ZIP and normalized JSON export; no training imports."""
from pathlib import Path
import argparse,base64,gzip,hashlib,importlib.util,json,time,zipfile
ROOT=Path('/home/wnlab/Documents/Codex/2026-10-20/files-pasted-by-the-user-standalone/outputs/ova_CK')
OUT=ROOT/'results/overnight_energy_v2/korea_joint_seed0_pilot_v2'
METHOD='proposed_fu_guard_cliperase'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def package(dataset):
    dest=OUT/f'{dataset.lower()}_seed0'
    assert (dest/'finished.json').exists() or (dest/'failure.json').exists(),'Export only terminal pilots'
    files=sorted(dest.rglob('*.json'))+[OUT/n for n in ['protocol.json','cpu_preflight.json','preparation.json']]
    signatures={str(p.relative_to(ROOT)):sha(p) for p in files}
    fingerprint=hashlib.sha256(json.dumps(signatures,sort_keys=True).encode()).hexdigest()
    cache=OUT/'exports'/f'{dataset.lower()}_seed0.json'
    if cache.exists():
        data=json.loads(cache.read_text())
        assert data['source_fingerprint']==fingerprint,'Terminal source changed; preserve prior export and investigate'
        archive=ROOT/data['csv_archive']['path']
        assert sha(archive)==data['csv_archive']['sha256']
        return data
    normal=load(ROOT/'tools/overnight_energy_v2/comparison_figures_two_datasets_v1/export_rsicd.py','pilot_normal')
    normal.CAMPAIGN=dest;normal.ORDER=[METHOD]
    data=normal.export();data.update(family='joint_pilot',dataset=dataset,seed=0,source_fingerprint=fingerprint,
        scope='Proposed-only seed0 round-triggered diagnostic. Historical schedules differ. Old mission deadlines are not comparable. TX bound and pure uplink only; total energy uncalibrated.')
    rec=data['methods'][METHOD]
    if (dest/'failure.json').exists():rec.update(status='failed',failure=json.loads((dest/'failure.json').read_text()))
    rec['details']={p.name:json.loads(p.read_text()) for p in (dest/METHOD).glob('*.json')}
    for p in dest.glob('*.json'):data[p.name]=json.loads(p.read_text())
    data['protocol.json']=json.loads((OUT/'protocol.json').read_text())
    pack=load(ROOT/'tools/overnight_energy_v2/korea_mission_exports_v1/package_results.py','pilot_csv')
    archive_info=pack.csv_package(data)
    folder=ROOT/Path(archive_info['path']).with_suffix('')
    rows=[dict(file=str(p.relative_to(ROOT)),sha256=sha(p),json=p.read_text()) for p in files if p.parent==dest or p.parent==OUT]
    pack.table(folder/'campaign_raw_documents.csv',rows)
    pack.table(folder/'all_source_sha256.csv',[dict(file=k,sha256=v) for k,v in signatures.items()])
    archive=ROOT/archive_info['path']
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(folder.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(folder))
    archive_info.update(bytes=archive.stat().st_size,sha256=sha(archive),base64_chars=4*((archive.stat().st_size+2)//3))
    data['csv_archive']=archive_info
    cache.parent.mkdir(exist_ok=True)
    with cache.open('x') as f:json.dump(data,f,ensure_ascii=True)
    return data


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['RSITMD','RSICD'],required=True)
    parser.add_argument('--base64',action='store_true');parser.add_argument('--chunk',type=int);parser.add_argument('--chunk-size',type=int,default=20000)
    a=parser.parse_args()
    if a.chunk is not None:
        archive=ROOT/f'results/overnight_energy_v2/csv_archive_v1/joint_pilot/{a.dataset.lower()}_seed0.zip'
        text=base64.b64encode(archive.read_bytes()).decode()[a.chunk:a.chunk+a.chunk_size]
        print('CHUNK_START'+text+'CHUNK_END')
    else:
        text=json.dumps(package(a.dataset),ensure_ascii=True)
        print('EXPORT_START'+base64.b64encode(gzip.compress(text.encode())).decode()+'EXPORT_END' if a.base64 else text)
