"""Terminal-only new-model results; lossless raw JSON CSV, no training imports."""
from pathlib import Path
import argparse,base64,gzip,hashlib,importlib.util,json,zipfile
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/overnight_energy_v2/uav_single_model_two_dataset_v1'
FAMILY='uav_b32_adapter_seed0_v1'
ORDER=['proposed_fu_guard_cliperase','separate_aircomp_periodic','aircomp_ofdma_periodic','sequential_digital_periodic']
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def package(dataset):
 dest=OUT/f'{dataset.lower()}_seed0'
 assert (dest/'finished.json').exists() or (dest/'failure.json').exists(),'Only terminal dataset exports'
 rootfiles=['protocol.json','cpu_preflight.json','model_provenance.json','development_actual_batch_smoke.json','launched.json','started.json','registration_failure_001.json']
 files=sorted(dest.rglob('*.json'))+[OUT/n for n in rootfiles if (OUT/n).exists()]
 signatures={str(p.relative_to(ROOT)):sha(p) for p in files}
 fingerprint=hashlib.sha256(json.dumps(signatures,sort_keys=True).encode()).hexdigest()
 cache=OUT/'exports'/f'{dataset.lower()}_seed0.json'
 if cache.exists():
  d=read(cache);assert d['source_fingerprint']==fingerprint,'Source changed; preserve existing export and investigate'
  assert sha(ROOT/d['csv_archive']['path'])==d['csv_archive']['sha256']
  return d
 normal=load(ROOT/'tools/overnight_energy_v2/comparison_figures_two_datasets_v1/export_rsicd.py','uav_normal');normal.CAMPAIGN=dest;normal.ORDER=ORDER
 d=normal.export();d.update(dataset=dataset,seed=0,family=FAMILY,source_fingerprint=fingerprint,
  scope='One new frozen CLIP B32 + residual rank10 model; four methods, same round-triggered requests and common fresh model-specific prefix/reference. Two datasets use the same protocol. n=1: no seed SD/CI. Whole-system comparison because objectives differ. No claim of onboard training or measured total UAV energy.')
 for method,rec in d['methods'].items():
  ad=dest/method;rec['details']={p.name:read(p) for p in ad.glob('*.json')}
  if (dest/'failure.json').exists() and not (ad/'manifest.json').exists():rec.update(status='not_run_dataset_failed',failure=read(dest/'failure.json'))
  tx={p.stem:read(p) for p in (ad/'transactions').glob('*.json')}
  for r in rec['transactions']:
   raw=tx.get(f"{r['sequence']:04d}",{});r.update(kind=raw.get('kind'),site=raw.get('site'),mission_clock_s=raw.get('mission_clock_s'))
  rec['prefix_curve']=[dict(round=r['round'],mR=r['evaluation']['retained']['mR'],loss=r['evaluation']['retained']['loss']) for r in read(dest/'initial20/rounds.json') if r.get('evaluation')] if (dest/'initial20/rounds.json').exists() else []
 for p in dest.glob('*.json'):d[p.name]=read(p)
 d['feature_cache_manifest']=read(dest/'feature_cache/manifest.json') if (dest/'feature_cache/manifest.json').exists() else None
 d['protocol.json']=read(OUT/'protocol.json');d['model_provenance.json']=read(OUT/'model_provenance.json')
 d['fresh_prefix']={k:read(dest/k/'finished.json') if (dest/k/'finished.json').exists() else None for k in ['initial20','proxy20','reference200']}
 pack=load(ROOT/'tools/overnight_energy_v2/korea_mission_exports_v1/package_results.py','uav_csv');info=pack.csv_package(d)
 folder=ROOT/Path(info['path']).with_suffix('')
 # Include *all* dataset documents: prefix rounds, cached-feature lineage,
 # work attempts, failures and final evaluations, including nested lists.
 pack.table(folder/'all_raw_json_documents.csv',[dict(file=str(p.relative_to(ROOT)),sha256=sha(p),json=p.read_text()) for p in files])
 pack.table(folder/'all_source_sha256.csv',[dict(file=n,sha256=h) for n,h in signatures.items()])
 for kind in ['initial20','proxy20','reference200']:
  pack.table(folder/(kind+'_rounds.csv'),[dict(file=str(p.relative_to(ROOT)),**pack.flatten(read(p))) for p in sorted((dest/kind/'rounds').glob('*.json'))])
 archive=ROOT/info['path']
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(folder.rglob('*')):
   if p.is_file():z.write(p,p.relative_to(folder))
 with zipfile.ZipFile(archive) as z:assert z.testzip() is None
 info.update(bytes=archive.stat().st_size,sha256=sha(archive),base64_chars=4*((archive.stat().st_size+2)//3))
 d['csv_archive']=info;cache.parent.mkdir(exist_ok=True)
 with cache.open('x') as f:json.dump(d,f,ensure_ascii=True)
 return d
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['RSITMD','RSICD'],required=True);p.add_argument('--base64',action='store_true');p.add_argument('--chunk',type=int);p.add_argument('--chunk-size',type=int,default=20000);a=p.parse_args()
 if a.chunk is not None:
  archive=ROOT/f'results/overnight_energy_v2/csv_archive_v1/{FAMILY}/{a.dataset.lower()}_seed0.zip'
  text=base64.b64encode(archive.read_bytes()).decode()[a.chunk:a.chunk+a.chunk_size];print('CHUNK_START'+text+'CHUNK_END')
 else:
  text=json.dumps(package(a.dataset),ensure_ascii=True);print('EXPORT_START'+base64.b64encode(gzip.compress(text.encode())).decode()+'EXPORT_END' if a.base64 else text)
