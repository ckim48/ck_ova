"""Read-only source export; derived CSV/ZIP files only, no training imports."""
from pathlib import Path
import json,csv,hashlib,gzip,base64,argparse,importlib.util,zipfile,time,io
ROOT=Path('/home/wnlab/Documents/Codex/2026-10-20/files-pasted-by-the-user-standalone/outputs/ova_CK')
RESULTS=ROOT/'results/overnight_energy_v2'
ORDER=['proposed_fu_guard_cliperase','separate_aircomp_periodic','aircomp_ofdma_periodic','sequential_digital_periodic']

def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def read(p):return json.loads(p.read_text())
def flatten(obj,prefix=''):
 out={}
 if isinstance(obj,dict):
  for k,v in obj.items():out.update(flatten(v,prefix+'.'+str(k) if prefix else str(k)))
 else:out[prefix]=json.dumps(obj,ensure_ascii=True) if isinstance(obj,list) else obj
 return out
def table(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True);keys=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)

def get_export(family,dataset,seed):
 if family=='mission':
  m=load(ROOT/'tools/overnight_energy_v2/comparison_figures_two_datasets_v1/export_rsicd.py','mission_export')
  m.CAMPAIGN=RESULTS/'korea_mission_v1'/f'{dataset.lower()}_seed{seed}';d=m.export()
 elif dataset=='RSITMD' and seed==0:
  m=load(ROOT/'tools/overnight_energy_v2/comparison_figures_sequential_cap100_v2/export_results.py','baseline_export');d=m.export()
 else:
  m=load(ROOT/'tools/overnight_energy_v2/comparison_figures_two_datasets_v1/export_rsicd.py','seed_export')
  if seed:m.CAMPAIGN=RESULTS/'three_seed_cap100_v1'/f'{dataset.lower()}_seed{seed}'
  d=m.export()
 d.update(family=family,dataset=dataset,seed=seed)
 d['scope']='Complete-system comparison; raw independent seed results, no smoothing. TXbound and pureuplink; calibrated totalenergy unavailable. Missionclock includes prescribed motion/waiting but excludes localcompute/recovery/control/downlink. Procedures are not deletion guarantees.'
 # Preserve all quality, individual-request and task service diagnostics.
 for name,rec in d['methods'].items():
  ad=ROOT/rec['source'];rec['details']={}
  for f in ['manifest.json','mia.json','request_records.json','common_horizon.json','failure.json']:
   if (ad/f).exists():rec['details'][f]=read(ad/f)
  if (ad/'manifest.json').exists():
   obj=read(ad/'manifest.json');rec.update(quality=obj.get('evaluation'),request_metrics=obj.get('requests'),joint_target=obj.get('joint_target'))
  tx={p.stem:read(p) for p in sorted((ad/'transactions').glob('*.json'))}
  for row in rec['transactions']:
   raw=tx.get(f"{row['sequence']:04d}",{})
   row.update(kind=raw.get('kind'),site=raw.get('site'),mission_clock_s=raw.get('mission_clock_s'),
      pending=raw.get('requests',{}).get('pending'),arrived=raw.get('requests',{}).get('arrived'))
  for row in rec['curve']:
   f=ad/'evaluations'/f"{row['sequence']:04d}.json"
   if f.exists():
    raw=read(f);ev=raw.get('evaluation') or {};row.update(mission_clock_s=raw.get('mission_clock_s'),
      mean_forget_KL=ev.get('forget',{}).get('KL_retrain_to_candidate'),worst_request_KL=ev.get('worst_request_KL'))
 if family=='mission':
  d['mission_spec']=read(RESULTS/'korea_mission_v1/mission_spec.json')
 return d

def csv_package(data):
 dest=RESULTS/'csv_archive_v1'/data['family']/f"{data['dataset'].lower()}_seed{data['seed']}";dest.mkdir(parents=True,exist_ok=True)
 summary=[];index=[]
 for name,rec in data['methods'].items():
  ad=ROOT/rec['source'];md=dest/name
  summary.append(dict(method=name,status=rec['status'],**flatten(rec.get('final',{})),**flatten(rec.get('quality',{}),'quality'),**flatten(rec.get('mia',{}),'mia')))
  table(md/'curve.csv',rec['curve']);table(md/'transactions_summary.csv',rec['transactions'])
  rawrows=[];evaluations=[];transactions=[];work=[];perrequest=[];requests=[]
  files=sorted(ad.rglob('*.json'))
  for p in files:
   raw=read(p);key=str(p.relative_to(ROOT));digest=hashlib.sha256(p.read_bytes()).hexdigest()
   index.append(dict(method=name,file=key,sha256=digest,bytes=p.stat().st_size))
   rawrows.append(dict(file=key,sha256=digest,json=json.dumps(raw,ensure_ascii=True,separators=(',',':'))))
   if p.parent.name=='evaluations':
    evaluations.append(dict(file=key,**flatten(raw)))
    for rid,v in (raw.get('evaluation') or {}).get('per_request',{}).items():perrequest.append(dict(sequence=raw['sequence'],round=raw['completed_FL_rounds'],request=rid,**flatten(v)))
   elif p.parent.name=='transactions':transactions.append(dict(file=key,**flatten(raw)))
   elif p.parent.name=='work_attempts':work.append(dict(file=key,**flatten(raw)))
   elif p.name=='request_records.json':requests=[flatten(r) for r in raw]
  for filename,rows in [('raw_json_documents.csv',rawrows),('evaluations_full.csv',evaluations),('transactions_full.csv',transactions),('work_attempts.csv',work),('per_request_quality.csv',perrequest),('requests.csv',requests)]:table(md/filename,rows)
 table(dest/'final_summary.csv',summary);table(dest/'source_index.csv',index)
 (dest/'normalized.json').write_text(json.dumps(data,ensure_ascii=True),encoding='utf8')
 (dest/'README.txt').write_text('All recorded result JSON documents are losslessly retained in raw_json_documents.csv (one JSON cell per source file, SHA256 in source_index.csv). Nested arrays remain JSON cells in flattened tables. Checkpoints stay PT files. Missing energy calibration stays empty/null, never zero. Raw reports may contain both partial and final rows: use status and source file. No credential/log files are included.\n',encoding='utf8')
 archive=dest.parent/(dest.name+'.zip')
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(dest.rglob('*')):
   if p.is_file():z.write(p,p.relative_to(dest))
 return dict(path=str(archive.relative_to(ROOT)),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),base64_chars=4*((archive.stat().st_size+2)//3))

def assets():
 root=RESULTS/'korea_mission_v1';osm=read(ROOT/'data/korea_mission_daejeon_v1/osm_snapshot.json')
 features=[]
 for e in osm['elements']:
  if e['type']!='way' or 'geometry' not in e:continue
  tags=e.get('tags',{})
  if not any(k in tags for k in ['highway','waterway','natural','leisure']):continue
  features.append(dict(id=e['id'],tags=tags,geometry=[[round(v['lat'],7),round(v['lon'],7)] for v in e['geometry']]))
 data=dict(mission_spec=read(root/'mission_spec.json'),motion_checks=read(root/'motion_checks.json'),features=features,
   attribution='© OpenStreetMap contributors',license='ODbL',copyright_url='https://www.openstreetmap.org/copyright',
   trajectory_csv=(root/'trajectory.csv').read_text(),protocol=read(root/'protocol.json'),cpu_preflight=read(root/'cpu_preflight.json'))
 return data

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--family',choices=['baseline','mission']);p.add_argument('--dataset',choices=['RSITMD','RSICD']);p.add_argument('--seed',type=int,choices=[0,1,2]);p.add_argument('--base64',action='store_true');p.add_argument('--csv',action='store_true');p.add_argument('--assets',action='store_true');p.add_argument('--chunk',type=int);p.add_argument('--chunk-size',type=int,default=20000);a=p.parse_args()
 if a.chunk is not None:
  file=RESULTS/'csv_archive_v1'/a.family/f'{a.dataset.lower()}_seed{a.seed}.zip';s=base64.b64encode(file.read_bytes()).decode();value=s[a.chunk:a.chunk+a.chunk_size];print('CHUNK_START'+value+'CHUNK_END')
 else:
  data=assets() if a.assets else get_export(a.family,a.dataset,a.seed)
  if a.csv:data['csv_archive']=csv_package(data)
  value=json.dumps(data,ensure_ascii=True)
  print('EXPORT_START'+base64.b64encode(gzip.compress(value.encode())).decode()+'EXPORT_END' if a.base64 else value)
