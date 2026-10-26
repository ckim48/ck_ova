from pathlib import Path
import urllib.request,hashlib,json,zipfile,time,re
ROOT=Path.cwd()
dest=ROOT/'data/rsicd_original_v1'
dest.mkdir(parents=True,exist_ok=True)
status=dest/'download_status.json'
def write(**kw):status.write_text(json.dumps(dict(unix=time.time(),**kw),indent=2))
def download(url,path):
 if path.exists():return
 tmp=path.with_suffix(path.suffix+'.partial')
 with urllib.request.urlopen(url,timeout=60) as response,tmp.open('wb') as out:
  while True:
   block=response.read(1024*1024)
   if not block:break
   out.write(block)
 tmp.replace(path)
try:
 write(stage='metadata')
 base='https://raw.githubusercontent.com/201528014227051/RSICD_optimal/master/'
 download(base+'dataset_rsicd.json',dest/'dataset_rsicd.json')
 pointer=urllib.request.urlopen(base+'RSICD_images.zip',timeout=60).read().decode()
 expected=re.search(r'oid sha256:([a-f0-9]+)',pointer).group(1)
 expected_size=int(re.search(r'size (\d+)',pointer).group(1))
 write(stage='downloading_images',expected_bytes=expected_size)
 archive=dest/'RSICD_images.zip'
 download('https://media.githubusercontent.com/media/201528014227051/RSICD_optimal/refs/heads/master/RSICD_images.zip',archive)
 assert archive.stat().st_size==expected_size
 h=hashlib.sha256()
 with archive.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 assert h.hexdigest()==expected
 write(stage='extracting',sha256=expected)
 with zipfile.ZipFile(archive) as z:
  for item in z.infolist():
   p=(dest/item.filename).resolve()
   assert str(p).startswith(str(dest.resolve())+'/'),item.filename
  z.extractall(dest)
 data=json.loads((dest/'dataset_rsicd.json').read_text())
 images=data['images']
 write(stage='completed',images=len(images),image_sha256=expected,
  annotation_sha256=hashlib.sha256((dest/'dataset_rsicd.json').read_bytes()).hexdigest(),
  source='https://github.com/201528014227051/RSICD_optimal',
  splits={s:sum(x['split']==s for x in images) for s in set(x['split'] for x in images)},
  caption_counts=sorted(set(len(x['sentences']) for x in images)))
except Exception as e:
 write(stage='failed',error=repr(e));raise
