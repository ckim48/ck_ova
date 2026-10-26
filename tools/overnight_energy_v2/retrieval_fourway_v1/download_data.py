from pathlib import Path
from html.parser import HTMLParser
import requests,zipfile,hashlib,json,time,traceback
root=Path.cwd();out=root/'results/overnight_energy_v2/retrieval_fourway_v1';dest=root/'data/rsitmd_original_v1';dest.mkdir(exist_ok=True)
class Form(HTMLParser):
 def __init__(self):super().__init__();self.url=None;self.params={}
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if tag=='form' and a.get('id')=='download-form':self.url=a['action']
  if tag=='input' and 'name' in a:self.params[a['name']]=a.get('value','')
def status(**x):(out/'data_status.json').write_text(json.dumps(dict(updated=time.time(),**x),indent=2))
try:
 status(status='downloading',source='official AMFMN RSITMD Google Drive release')
 s=requests.Session();r=s.get('https://drive.google.com/uc',params={'export':'download','id':'1NJY86TAAUd8BVs7hyteImv8I2_Lh95W6'},timeout=30);r.raise_for_status()
 f=Form();f.feed(r.text);assert f.url=='https://drive.usercontent.google.com/download'
 z=dest/'RSITMD.zip';partial=z.with_suffix('.partial');r=s.get(f.url,params=f.params,stream=True,timeout=60);r.raise_for_status();assert 'text/html' not in r.headers.get('Content-Type','')
 h=hashlib.sha256();size=0
 with partial.open('wb') as file:
  for chunk in r.iter_content(1024*1024):
   file.write(chunk);h.update(chunk);size+=len(chunk)
   if size%(20*1024*1024)==0:status(status='downloading',bytes=size)
 partial.replace(z)
 with zipfile.ZipFile(z) as archive:
  for info in archive.infolist():
   p=(dest/info.filename).resolve();assert p.is_relative_to(dest.resolve())
   if info.filename.endswith(('.jpg','.jpeg','.png','.tif','.tiff','.json')):archive.extract(info,dest)
 status(status='ready',archive_sha256=h.hexdigest(),bytes=size,json_files=[str(p.relative_to(root)) for p in dest.rglob('*.json')],image_count=sum(1 for p in dest.rglob('*') if p.suffix.lower() in ['.jpg','.jpeg','.png','.tif','.tiff']))
except Exception as e:status(status='failed',error=repr(e));traceback.print_exc();raise
