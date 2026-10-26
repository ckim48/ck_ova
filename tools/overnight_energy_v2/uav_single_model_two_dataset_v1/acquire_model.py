"""Download only the fixed public CLIP B/32 files, recording byte hashes."""
from pathlib import Path
import hashlib,json,time,urllib.request,traceback
ROOT=Path(__file__).resolve().parents[3]
DEST=ROOT/'checkpoints/clip-vit-base-patch32-uav-v1'
OUT=ROOT/'results/overnight_energy_v2/uav_single_model_two_dataset_v1'
REV='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
FILES=['config.json','merges.txt','preprocessor_config.json','pytorch_model.bin','special_tokens_map.json','tokenizer.json','tokenizer_config.json','vocab.json']
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)
def main():
 DEST.mkdir(parents=True,exist_ok=True);items={}
 for name in FILES:
  p=DEST/name
  if not p.exists():
   url=f'https://huggingface.co/openai/clip-vit-base-patch32/resolve/{REV}/{name}'
   tmp=p.with_suffix(p.suffix+'.download')
   with urllib.request.urlopen(url,timeout=120) as r,tmp.open('wb') as f:
    while True:
     chunk=r.read(1024*1024)
     if not chunk:break
     f.write(chunk)
   tmp.replace(p)
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
  items[name]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
  write(OUT/'model_download_status.json',{'stage':'downloading','completed_files':items,'unix':time.time()})
 config=json.loads((DEST/'config.json').read_text())
 assert config['vision_config']['patch_size']==32 and config['projection_dim']==512
 write(OUT/'model_provenance.json',{'repository':'openai/clip-vit-base-patch32','revision':REV,'checkpoint':str(DEST.relative_to(ROOT)),'files':items,'completed_unix':time.time(),'config':config})
 write(OUT/'model_download_status.json',{'stage':'completed','completed_files':items,'unix':time.time()})
if __name__=='__main__':
 try:main()
 except Exception as e:
  write(OUT/'model_download_failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'unix':time.time()});raise
