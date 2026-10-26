"""Author-hosted UAV RGB/text subsets, fixed before any learned result.

GeoText: four evenly spaced drone views per official training building.
UAVReason: evenly spaced frames per official training run; full runs held out.
No captions generated, no synthetic replacement images, no official test use.
"""
from common import *
import collections, concurrent.futures, io, re, tarfile, zipfile, zlib
import requests
import numpy as np
DATA=ROOT/'data/uav_extra_v1'

def fetch(url,headers=None):
    for attempt in range(5):
        try:
            r=requests.get(url,headers=headers,timeout=(20,120));r.raise_for_status();return r
        except requests.RequestException:
            if attempt==4:raise
            time.sleep(min(2**attempt,10))

def download(repo,name,path):
    if path.exists() and path.stat().st_size>1024:return
    response=fetch(f'https://huggingface.co/datasets/{repo}/resolve/main/{name}?download=true')
    path.parent.mkdir(parents=True,exist_ok=True);path.with_suffix('.part').write_bytes(response.content);path.with_suffix('.part').replace(path)

class RangeFile(io.RawIOBase):
    def __init__(self,url,size):self.url=url;self.size=size;self.pos=0;self.cache_start=0;self.cache=b''
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        assert 0<=self.pos<=self.size;return self.pos
    def read(self,n=-1):
        if n<0:n=self.size-self.pos
        n=min(n,self.size-self.pos)
        if n==0:return b''
        if not (self.cache_start<=self.pos and self.pos+n<=self.cache_start+len(self.cache)):
            end=min(self.size-1,self.pos+max(n,65536)-1)
            r=fetch(self.url,{'Range':f'bytes={self.pos}-{end}'})
            assert r.status_code==206 and r.headers['Content-Range'].startswith(f'bytes {self.pos}-'),(r.status_code,r.headers.get('Content-Range'))
            self.cache_start=self.pos;self.cache=r.content
        start=self.pos-self.cache_start;self.pos+=n;return self.cache[start:start+n]

def geotext():
    directory=DATA/'geotext';download('truemanv5666/GeoText1652_Dataset','train.json',directory/'train.json')
    grouped=collections.defaultdict(list)
    for r in read(directory/'train.json'):
        if re.fullmatch(r'image-\d+\.jpeg',Path(r['image']).name):grouped[r['image']].append(r['caption'])
    buildings=collections.defaultdict(list)
    for path in grouped:buildings[Path(path).parent.name].append(path)
    keys=sorted(buildings);validation=set(np.random.default_rng(0).permutation(keys)[:64]);selected=[]
    for building in keys:
        paths=sorted(buildings[building],key=lambda p:int(re.search(r'image-(\d+)',p).group(1)))
        assert len(paths)>=4
        for i in np.linspace(0,len(paths)-1,4).round().astype(int):
            path=paths[i];assert len(grouped[path])==3
            selected.append(dict(source_key=path,path=str((directory/'images'/path).relative_to(ROOT)),
                captions=grouped[path],label=building,split='validation' if building in validation else 'train'))
    write(directory/'selection.json',dict(policy='Four evenly spaced original drone-view filenames per official training building; seed0 chooses 64 entire validation buildings; official test untouched.',items=selected))
    def get_building(building):
        items=[r for r in selected if r['label']==building]
        if all((ROOT/r['path']).exists() for r in items):return
        url=f'https://huggingface.co/datasets/truemanv5666/GeoText1652_Dataset/resolve/main/images/train/{building}.tar.gz?download=true'
        payload=fetch(url).content
        with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as archive:
            files={Path(m.name).name:m for m in archive.getmembers() if m.isfile()}
            for row in items:
                member=files[Path(row['source_key']).name];p=ROOT/row['path'];p.parent.mkdir(parents=True,exist_ok=True)
                p.write_bytes(archive.extractfile(member).read())
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for i,_ in enumerate(pool.map(get_building,keys)):
            if i%25==0:state(directory,stage='images',done_buildings=i+1,total_buildings=len(keys))
    verify_media(directory,selected,'GeoText-1652')

def uavreason():
    directory=DATA/'uavreason'
    for split in ['train','test']:
        download('jarvissun/UAVReason_vqa',f'llava_vqa_scene_caption_{split}.jsonl',directory/(split+'.jsonl'))
    records=[json.loads(s) for s in (directory/'train.jsonl').read_text().splitlines()]
    test={r['image'][0] for r in (json.loads(s) for s in (directory/'test.jsonl').read_text().splitlines())}
    runs=collections.defaultdict(list)
    for r in records:
        assert r['image'][0] not in test
        runs[r['meta']['scene']].append(r)
    held={'interval5_AMtown02','interval5_HKisland02'};assert held<=runs.keys()
    selected=[]
    for scene in sorted(runs):
        rows=sorted(runs[scene],key=lambda r:float(r['meta']['stem']))
        count=128 if scene in held else 160;assert len(rows)>=count
        for i in np.linspace(0,len(rows)-1,count).round().astype(int):
            r=rows[i];captions=[v['value'] for v in r['conversations'] if v['from']=='gpt'];assert len(captions)==1
            source=r['image'][0].lstrip('/').removeprefix('UAVScenes/')
            selected.append(dict(source_key=source,path=str((directory/'images'/source).relative_to(ROOT)),captions=captions,
                label=re.sub(r'(?:_GNSS)?(?:_Evening|\d+)$','',scene),run=scene,
                split='validation' if scene in held else 'train'))
    write(directory/'selection.json',dict(policy='160 uniformly spaced frames per official training run, except two entire validation runs with128 each. Official test untouched. Validation run-disjoint, not guaranteed geographic-disjoint. Native released machine-enriched captions, not newly generated.',items=selected))
    url='https://huggingface.co/datasets/sijieaaa/UAVScenes/resolve/main/interval5_CAM_LIDAR.zip?download=true'
    with zipfile.ZipFile(RangeFile(url,28682115865)) as archive:infos={i.filename:i for i in archive.infolist()}
    import struct
    def get_image(row):
        p=ROOT/row['path']
        if p.exists():return
        info=infos[row['source_key']];start=info.header_offset
        # Read the local header followed by precisely the declared compressed data.
        f=RangeFile(url,28682115865);f.seek(start);header=f.read(30)
        assert header[:4]==b'PK\x03\x04';name_len,extra_len=struct.unpack_from('<HH',header,26)
        f.seek(start+30+name_len+extra_len);compressed=f.read(info.compress_size)
        payload=zlib.decompress(compressed,-15) if info.compress_type==zipfile.ZIP_DEFLATED else compressed
        assert len(payload)==info.file_size and zlib.crc32(payload)==info.CRC,row['source_key']
        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(payload)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for i,_ in enumerate(pool.map(get_image,selected)):
            if i%50==0:state(directory,stage='images',done=i+1,total=len(selected))
    verify_media(directory,selected,'UAVReason')

def verify_media(directory,selected,name):
    from PIL import Image
    hashes={};pixels={};split_hashes=collections.defaultdict(set)
    for row in selected:
        p=ROOT/row['path']
        with Image.open(p) as image:
            image.load();assert min(image.size)>32
            ph=hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()
        hashes[row['source_key']]=sha(p);split_hashes[row['split']].add(ph)
    assert split_hashes['train'].isdisjoint(split_hashes['validation']),'Exact duplicate pixel leak'
    write(directory/'ready.json',dict(status='ready',dataset=name,images=len(selected),
        split_counts=dict(collections.Counter(r['split'] for r in selected)),image_sha256=hashes,
        captions_per_image=sorted(set(len(r['captions']) for r in selected)),selection_sha256=sha(directory/'selection.json'),exact_pixel_train_validation_disjoint=True))
    state(directory,stage='ready',images=len(selected))

if __name__=='__main__':
    import argparse,traceback
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['geotext','uavreason']);a=p.parse_args()
    try:globals()[a.dataset]()
    except Exception as e:
        write(DATA/a.dataset/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
