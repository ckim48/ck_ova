"""Fresh RemoteCLIP prefixes for two author-hosted UAV visual-language subsets."""
from common import *
import inspect, fcntl
from types import SimpleNamespace
import numpy as np
from acquire import DATA
FOLDERS={'GeoText-1652':'geotext','UAVReason':'uavreason'}

class Data:
    def __init__(self,dataset):
        self.dataset=dataset;directory=DATA/FOLDERS[dataset]
        ready=read(directory/'ready.json');assert ready['selection_sha256']==sha(directory/'selection.json')
        rows=read(directory/'selection.json')['items'];self.items=[];self.media={};self.captions=[];self.caption_ids=[];self.owners=[]
        for i,row in enumerate(rows):
            self.media[i]=ROOT/row['path'];ids=[]
            for text in row['captions']:
                assert text.strip();ids.append(len(self.captions));self.captions.append(text);self.owners.append(i)
            self.caption_ids.append(ids)
            self.items.append(dict(imgid=i,filename=Path(row['path']).name,split=row['split'],label=row['label'],source_key=row['source_key'],sentences=[dict(raw=t) for t in row['captions']]))
        self.owners=np.asarray(self.owners);self.caption_count=len(self.caption_ids[0]);assert all(len(c)==self.caption_count for c in self.caption_ids)
    def training_caption_ids(self,ids,round_number):return [self.caption_ids[int(i)][(round_number+int(i))%self.caption_count] for i in ids]
    def frames(self,index):
        from PIL import Image
        with Image.open(self.media[int(index)]) as image:return [image.convert('RGB')],[0]

def native_engine(base,count):
    """Change only positive-caption indexing, retaining original metric definitions."""
    evaluate=base.repeat.clone_function(inspect.unwrap(base.Engine.evaluate),
        [('repeat_interleave(5)','repeat_interleave(NCAP)',1),('*5','*NCAP',2),('arange(5)','arange(NCAP)',2)],{'NCAP':count})
    mia=base.repeat.clone_function(inspect.unwrap(base.Engine.mia_scores),[('*5','*NCAP',1),('arange(5)','arange(NCAP)',1)],{'NCAP':count})
    return type('NativeCaptionEngine',(base.Engine,),dict(evaluate=evaluate,mia_scores=mia))

def native_base(base,count):
    stream=SimpleNamespace(**base.clip.stream.__dict__)
    stream.evaluation=base.repeat.clone_function(base.clip.stream.evaluation,[('*5','*NCAP',1),('arange(5)','arange(NCAP)',1)],{'NCAP':count})
    clip=SimpleNamespace(**dict(base.clip.__dict__,stream=stream))
    return SimpleNamespace(**dict(base.__dict__,clip=clip))

def register_data(base,dataset,data,dest):
    from data_schedule import dirichlet_partition,make_schedule
    train=np.array([r['imgid'] for r in data.items if r['split']=='train']);validation=[r['imgid'] for r in data.items if r['split']=='validation']
    labels=np.array([r['label'] for r in data.items]);rng=np.random.default_rng(0)
    parts=[train[p] for p in dirichlet_partition(labels[train],10,.1,0,minimum_size=16)]
    records=make_schedule(parts,rounds=200,initial_rounds=20,requests=(),initial_fraction=1.,batch_size=16,seed=0)
    seen=[set() for _ in range(10)]
    for record in records[:20]:
        for user,batch in enumerate(record['batches']):seen[user].update(batch)
    eligible=[u for u in range(10) if len(seen[u])>=36];assert len(eligible)>=5
    cohort=[[] for _ in range(10)]
    for u in sorted(map(int,np.random.default_rng(231023).choice(eligible,5,replace=False))):
        pool=sorted(seen[u]);cohort[u]=sorted(map(int,rng.choice(pool,len(pool)//2,replace=False)))
    requests=base.clip.stream.make_requests(cohort);releases=[5,21,38,54,75,99,120,144,165]
    for r in requests:
        release=releases[r['burst']];r.update(arrival_s=None,deadline_s=None,release_after_adaptation_FL=release,release_before_total_FL_round=release+21,mission_site=release//36)
    assert len(requests)==45 and all(r['required_steps']==5 and set(r['images'])<=seen[r['user']] for r in requests)
    forgotten=sorted({i for r in requests for i in r['images']});fs=set(forgotten);references=[]
    for rec in records:
        pools=[sorted(set(p)-fs) for p in rec['retained']]
        batches=[np.random.default_rng(seed).choice(p,min(16,len(p)),replace=False).astype(int).tolist() for p,seed in zip(pools,rec['batch_seeds'])]
        references.append(dict(round=rec['round'],retained=pools,batches=batches))
    plan=read(SOURCE/'capera_seed0/plan.json')
    plan.update(dataset=dataset,version='iot_sensitivity_v1',partitions=[list(map(int,p)) for p in parts],training_images=len(train),
        validation_images=validation,official_test_images=[],unused_official_validation_images=[],
        forgotten_images=forgotten,forget_probe=forgotten,mia_member_images=forgotten[:min(128,len(forgotten))],
        mia_nonmember_images=validation[:min(128,len(forgotten))],gallery_images=sorted(validation+forgotten),
        caption_count=data.caption_count,annotation_sha256=sha(DATA/FOLDERS[dataset]/'selection.json'),
        scope='Fixed native-caption UAV subset; official test is excluded entirely and never downloaded for adaptation. Validation chosen by building/run before training. Not full benchmark reproduction.')
    for name,value in [('plan',plan),('requests',requests),('initial_records',records[:20]),('proxy_records',base.core.proxy_schedule(records,fs)),('reference_records',references)]:write(dest/(name+'.json'),value)
    return plan

def prepare(dataset):
    dest=OUT/'prepared'/(dataset.lower()+'_seed0')
    if (dest/'ready.json').exists():return read(dest/'ready.json')
    directory=DATA/FOLDERS[dataset]
    if not (directory/'ready.json').exists():
        launch=read(directory/'launch.json') if (directory/'launch.json').exists() else {}
        for _ in range(180):
            proc=Path('/proc')/str(launch.get('pid',0))/'cmdline'
            active=proc.exists() and b'acquire.py' in proc.read_bytes()
            if (directory/'ready.json').exists() or not active:break
            state(dest,stage='waiting_for_image_download',dataset=dataset);time.sleep(10)
    if not (directory/'ready.json').exists():
        from acquire import geotext,uavreason
        (geotext if dataset=='GeoText-1652' else uavreason)()
    event,base=load_base();data=Data(dataset)
    plan=read(dest/'plan.json') if (dest/'plan.json').exists() else register_data(base,dataset,data,dest)
    import torch
    from accounting import instrumented_engine
    assert torch.cuda.is_available();torch.set_num_threads(1)
    gpu=(ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a');fcntl.flock(gpu,fcntl.LOCK_EX)
    if (dest/'started.json').exists():raise RuntimeError('Prior preparation incomplete; preserve and review '+str(dest))
    write(dest/'started.json',dict(unix=time.time(),dataset=dataset))
    status=lambda **kw:state(dest,**kw)
    try:
        payload,metadata=event.recover.original.prepare_features(data,plan,ROOT/'checkpoints/remoteclip_vitb32_uav_v1',dest/'feature_cache',write,status)
        metadata.update(video_representation='One original RGB image per sample.',native_captions_per_image=data.caption_count,subset_manifest_sha256=sha(directory/'selection.json'))
        write(dest/'feature_cache/manifest.json',metadata)
        engine=instrumented_engine(native_engine(base,data.caption_count))(data,plan,payload)
        ids=read(dest/'initial_records.json')[0]['batches'][0]
        engine.start_work();g,loss=engine.gradient(engine.theta0,ids,1);work=engine.finish_work()
        _,next_loss=engine.gradient(engine.theta0-1e-5*g,ids,1)
        assert torch.isfinite(g).all() and g[:-1].numel()==20480 and float(g[:-1].norm())>0 and next_loss<=loss+1e-6
        write(dest/'preflight.json',dict(passed=True,native_captions=data.caption_count,gradient_norm=float(g[:-1].norm()),initial_loss=loss,small_step_loss=next_loss,compute=work))
        pure=base.repeat.clone_function(base.pure_trajectory,[],{'state':state})
        ch=np.load(GEOMETRY/'channels_by_site.npy')[0]
        warm,warmcost=pure(engine,plan,read(dest/'initial_records.json'),ch,'initial20',dest)
        proxy,proxycost=pure(engine,plan,read(dest/'proxy_records.json'),ch,'proxy20',dest)
        write(dest/'common_prefix_cost.json',{k:warmcost[k]+proxycost[k] for k in warmcost});engine.set_teacher(warm)
        ref,_=pure(engine,plan,read(dest/'reference_records.json'),ch,'reference200',dest)
        metrics,_=engine.evaluate(ref,None);mia,_=engine.mia_scores(ref)
        write(dest/'reference_metrics.json',dict(metrics=metrics,retained_mR_threshold=.95*metrics['retained']['mR'],reporting_only=True));write(dest/'reference_mia.json',mia)
        result=dict(status='ready',dataset=dataset,unix=time.time(),images=len(data.items),native_captions=data.caption_count,
                    checkpoint_sha256={stage:sha(dest/stage/'final.pt') for stage in ['initial20','proxy20','reference200']})
        write(dest/'ready.json',result);return result
    finally:base.base.parent.common.base.close()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=EXTRA_DATASETS);args=p.parse_args();print(prepare(args.dataset))
