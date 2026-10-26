"""Frozen B/32 features with two rank10 residual adapters (20480 coordinates).

No encoder backward. Cached features retain sample lineage; their existence is
not a claim of deletion. Every method uses the same new initial vector/cache.
"""
from pathlib import Path
from collections import Counter
import gc,hashlib,json,math,time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPModel,CLIPImageProcessor,CLIPTokenizer
from train import Engine as OriginalEngine

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def prepare_features(data,plan,checkpoint,dest,write,status):
 """Cold extraction once, shared execution but independently charged per arm."""
 dest.mkdir(parents=True,exist_ok=True)
 assert not (dest/'features.pt').exists(),'No silent cache replacement or retry'
 torch.set_num_threads(4);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
 ids=sorted(set(sum(plan['partitions'],[])+plan['gallery_images']+plan['mia_member_images']+plan['mia_nonmember_images']))
 assert set(ids).isdisjoint(plan['official_test_images'])
 cids=[c for i in ids for c in data.caption_ids[i]]
 data.processor=CLIPImageProcessor.from_pretrained(checkpoint,local_files_only=True)
 tokenizer=CLIPTokenizer.from_pretrained(checkpoint,local_files_only=True)
 enc=tokenizer(data.captions,return_tensors='pt',padding='max_length',truncation=True,max_length=75)
 data.tokens={k:v for k,v in enc.items() if k in ['input_ids','attention_mask']}
 torch.cuda.synchronize();start=time.perf_counter()
 model=CLIPModel.from_pretrained(checkpoint,local_files_only=True).float().cuda().eval()
 for p in model.parameters():p.requires_grad_(False)
 assert model.config.vision_config.patch_size==32 and model.config.projection_dim==512
 load_s=time.perf_counter()-start
 torch.cuda.reset_peak_memory_stats();rows=[];im=[];tx=[];image_s=text_s=0.
 with torch.no_grad():
  for modality,allids,batch_size,out in [('image',ids,64,im),('text',cids,128,tx)]:
   for j in range(0,len(allids),batch_size):
    batch=allids[j:j+batch_size];torch.cuda.synchronize();t=time.perf_counter()
    if modality=='image':z=model.get_image_features(pixel_values=data.pixels(batch,'cuda'))
    else:z=model.get_text_features(**data.texts(batch,'cuda'))
    assert z.shape==(len(batch),512) and torch.isfinite(z).all()
    z=F.normalize(z,dim=-1).cpu();out.append(z)
    torch.cuda.synchronize();elapsed=time.perf_counter()-t
    if modality=='image':image_s+=elapsed
    else:text_s+=elapsed
    rows.append(dict(modality=modality,first_index=j,count=len(batch),server_seconds=elapsed))
    status(stage='cold_feature_extraction',modality=modality,done=j+len(batch),total=len(allids))
  # Check cache against an independent deterministic forward on real inputs.
  a=F.normalize(model.get_image_features(pixel_values=data.pixels(ids[:2],'cuda')),dim=-1).cpu()
  b=F.normalize(model.get_text_features(**data.texts(cids[:5],'cuda')),dim=-1).cpu()
  assert torch.allclose(a,im[0][:2],rtol=2e-5,atol=2e-6)
  assert torch.allclose(b,tx[0][:5],rtol=2e-5,atol=2e-6)
  assert all(p.grad is None and not p.requires_grad for p in model.parameters())
  payload=dict(image_ids=torch.tensor(ids),caption_ids=torch.tensor(cids),images=torch.cat(im),texts=torch.cat(tx),logit_scale=model.logit_scale.detach().cpu())
 nparams=sum(p.numel() for p in model.parameters());peak=torch.cuda.max_memory_allocated()
 t=time.perf_counter();torch.save(payload,dest/'features.pt');write_s=time.perf_counter()-t
 provenance=dict(status='completed',backbone='CLIP ViT-B/32',backbone_parameters=nparams,cache_images=len(ids),cache_captions=len(cids),image_ids=ids,caption_ids=cids,
  caption_owner=[int(data.owners[c]) for c in cids],requested_image_ids=plan['forgotten_images'],official_test_features_not_extracted=True,
  image_server_seconds=image_s,text_server_seconds=text_s,model_load_server_seconds=load_s,cache_write_seconds=write_s,
  cache_bytes=(dest/'features.pt').stat().st_size,cache_sha256=digest(dest/'features.pt'),peak_GPU_bytes=peak,batches=rows,
  frozen_encoder_no_grad=True,deterministic_recheck=True,precision='float32',max_text_tokens=75,
  deletion_scope='Raw frozen feature archive retained for evaluation/audit, including requested images. Training excludes arrived deletion IDs. No physical cache erasure or certified deletion claim.',
  costing='Cold feature extraction is shared only for execution. Charge training-feature construction and cache work per independent deployment arm; evaluation/gallery-only feature work is separate. RTX3080 wall time is not UAV or energy measurement.')
 write(dest/'manifest.json',provenance)
 del model;gc.collect();torch.cuda.empty_cache()
 return payload,provenance

class CachedData:
 def __init__(self,original,payload):
  self.original=original;self.image_pos={int(x):i for i,x in enumerate(payload['image_ids'])};self.text_pos={int(x):i for i,x in enumerate(payload['caption_ids'])}
  self.images=payload['images'].cuda();self.text_embeddings=payload['texts'].cuda()
 def __getattr__(self,name):return getattr(self.original,name)
 def pixels(self,ids,device):return self.images[[self.image_pos[int(i)] for i in ids]].to(device)
 def texts(self,cids,device):return dict(input_ids=self.text_embeddings[[self.text_pos[int(i)] for i in cids]].to(device),attention_mask=None)

class ResidualAdapters:
 visual_dimension=10240
 text_dimension=10240
 def __init__(self,scale,device):
  gen=torch.Generator(device='cpu').manual_seed(0)
  blocks=[]
  for _ in range(2):
   down=torch.randn(10,512,generator=gen)/math.sqrt(512)
   up=torch.zeros(512,10)
   blocks.extend([down.flatten(),up.flatten()])
  self.initial_vector=torch.cat(blocks+[scale.detach().reshape(1).cpu()]).to(device)
  self.counts=Counter()
 def parts(self,theta):
  assert theta.numel()==20481
  return theta[:10240],theta[10240:20480],theta[-1]
 def apply(self,z,weights,modality):
  grad=torch.is_grad_enabled() and weights.requires_grad
  self.counts[modality+('_grad_samples' if grad else '_no_grad_samples')]+=len(z)
  self.counts[modality+'_calls']+=1
  down=weights[:5120].reshape(10,512);up=weights[5120:].reshape(512,10)
  return z+F.linear(F.gelu(F.linear(z,down)),up)
 def image_features(self,pixels,visual):return self.apply(pixels,visual,'image')
 def text_features(self,input_ids,attention_mask,text):return self.apply(input_ids,text,'text')

class Engine(OriginalEngine):
 def __init__(self,data,plan,payload):
  self.data=CachedData(data,payload);self.plan=dict(plan);self.device=torch.device('cuda')
  self.model=ResidualAdapters(payload['logit_scale'],self.device)
  self.theta0=self.model.initial_vector.detach().clone();self.teacher=None;self.teacher_images={};self.teacher_texts={}
  assert self.theta0.numel()-1==20480
  self.work_started=None
 def start_work(self):
  torch.cuda.synchronize();self.work_started=time.perf_counter();self.before=dict(self.model.counts)
 def finish_work(self):
  torch.cuda.synchronize();elapsed=time.perf_counter()-self.work_started
  counts={k:v-self.before.get(k,0) for k,v in self.model.counts.items()}
  return dict(server_seconds=elapsed,adapter_operations=counts,encoder_forward_calls=0,encoder_backward_calls=0,
   dense_forward_MACs=sum(v for k,v in counts.items() if k.endswith('_samples'))*10240,
   forward_MACs_scope='Two matrix products per modality sample only; backward/activation/normalization and CPU overhead excluded from this count. No conversion to joules without device calibration.')
