from common import *
import re
import numpy as np
import torch
from PIL import Image
from transformers import CLIPImageProcessor,CLIPTokenizer
from functools import lru_cache

class RetrievalData:
 def __init__(self):
  self.root=ROOT/'data/rsitmd_original_v1'
  self.items=json.loads((self.root/'dataset_RSITMD.json').read_text())['images']
  self.items.sort(key=lambda x:int(x['imgid']))
  assert [x['imgid'] for x in self.items]==list(range(len(self.items)))
  ckpt=ROOT/'checkpoints/clip-vit-base-patch16'
  self.processor=CLIPImageProcessor.from_pretrained(ckpt,local_files_only=True)
  tokenizer=CLIPTokenizer.from_pretrained(ckpt,local_files_only=True)
  self.captions=[];self.owners=[];self.caption_ids=[]
  for item in self.items:
   ids=[]
   assert len(item['sentences'])==5
   for sentence in item['sentences']:
    assert sentence['raw'].strip()
    ids.append(len(self.captions));self.captions.append(sentence['raw']);self.owners.append(item['imgid'])
   self.caption_ids.append(ids)
  enc=tokenizer(self.captions,return_tensors='pt',padding='max_length',truncation=True,max_length=75)
  self.tokens={k:v for k,v in enc.items() if k in ['input_ids','attention_mask']}
  self.owners=np.asarray(self.owners)
  self.paths={p.name:p for p in self.root.rglob('*') if p.suffix.lower() in ['.jpg','.jpeg','.png','.tif','.tiff']}
  assert all(x['filename'] in self.paths for x in self.items)
 @lru_cache(maxsize=128)
 def pixel(self,i):
  with Image.open(self.paths[self.items[int(i)]['filename']]) as im:
   return self.processor(images=im.convert('RGB'),return_tensors='pt')['pixel_values'][0]
 def pixels(self,ids,device):return torch.stack([self.pixel(int(i)) for i in ids]).to(device)
 def texts(self,cids,device):return {k:v[cids].to(device) for k,v in self.tokens.items()}
 def training_caption_ids(self,ids,round_number):return [self.caption_ids[int(i)][(round_number+int(i))%5] for i in ids]

def prepare():
 from data_schedule import dirichlet_partition,make_schedule
 from transport import link_matrix
 data=RetrievalData();rng=np.random.default_rng(0)
 original_train=np.array([x['imgid'] for x in data.items if x['split']=='train'])
 official_test=[x['imgid'] for x in data.items if x['split']=='test']
 order=rng.permutation(original_train);validation=sorted(map(int,order[:256]));train=order[256:]
 labels=np.array([re.sub(r'[_\d]+$','',Path(x['filename']).stem) for x in data.items])
 parts=dirichlet_partition(labels[train],10,.1,0,minimum_size=16)
 parts=[train[x] for x in parts]
 records=make_schedule(parts,rounds=40,initial_rounds=20,requests=(21,),initial_fraction=1.,
                       deletion_fraction=.5,requesting_users=5,batch_size=16,seed=0)
 forget=records[20]['forget'];forgotten=sorted(set().union(*map(set,forget)))
 assert set(forgotten).isdisjoint(validation+official_test)
 probe=sorted(map(int,rng.choice(forgotten,min(64,len(forgotten)),replace=False)))
 mia_members=sorted(map(int,rng.choice(forgotten,min(128,len(forgotten)),replace=False)))
 ref_records=[]
 for rec in records:
  pools=[sorted(set(pool)-set(forgotten)) for pool in rec['retained']]
  batches=[np.random.default_rng(seed).choice(pool,min(16,len(pool)),replace=False).astype(int).tolist()
           for pool,seed in zip(pools,rec['batch_seeds'])]
  ref_records.append(dict(round=rec['round'],retained=pools,batches=batches))
 c=dict(users=10,block_dimension=128,symbols_per_coordinate=16,quantization_bits=8,
   scale_metadata_bits=32,relative_tolerance=.2,absolute_tolerance_floor=1e-8,
   sic_residual_fraction=0.,area_m=500,reference_gain_db=-60,noise_dbm=-90,power_limit_w=.32,
   symbol_rate_hz=1e6,base_overhead_j=0.,head_overhead_j=0.,available_energy_j=.5,
   coherence_time_s=.01,round_uplink_deadline_s=2.,max_ao_iterations=8,receive_initializations=3)
 sources=['clip_model.py','data_schedule.py','wireless.py','transport.py',
   'tools/overnight_energy_v2/phase_split_grid.py',
   'tools/overnight_energy_v2/initialization_v11/wireless_init_v11.py',
   'tools/overnight_energy_v2/deletion_window_screen_v1/trajectory_transport.py',
   'tools/overnight_energy_v2/deadline_frontier_v1/run_frontier.py']
 plan=dict(version='retrieval_fourway_v1',created_unix=time.time(),config=c,seed=0,
   dataset='Original RSITMD: real image-caption pairs',official_train_images=len(original_train),
   training_images=len(train),validation_images=validation,official_test_images=official_test,
   test_policy='Test remains untouched until final evaluation after all registered arms finish; no hyperparameter selection from test.',
   annotation_sha256=sha(data.root/'dataset_RSITMD.json'),data_download=json.loads((OUT/'data_status.json').read_text()),
   partitions=[list(map(int,x)) for x in parts],forgotten_images=forgotten,forget_probe=probe,
   mia_member_images=mia_members,mia_nonmember_images=validation[:len(mia_members)],
   gallery_images=sorted(set(validation+probe)),
   methods=['proposed','hybrid_no_crossmodal','aircomp_only','aircomp_oma','oma_only'],
   rounds=40,initial_learning_rounds=20,request_round=21,request_users=[i for i,x in enumerate(forget) if x],
   unlearning_rounds=[21,22,23,24,25],batch_size=16,prompt_layers=8,prompt_tokens=2,
   learning_rate=.0035,optimizer='SGD without momentum; constant rate pilot',forget_weight=1.,crossmodal_weight=1.,
   eval_rounds=[0,5,10,15,20,21,25,30,35,40],
   quality_thresholds=dict(retained_mR_fraction_of_final_retrain=.95,forget_bidirectional_KL_max=.01,
     gate='joint target only after round25; both required; utility-only targets also reported'),
   utility_target_grid_mR=[.2,.3,.4,.5,.6,.7,.8],loss_target_grid=[2.,3.,4.,5.,6.],
   loss='Symmetric multi-positive image-caption contrastive CE; one of five real captions per image per round, deterministic cycle.',
   proposed_term='During five unlearning steps, preserve retained cross-modal pair distributions by symmetric KL(pre-request model || current model), on retained training batches. This replaces the previous projected Hessian P. Weight1 fixed before results.',
   attribution='AIR motivates relational cross-modal retention. This frozen-teacher relation KL is our adaptation, not an exact AIR reproduction and not a deletion certificate.',
   comparison='Five main arms: proposed has the extra cross-modal term; hybrid_no_crossmodal and all three baselines share identical base objective. Paired hybrid ablation isolates that term; no-crossmodal four-way isolates transport.',
   transmission='No receiver-specific correction. All schemes deliver the same additive target. AirComp-only combines learning and canonical Q8 unlearning in one shared-resource analog sum. Hybrid uses full shared block for analog plus digital, as manuscript.',
   codec='Q8 deletion vectors canonicalized identically across methods, including local head. OMA learns with lowest certified scalar precision meeting common analog error budget. Report quantization deviation separately.',
   delay_energy='Uplink TX waveform energy bound and occupied uplink time only; expectation reaccounted separately. Compute, decoder, SIC, control, downlink and flight excluded.',
   deadline_selection='CPU first-request fixed real updates: smallest multiplier in [16,32,64] for which all four physical schemes find a feasible design, then fixed for every arm and round. Failure is censored, never zero-energy success.',
   physical_changes='Blocks128 to keep longest frame below 10ms coherence; per-round TX allowance0.5J for relaxed-deadline comparison, equal for every scheme. Fixed hovering channel realization; perfect CSI/SIC and ideal coding.',
   scope='Single seed0,40-round initial experiment, not full paper protocol or a converged retraining oracle. No data arrivals in this controlled first comparison.',
   sources={n:sha(ROOT/n) for n in sources},runner_hashes={p.name:sha(p) for p in HERE.glob('*.py')},
   references=['https://doi.org/10.1145/3664647.3681280','https://doi.org/10.1145/3664647.3681421','https://github.com/xiaoyuan1996/AMFMN/tree/master/RSITMD'])
 write(OUT/'plan.json',plan);write(OUT/'records.json',records);write(OUT/'retrain_records.json',ref_records)
 np.save(OUT/'channels.npy',link_matrix(10,0,c))
 return data,plan,records,ref_records

