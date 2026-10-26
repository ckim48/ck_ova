from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[3]
import re
import numpy as np
import torch
from PIL import Image
from transformers import CLIPImageProcessor,CLIPTokenizer
from functools import lru_cache

class RetrievalData:
 def __init__(self):
  self.root=ROOT/'data/rsicd_original_v1'
  self.items=json.loads((self.root/'dataset_rsicd.json').read_text())['images']
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
