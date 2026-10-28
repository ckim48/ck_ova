"""Check retained-pool eligibility and SalUn's fixed request mask on toy data."""
import math
import numpy as np
import torch
import torch.nn.functional as F
import learning_quality_v3 as s


def main():
 torch.manual_seed(7);image=F.normalize(torch.randn(8,4,dtype=torch.double),dim=-1);text=F.normalize(torch.randn(8,4,dtype=torch.double),dim=-1)
 theta=torch.cat([torch.randn(16,dtype=torch.double)*.05,torch.tensor([1.],dtype=torch.double)])
 class Data:
  def training_caption_ids(self,ids,k):return list(ids)
 class Engine:
  data=Data();work_started=None
  def __init__(self):self.salun_masks={};self.selection_losses={};self.selection_records=[]
  def features(self,v,ids,cids):
   matrix=torch.eye(4,dtype=torch.double)+v[:-1].reshape(4,4)
   return F.normalize(image[ids]@matrix,dim=-1),F.normalize(text[cids]@matrix,dim=-1),v[-1]
 e=Engine();g,meta=s.salun_gradient(e,theta,[0,1,2],[3,4,5,6],0,None);mask=e.salun_masks[(0,1,2)][0]
 assert torch.isfinite(g).all() and g[-1]==0 and torch.all(g[mask==0]==0)
 g2,meta2=s.salun_gradient(e,theta-.001*g,[0,1,2],[3,4,5,6],1,None)
 assert meta['mask_sha256']==meta2['mask_sha256'] and not meta2['mask_created']
 assert meta['random_label_shift']>0 and .4<=meta['mask_fraction']<=.6
 s.CONFIG={'selection':4};e.selection_losses={0:{0:100.,1:2.,2:4.,3:1.,4:3.}}
 a=s.select_batch(e,theta,[1,2,3,4,5,6],21,0);b=s.select_batch(e,theta,[1,2,3,4,5,6],21,0)
 assert a==b and len(set(a))==4 and 0 not in a and 0 not in e.selection_losses[0]
 r=dict(passed=True,masked_gradient_zero_outside_saliency=True,request_mask_reused=True,wrong_pair_labels=True,selection_deterministic_and_retained_only=True)
 s.write(s.OUT/'unit_checks.json',r);print(r)


if __name__=='__main__':main()
