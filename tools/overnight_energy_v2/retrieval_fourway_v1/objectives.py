import torch
import torch.nn.functional as F

def contrastive(images,texts,scale,image_ids,text_owner_ids):
 logits=scale.exp()*images@texts.T
 positive=image_ids[:,None]==text_owner_ids[None,:]
 assert positive.any(1).all() and positive.any(0).all()
 # Sum positive probabilities; multiple captions of one image never become negatives.
 a=torch.logsumexp(logits,1)-torch.logsumexp(logits.masked_fill(~positive,-torch.inf),1)
 b=torch.logsumexp(logits,0)-torch.logsumexp(logits.masked_fill(~positive,-torch.inf),0)
 return (a.mean()+b.mean())/2

def relation_kl(images,texts,teacher_images,teacher_texts,scale):
 logits=scale.exp()*images@texts.T
 old=scale.exp()*teacher_images@teacher_texts.T
 a=F.kl_div(F.log_softmax(logits,1),F.softmax(old,1),reduction='batchmean')
 b=F.kl_div(F.log_softmax(logits.T,1),F.softmax(old.T,1),reduction='batchmean')
 return (a+b)/2

def self_check():
 torch.manual_seed(122)
 x=F.normalize(torch.randn(4,8),dim=-1);y=F.normalize(torch.randn(4,8),dim=-1)
 ids=torch.arange(4);s=torch.tensor(1.)
 loss=contrastive(x,y,s,ids,ids)
 duplicate=contrastive(x,y.repeat_interleave(2,0),s,ids,ids.repeat_interleave(2))
 assert torch.allclose(loss,duplicate,atol=1e-6)
 assert abs(float(relation_kl(x,y,x,y,s)))<1e-6
 w=x.clone().requires_grad_(True);g=torch.autograd.grad(contrastive(w,y,s,ids,ids),w)[0]
 assert contrastive(w+.0001*g,y,s,ids,ids)>loss
 return dict(multi_positive_duplicate_invariance=True,teacher_identity_KL_zero=True,forget_ascent_sign=True)

