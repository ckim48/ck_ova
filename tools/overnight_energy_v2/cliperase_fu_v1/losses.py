"""ACL 2025 CLIPErase Eq.(2)-(5) inspired prompt-only FU loss.

Paper implementation, not the inconsistent public repository's CE-ascent loss.
Embeddings are normalized by the existing engine. Retain pairs remain the
existing one-caption/image minibatch. Forget data include all five captions.
"""
import torch
import torch.nn.functional as F

WEIGHTS = dict(retention=1.0, forgetting=3.0, consistency=3.0)

def forget_loss(image, text, owner):
    owner = torch.as_tensor(owner, device=image.device, dtype=torch.long)
    assert len(owner) == len(text) and owner.min() >= 0 and owner.max() < len(image)
    # Each positive image-caption pair contributes once; unrelated captions do
    # not become false positives, and all five captions have equal weight.
    return (image[owner] * text).sum(-1).mean()

def retain_losses(image, text, old_image, old_text, log_scale):
    assert len(image) == len(text) and len(image) >= 2
    logits = log_scale.exp() * image @ text.T
    target = torch.arange(len(image), device=image.device)
    rm = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
    # Paper Eq.(4): distributions over embedding dimensions, not retrieval
    # logits or within-modality Gram matrices; teacher is detached.
    cm = F.kl_div(F.log_softmax(image, -1), F.softmax(old_image.detach(), -1), reduction='batchmean')
    cm = cm + F.kl_div(F.log_softmax(text, -1), F.softmax(old_text.detach(), -1), reduction='batchmean')
    return rm, cm

def self_check():
    torch.manual_seed(94)
    image = F.normalize(torch.randn(3, 7, dtype=torch.double), dim=-1).requires_grad_()
    text = F.normalize(torch.randn(15, 7, dtype=torch.double), dim=-1).requires_grad_()
    owner = torch.arange(3).repeat_interleave(5)
    value = forget_loss(image, text, owner)
    explicit = torch.stack([(image[i] * text[j]).sum() for j, i in enumerate(owner)]).mean()
    assert torch.allclose(value, explicit)
    permutation = torch.randperm(15)
    assert torch.allclose(value, forget_loss(image, text[permutation], owner[permutation]))
    assert torch.autograd.gradcheck(lambda a,b: forget_loss(a,b,owner), (image,text))
    gi,gt = torch.autograd.grad(value,(image,text))
    assert gi.norm()>0 and gt.norm()>0
    before=float(value.detach())
    after=float(forget_loss(image-.001*gi,text-.001*gt,owner).detach())
    assert after<before
    ri=image.detach().clone().requires_grad_();rt=text[:3].detach().clone().requires_grad_()
    oi=ri.detach().clone().requires_grad_();ot=rt.detach().clone().requires_grad_()
    rm,cm=retain_losses(ri,rt,oi,ot,torch.tensor(1.,dtype=torch.double))
    assert abs(float(cm.detach()))<1e-12
    assert torch.autograd.grad(cm,(oi,ot),allow_unused=True)==(None,None)
    _,shift=retain_losses(ri+.1*torch.randn_like(ri),rt,oi,ot,torch.tensor(1.,dtype=torch.double))
    assert shift>0
    assert torch.autograd.gradcheck(lambda a,b: sum(retain_losses(a,b,oi,ot,torch.tensor(1.,dtype=torch.double))), (ri,rt))
    return dict(passed=True,tests=['all_positive_caption_mapping','caption_permutation_invariance','both_encoder_gradient_paths','FM_descent_direction','zero_CM_at_teacher','teacher_stop_gradient','positive_CM_after_change','double_precision_gradcheck'],weights=WEIGHTS)
