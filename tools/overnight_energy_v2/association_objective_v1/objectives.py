"""Prompt-only adaptations, not reproductions of CLIPErase or MultiDelete."""
import torch
import torch.nn.functional as F


def embedding_kl(student, teacher):
    return F.kl_div(F.log_softmax(student, dim=-1),
                    F.softmax(teacher.detach(), dim=-1), reduction='batchmean')


def retained_terms(images, texts, scale, labels, old_images, old_texts, kind):
    logits = scale.exp() * images @ texts.T
    ce = F.cross_entropy(logits, labels)
    if kind in ('retain_only', 'gradient_ascent'):
        consistency = ce * 0
    elif kind.startswith('cliperase'):
        consistency = embedding_kl(images, old_images) + embedding_kl(texts[labels], old_texts[labels])
    elif kind == 'neutral_balanced':
        old_logits = scale.detach().exp() * old_images @ old_texts.T
        consistency = embedding_kl(logits, old_logits)
        consistency = consistency + (images-old_images).square().sum(-1).mean()
        consistency = consistency + (texts[labels]-old_texts[labels]).square().sum(-1).mean()
    else:
        raise ValueError(kind)
    return ce, consistency


def forget_term(images, texts, scale, labels, kind, unrelated_target):
    matched = (images * texts[labels]).sum(-1)
    if kind == 'gradient_ascent':
        return -F.cross_entropy(scale.exp()*images@texts.T, labels)
    if kind.startswith('cliperase'):
        return matched.mean()
    if kind == 'neutral_balanced':
        return (matched-unrelated_target).square().mean()
    if kind == 'retain_only':
        return matched.mean()*0
    raise ValueError(kind)


def self_check():
    torch.manual_seed(123)
    a = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
    b = torch.randn(5, 8, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 1, 1, 3])
    olda, oldb = F.normalize(a.detach(), dim=-1), F.normalize(b.detach(), dim=-1)
    scale = torch.tensor(1., dtype=torch.float64)
    rows = []
    for kind in ['cliperase_paper', 'neutral_balanced']:
        x,y = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
        val = forget_term(x,y,scale,labels,kind,0.)
        ga,gb = torch.autograd.grad(val,(a,b))
        new = forget_term(F.normalize(a-1e-4*ga,dim=-1),F.normalize(b-1e-4*gb,dim=-1),scale,labels,kind,0.)
        assert new < val and ga.norm()>0 and gb.norm()>0
        _,cons = retained_terms(olda,oldb,scale,labels,olda,oldb,kind)
        assert abs(float(cons)) < 1e-10
        rows.append(dict(kind=kind,both_modalities_receive_gradients=True,forget_descent=True,teacher_consistency_zero=True))
    return rows


if __name__ == '__main__':
    import json
    print(json.dumps(self_check()))
