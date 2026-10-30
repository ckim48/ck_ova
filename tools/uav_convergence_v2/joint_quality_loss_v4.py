"""Final development screen: soften FL confidence while retaining ranking signal."""
import argparse
from pathlib import Path
import time
import joint_quality_control_v1 as first

ROOT,read,write,sha=first.ROOT,first.read,first.write,first.sha
OUT=ROOT/'results/joint_quality_loss_v4_20261030'
BASE_ENGINE=first.quality.engine_wrapper
VARIANTS={'control':first.spec()}
for rho in [None,1.]:
    for smooth in [.02,.05,.1,.2]:
        VARIANTS[f'smooth{smooth}_rho{rho}']=dict(first.spec(rho=rho),smoothing=smooth,rank_weight=0.)
for weight in [.25,.5]:
    VARIANTS[f'rank{weight}']=dict(first.spec(rho=1.),smoothing=0.,rank_weight=weight)


def register():
    path=OUT/'protocol.json'
    if path.exists():
        p=read(path);assert p['source_sha256']==sha(Path(__file__));return p
    p=dict(registered_unix=time.time(),source_sha256=sha(Path(__file__)),first_source_sha256=sha(Path(first.__file__)),
        variants=VARIANTS,seed=0,datasets=first.audit.DATASETS,
        motivation='Prior screens show excessive distribution drift with improved ranking. Test training loss smoothing or a hard-negative margin objective, rather than posthoc temperature fitting to a reporting reference. This changes actual adapter training; logit scale remains frozen.',
        smoothing='Symmetric local contrastive cross entropy with identical fixed label smoothing in both retrieval directions. Original unsmoothed per-example training CE still updates the priority cache, so priority scores have the original meaning.',
        rank='Mix symmetric CE with symmetric hardest-negative hinge at margin1 in original logit units. Weight0.25/0.5, with rho1 uniform-gradient bound.',
        protocol='All11 settings shared by datasets; same exact200-round/225-FU protocol. Preserve all earlier screens. Adaptive seed0 development only; no test/reference data in updates, no guarantee of joint leadership. Extra scalar loss operations charged on physical rerun.')
    write(path,p);return p


def engine_wrapper(parent):
    base=BASE_ENGINE(parent)
    class Engine(base):
        def gradient(self,theta,ids,k,kind='ce'):
            cfg=first.CURRENT
            if kind!='ce' or not (cfg.get('smoothing',0) or cfg.get('rank_weight',0)):
                return super().gradient(theta,ids,k,kind)
            import torch
            import torch.nn.functional as F
            if len(ids)<2:return torch.zeros_like(theta),0.
            cids=self.data.training_caption_ids(ids,k);variable=theta.detach().requires_grad_(True)
            im,tx,scale=self.features(variable,ids,cids);z=scale.exp()*(im@tx.T)
            target=torch.arange(len(ids),device=theta.device)
            per=.5*(F.cross_entropy(z,target,reduction='none')+F.cross_entropy(z.T,target,reduction='none'))
            epsilon=cfg.get('smoothing',0)
            loss=.5*(F.cross_entropy(z,target,label_smoothing=epsilon)+F.cross_entropy(z.T,target,label_smoothing=epsilon))
            weight=cfg.get('rank_weight',0)
            if weight:
                negative=z.masked_fill(torch.eye(len(ids),device=theta.device,dtype=torch.bool),float('-inf'))
                hinge=.5*(F.relu(1.+negative.max(1).values-z.diag()).mean()+F.relu(1.+negative.max(0).values-z.diag()).mean())
                loss=(1-weight)*loss+weight*hinge
            g=torch.autograd.grad(loss,variable)[0].detach();g[-1]=0
            user=self.sample_owner[int(ids[0])];cache=self.selection_losses.setdefault(user,{})
            for i,v in zip(ids,per.detach().cpu().tolist()):cache[int(i)]=float(v)
            ops=16*len(ids)+12*len(ids)**2+(16*len(ids)**2 if weight else 0)
            first.quality.infra.lib.infra.charge(self,ids,ops)
            return g,float(loss.detach())
    return Engine


def install():
    first.OUT=OUT;first.register=register;first.VARIANTS=VARIANTS;first.quality.engine_wrapper=engine_wrapper


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','screen','physical']);p.add_argument('--dataset',choices=first.audit.DATASETS);p.add_argument('--variant')
    a=p.parse_args();install()
    if a.action=='register':register()
    elif a.action=='physical':
        import joint_quality_physical_v2 as physical
        physical.run('loss',a.variant)
    else:
        for ds in [a.dataset] if a.dataset else first.audit.DATASETS:first.screen(ds)
