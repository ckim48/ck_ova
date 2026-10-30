"""Magnitude/direction controls after the registered initial tradeoff screen."""
import argparse
from pathlib import Path
import time
import numpy as np
import joint_quality_control_v1 as first

ROOT,read,write,sha=first.ROOT,first.read,first.write,first.sha
OUT=ROOT/'results/joint_quality_focused_v3_20261030'
BASE_FL=first.fl_vectors
VARIANTS={'control':first.spec()}
for batch in [16,24]:
    for alpha in [.25,.5,1.]:
        VARIANTS[f'norm_b{batch}_a{alpha}']=dict(first.spec(batch=batch),mode='norm',alpha=alpha)
for rho in [.25,.5,1.]:
    VARIANTS[f'orthogonal_rho{rho}']=dict(first.spec(rho=rho),mode='orthogonal')
VARIANTS.update(batch24_bound50=first.spec(batch=24,rho=.5),batch24_bound100=first.spec(batch=24,rho=1.),
                bound100_retain75=first.spec(rho=1.,retention=.75),uniform_after80=first.spec(late_uniform=80))


def register():
    path=OUT/'protocol.json'
    if path.exists():
        p=read(path);assert p['source_sha256']==sha(Path(__file__));return p
    p=dict(registered_unix=time.time(),source_sha256=sha(Path(__file__)),first_source_sha256=sha(Path(first.__file__)),
        variants=VARIANTS,seed=0,datasets=first.audit.DATASETS,
        rationale='Initial rho1/late-uniform candidates retained mR while reducing KL, but did not lead fixed-reference KL. Test whether the selected gradient magnitude, rather than its useful direction, drives distribution drift. The negative-context screen is also retained.',
        norm='Compute uniform16 gU and priority gH. Normalize gH to norm(gU), then use (1-alpha)*gU+alpha*normalized(gH). Its norm is at most norm(gU), including zero-vector handling.',
        orthogonal='Remove from gH-gU its component along gU; cap the orthogonal remainder at rho*norm(gU), and add it to gU. This changes only the extra priority direction, not the uniform descent component. It is not a KL or utility guarantee.',
        other='Two intermediate batch24/bounded settings, one rho1 with retention0.75, and a predeclared middle FL-switch round80. All shared between datasets; no per-dataset selection.',
        evaluation='Same exact-screen protocol, unchanged baseline/reference/gallery. All candidates reported. These are adaptive development stages, not preregistered independent confirmations.',
        compute='Both gradient passes and direction/norm arithmetic charged if physically rerun.')
    write(path,p);return p


def fl_vectors(engine,theta,plan,queue,t,fl):
    cfg=first.CURRENT
    if 'mode' not in cfg:return BASE_FL(engine,theta,plan,queue,t,fl)
    base=engine.joint_base;k=21+fl;lr=base.core.lr_at(k)
    pools=base.clip.stream.retained_pools(plan,queue,t);total=sum(map(len,pools))
    result=np.zeros((10,len(theta)-1));records=[]
    for user,pool in enumerate(pools):
        uniform=np.random.default_rng(730000+k*10+user).choice(pool,min(16,len(pool)),replace=False).astype(int).tolist()
        priority=first.quality.select_batch(engine,theta,pool,k,user)
        gu,_=engine.gradient(theta,uniform,k);gh,_=engine.gradient(theta,priority,k)
        un=float(gu.norm());hn=float(gh.norm())
        if cfg['mode']=='norm':
            normalized=gh*(un/max(hn,1e-30));g=(1-cfg['alpha'])*gu+cfg['alpha']*normalized
            assert float(g.norm())<=un*(1+1e-5)+1e-8
        else:
            delta=gh-gu;delta=delta-(delta@gu)/max(float(gu@gu),1e-30)*gu
            norm=float(delta.norm());beta=min(1.,cfg['rho']*un/max(norm,1e-30));g=gu+beta*delta
            assert abs(float((g-gu)@gu))<=1e-5*max(un*float((g-gu).norm()),1e-20)
        first.quality.infra.lib.infra.charge(engine,uniform,12*len(theta)+16)
        result[user]=(-lr*len(pool)/total*g[:-1]).cpu().numpy()
        records.append(dict(user=user,batch=priority,round=k,uniform_norm=un,priority_norm=hn,applied_norm=float(g.norm())))
    engine.joint_fl_records=records
    return result


def install():
    first.OUT=OUT;first.register=register;first.VARIANTS=VARIANTS;first.fl_vectors=fl_vectors
    # Keep references accessible to the original live-namespace seed wrapper.
    first.__dict__.update(first=first,BASE_FL=BASE_FL)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','screen']);p.add_argument('--dataset',choices=first.audit.DATASETS)
    a=p.parse_args();install()
    if a.action=='register':register()
    else:
        for ds in [a.dataset] if a.dataset else first.audit.DATASETS:first.screen(ds)
