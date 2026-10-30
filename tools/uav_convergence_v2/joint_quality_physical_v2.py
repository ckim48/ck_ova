"""Physical runner correction; preserve first factory rejection before training."""
import argparse
from pathlib import Path
import time
import numpy as np
import joint_quality_control_v1 as first

ROOT,read,write,sha=first.ROOT,first.read,first.write,first.sha
OUT=ROOT/'results/joint_quality_physical_v2_20261030'


def seeded_vectors(engine,theta,plan,queue,t,fl,kind,ids,proxy,teachers,crossmodal):
    engine.joint_base=JOINT_BASE
    engine.joint_rng_fl=lambda k:np.random.default_rng(k)
    engine.joint_rng_fu=lambda k:np.random.default_rng(k)
    return CALL_VECTORS(engine,theta,plan,queue,t,fl,kind,ids,proxy,teachers,crossmodal)


def run(stage,variant):
    context=None
    if stage=='context':
        import joint_quality_context_v2 as context
        context.install()
    if stage=='focused':
        import joint_quality_focused_v3 as focused
        focused.install()
    first.register();first.quality.register();first.set_variant(variant)
    cfg=first.VARIANTS[variant];dest=OUT/stage/variant
    assert not (dest/'protocol.json').exists(),'Preserve every physical attempt.'
    protocol=dict(registered_unix=time.time(),stage=stage,variant=variant,configuration=cfg,
        selection='Shortlisted after exploratory exact screens; independent physical rerun, unchanged comparison methods. Seed0 only.',
        correction='v1 rejected while constructing an unused proxy-objective factory, before any update. Retain original factory for unused objectives and supply two live RNG adapters expected by the original seed-offset wrapper. No training rule changes.',
        transport='Original parallel AirComp, all-head optimizer, FU-relative distortion guard,0.1J network RF cap and0.32W device cap. Charge all executed new gradient/teacher work using original accounting.',
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),str(Path(first.__file__).relative_to(ROOT)):sha(Path(first.__file__))},
        screen_protocol=read(first.OUT/'protocol.json'))
    write(dest/'protocol.json',protocol)
    q=first.quality
    base_vectors=context.BASE_VECTORS if context else first.vectors
    # The live namespace preserves the registered CURRENT binding and patched FL function.
    fl_fn=q.infra.cached_clone(first.fl_vectors,[('np.random.default_rng(','engine.joint_rng_fl(',1)],{},live_globals=first.__dict__)
    vec_fn=q.infra.cached_clone(base_vectors,[('np.random.default_rng(','engine.joint_rng_fu(',1)],{},live_globals=first.__dict__)
    first.fl_vectors=fl_fn
    if context:
        context.BASE_VECTORS=vec_fn;first.vectors=context.vectors;call_vectors=context.vectors
    else:first.vectors=vec_fn;call_vectors=vec_fn
    def factory(base,objective):
        if objective!='cliperase':return base.control.synchronized_vectors(objective)
        return q.infra.cached_clone(seeded_vectors,[],dict(JOINT_BASE=base,CALL_VECTORS=call_vectors))
    q.OUT=OUT/stage;q.register=lambda:protocol
    q.VARIANTS={variant:dict(label=variant,objective='gaft',selection=cfg['batch'],equal=False,exact=False)}
    q.vectors=factory;q.cases=lambda:[dict(id='baseline',axis=None,value=None)]
    q.run(variant,'baseline')
    write(dest/'finished.json',dict(passed=True,variant=variant,stage=stage,finished_unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['first','context','focused'],default='first');p.add_argument('--variant',required=True)
    a=p.parse_args();run(a.stage,a.variant)
