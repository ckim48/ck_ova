"""Rerun explicitly shortlisted learning controls with the original AirComp solver."""
import argparse
from pathlib import Path
import time
import joint_quality_control_v1 as first

ROOT,read,write,sha=first.ROOT,first.read,first.write,first.sha
OUT=ROOT/'results/joint_quality_physical_v1_20261030'


def run(stage,variant):
    if stage=='context':
        import joint_quality_context_v2 as context
        context.install()
    if stage=='focused':
        import joint_quality_focused_v3 as focused
        focused.install()
    first.register();first.quality.register()
    first.set_variant(variant)
    cfg=first.VARIANTS[variant]
    dest=OUT/stage/variant
    assert not (dest/'protocol.json').exists(),'Preserve every physical attempt.'
    protocol=dict(registered_unix=time.time(),stage=stage,variant=variant,configuration=cfg,
        selection='Selected after exploratory exact screens; rerun full physical training, do not rescale old metrics.',
        transport='Original simultaneous FL/FU AirComp, all-head joint resource search, FU-relative distortion guard, original0.1J network RF cap and0.32W device cap. No forced comparison rank or changed baseline.',
        compute='All actual gradient/teacher passes and explicit extra arithmetic are charged through original operation counters. Same original accounting conventions and exclusions.',
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
                       str(Path(first.__file__).relative_to(ROOT)):sha(Path(first.__file__))},
        screen_protocol=first.read(first.OUT/'protocol.json'))
    write(dest/'protocol.json',protocol)
    q=first.quality
    def factory(base,objective):
        assert objective=='cliperase'
        def fn(engine,theta,plan,queue,t,fl,kind,ids,proxy,teachers,crossmodal):
            engine.joint_base=base
            return first.vectors(engine,theta,plan,queue,t,fl,kind,ids,proxy,teachers,crossmodal)
        return fn
    q.OUT=OUT/stage;q.register=lambda:protocol
    q.VARIANTS={variant:dict(label=variant,objective='gaft',selection=cfg['batch'],equal=False,exact=False)}
    q.vectors=factory;q.cases=lambda:[dict(id='baseline',axis=None,value=None)]
    q.run(variant,'baseline')
    write(dest/'finished.json',dict(passed=True,variant=variant,stage=stage,finished_unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['first','context','focused'],default='first');p.add_argument('--variant',required=True)
    a=p.parse_args();run(a.stage,a.variant)
