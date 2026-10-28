"""Narrow correction: keep the vector factory's two seeded RNG call sites.

The v3 selection attempts failed before the first transaction because a later
factory requires both call sites. Preserve those attempts; no results overwritten.
"""
from pathlib import Path
import time
import learning_quality_v3 as study

OUT=study.ROOT/'results/learning_selection_fix_v1_20261028'
ORIGINAL_REGISTER=study.register
ORIGINAL_VECTORS=study.vectors
select_with_rng=study.infra.cached_clone(study.select_batch,[
 ('def select_batch(engine,theta,pool,k,user):','def select_batch(engine,theta,pool,k,user,rng):',1),
 ('rng=np.random.default_rng(730000+k*10+user);','',1)],{})


def vectors(base,objective):
 fn=ORIGINAL_VECTORS(base,objective)
 if study.CONFIG['selection']:
  fn=study.infra.cached_clone(fn,[
   ('batch=select_batch(engine,theta,pool,k,user)',
    'batch=select_batch(engine,theta,pool,k,user,np.random.default_rng(730000+k*10+user))',1)],
   dict(select_batch=select_with_rng))
 return fn


def register():
 path=OUT/'protocol.json'
 if path.exists():
  result=study.read(path)
  for name,digest in result['source_sha256'].items():assert study.sha(study.ROOT/name)==digest,name
  return result
 previous=study.read(study.ROOT/'results/learning_quality_v3_20261028/protocol.json')
 sources=dict(previous['source_sha256']);sources[str(Path(__file__).relative_to(study.ROOT))]=study.sha(Path(__file__))
 result=dict(previous,registered_unix=time.time(),source_sha256=sources,
  correction='Preserve two explicit RNG call sites for the inherited seed-offset factory. Pass the same RNG to the identical selection rule. Original six failed v3 selection attempts stopped before any transaction and remain intact.')
 study.write(path,result);return result


def install():
 study.OUT=OUT;study.register=register;study.vectors=vectors
 return study
