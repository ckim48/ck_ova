"""Bind the registered selection configuration when constructing each arm."""
from pathlib import Path
import time
import learning_selection_fix as previous
study=previous.study
OUT=study.ROOT/'results/learning_selection_fix_v2_20261028'


def vectors(base,objective):
 previous.select_with_rng.__globals__['CONFIG']=study.CONFIG
 return previous.vectors(base,objective)


def register():
 path=OUT/'protocol.json'
 if path.exists():
  result=study.read(path)
  for name,digest in result['source_sha256'].items():assert study.sha(study.ROOT/name)==digest,name
  return result
 original=previous.register();sources=dict(original['source_sha256'])
 sources[str(Path(__file__).relative_to(study.ROOT))]=study.sha(Path(__file__))
 result=dict(original,registered_unix=time.time(),source_sha256=sources,
  correction_v2='Refresh CONFIG in the copied selection function namespace at arm construction. Prior six v1 correction attempts stopped in the first selection before any transaction; all are preserved. Selection rule, seed and workload unchanged.')
 study.write(path,result);return result


def install():
 study.OUT=OUT;study.register=register;study.vectors=vectors
 return study
