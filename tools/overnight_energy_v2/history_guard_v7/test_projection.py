import numpy as np
from pilot import project_direction

rng=np.random.default_rng(4701)
for _ in range(100):
    g=rng.normal(size=20);d=rng.normal(size=20);p=project_direction(d,g)
    assert g@p<=1e-10 and np.linalg.norm(p)<=np.linalg.norm(d)+1e-10
    if g@d<=0:np.testing.assert_array_equal(p,d)
    else:assert abs(g@p)<1e-10
d=np.arange(10.);np.testing.assert_array_equal(project_direction(d,np.zeros_like(d)),d)
print('PASS retained halfspace, norm contraction, nonconflict and zero-gradient cases')
