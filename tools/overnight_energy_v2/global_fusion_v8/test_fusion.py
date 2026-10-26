import numpy as np
from pilot import fuse_direction

rng=np.random.default_rng(811)
for _ in range(200):
    l=rng.normal(size=30);u=rng.normal(size=30)*100;p=fuse_direction(l,u)
    assert l@p>=-1e-10 and np.linalg.norm(p)<=np.linalg.norm(l)+1e-10
    np.testing.assert_allclose(fuse_direction(l,p),p,atol=1e-10)
l=np.array([1.,0.]);np.testing.assert_array_equal(fuse_direction(l,-l),np.zeros(2))
np.testing.assert_array_equal(fuse_direction(np.zeros(2),l),np.zeros(2))
u=np.array([-1.,1.]);l2=l+u;u2=np.zeros(2)
np.testing.assert_array_equal(l+u,l2+u2)
assert not np.allclose(l+fuse_direction(l,u),l2+fuse_direction(l2,u2))
print('PASS halfspace/norm/idempotence/zero cases; fusion depends on separated components')
