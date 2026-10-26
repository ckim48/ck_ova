import sys
from pathlib import Path
import numpy as np
from radio import inputs

root=Path(sys.argv[1]);sys.path.insert(0,str(root))
from wireless import quantize
rng=np.random.default_rng(3402);l=rng.normal(size=(3,513))*.01;u=l*.3;u[1]=0
c={'block_dimension':512,'scale_metadata_bits':32,'symbols_per_coordinate':8}
x=inputs(l,u,c,.01);lb,ub,decoded,delta,payload,rho,dims,symbols,depths=x
assert payload[1].sum()==0 and delta[1].sum()==0 and 1 not in depths
assert np.linalg.norm(delta.sum(axis=0))<=.0025
for user,bits in depths.items():
    q,d,p=quantize(ub[user:user+1],bits,32)
    np.testing.assert_array_equal(q[0],decoded[user]);np.testing.assert_array_equal(p[0]+3*(p[0]>0),payload[user])
assert lb.shape==(3,2,512) and np.all(symbols==4096)
print('PASS local budget, exact quantizer/metadata accounting, zero payload and padding')
