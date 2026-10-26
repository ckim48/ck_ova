"""Small, independently testable published-method operations.

FedMemPrune follows paper Algorithm 1; an adapter is a token-wise version of
the official FedWiper 1x1-convolution module. Neither is an unlearning guarantee.
"""
import numpy as np


def reset_low_gradient(theta, gradient, fresh, ratio):
    theta, gradient, fresh = map(np.asarray, (theta, gradient, fresh))
    assert theta.shape == gradient.shape == fresh.shape
    assert 0 <= ratio <= 1 and np.isfinite(gradient).all()
    selected = np.argsort(np.abs(gradient), kind='stable')[:int(theta.size*ratio)]
    result=theta.copy();result[selected]=fresh[selected]
    return result, selected


def minimum_norm_two(g1,g2):
    """Solve min_{0<=a<=1} ||a*g1+(1-a)*g2||^2, FedU Eq.(9)-(10).

    This DOES NOT define the ambiguous IAF loss gradient in Eq.(7).
    """
    g1,g2=map(np.asarray,(g1,g2));difference=g1-g2
    denominator=float(difference@difference)
    a=0.5 if denominator==0 else float(np.clip(-(g2@difference)/denominator,0,1))
    return -(a*g1+(1-a)*g2),a


def neumann_inverse_product(hessian,g,steps):
    """Literal deterministic version of FedU's HVP recurrence, no damping."""
    v=np.asarray(g).copy()
    for _ in range(steps):v=g+v-hessian@v
    return v


def disjoint_shards(partitions,number=4,seed=0):
    result=[]
    for user,ids in enumerate(partitions):
        ordered=np.random.default_rng(seed+user).permutation(ids)
        result.append([a.astype(int).tolist() for a in np.array_split(ordered,number)])
    return result


def self_check():
    g=np.array([100.,-0.01,3.,-0.02,2.])
    old=np.arange(5.)+10;fresh=np.arange(5.)-10
    changed,selected=reset_low_gradient(old,g,fresh,.4)
    assert selected.tolist()==[1,3]
    assert np.array_equal(changed[[0,2,4]],old[[0,2,4]])
    assert np.array_equal(changed[selected],fresh[selected])
    assert np.array_equal(reset_low_gradient(old,g,fresh,0)[0],old)
    assert np.array_equal(reset_low_gradient(old,g,fresh,1)[0],fresh)
    # HVP recurrence converges only under its spectral condition.
    h=np.diag([.3,.8,1.2]);v=np.array([2.,1.,-3.])
    approx=neumann_inverse_product(h,v,100)
    assert np.linalg.norm(approx-np.linalg.solve(h,v))<1e-10
    assert np.linalg.norm(neumann_inverse_product(np.eye(3)*3,v,10))>100*np.linalg.norm(v)
    rng=np.random.default_rng(8)
    for _ in range(20):
        g1=rng.normal(size=8);g2=rng.normal(size=8)
        direction,a=minimum_norm_two(g1,g2)
        grid=np.linspace(0,1,10001)
        loss=np.sum((grid[:,None]*g1+(1-grid[:,None])*g2)**2,axis=1)
        assert direction@direction<=float(loss.min())+1e-9
        assert g1@direction<=-(direction@direction)+1e-9
        assert g2@direction<=-(direction@direction)+1e-9
    shards=disjoint_shards([list(range(40)),list(range(40,80))])
    for original,ss in zip([list(range(40)),list(range(40,80))],shards):
        assert sorted(sum(ss,[]))==original
        assert all(not(set(ss[i])&set(ss[j])) for i in range(4) for j in range(i))
    return dict(passed=True,memprune_absolute_importance=True,unselected_preserved=True,
        original_initialization_replacement=True,neumann_matches_spd_solution=True,
        neumann_unstable_hessian_detected=True,mgda_matches_scalar_search=True,
        shard_isolation=True,fedu_loss_definition_verified=False)


if __name__=='__main__':
    import json
    print(json.dumps(self_check()))
