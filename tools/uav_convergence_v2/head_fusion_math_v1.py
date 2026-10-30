"""Two-task PCGrad and a predeclared noise-regularized adaptation.

Inputs are recovered estimates, never evaluation references or clean gradients.
The latter are used by the runner only for diagnostic error measurements.
"""
import numpy as np


def fuse(l, u, variance=None, noise_regularized=False):
    l, u = np.asarray(l, dtype=float), np.asarray(u, dtype=float)
    a, b, dot = float(l @ l), float(u @ u), float(l @ u)
    vl = vu = dot_sd = 0.
    if noise_regularized:
        v = np.asarray(variance, dtype=float)
        assert v.ndim == 2 and v.shape[0] == 2 and np.all(v >= 0)
        m = v.shape[1]
        assert l.size % m == 0
        d = l.size // m
        nl = np.maximum((l.reshape(m, d)**2).sum(1)-v[0], 0.)
        nu = np.maximum((u.reshape(m, d)**2).sum(1)-v[1], 0.)
        dot_sd = float(np.sqrt(np.sum(v[1]/d*nl + v[0]/d*nu + v[0]*v[1]/d)))
        vl, vu = v.sum(1)
    q = max(0., -dot-(1.645*dot_sd if noise_regularized else 0.))
    # Both tasks must be present. This also gives exact FL-only identity.
    if a == 0 or b == 0:
        q = 0.
    wl, wu = q/max(a+vl, 1e-30), q/max(b+vu, 1e-30)
    correction = wl*l + wu*u
    return l+u+correction, dict(dot=dot, cosine=dot/max(np.sqrt(a*b), 1e-30),
        conflict=dot < 0, corrected=q > 0, dot_noise_sd=dot_sd,
        L_multiplier=1+wl, U_multiplier=1+wu,
        correction_norm=float(np.linalg.norm(correction)),
        L_norm=float(np.sqrt(a)), U_norm=float(np.sqrt(b)))


def local_fuse(l, u):
    left, right, records = l.copy(), u.copy(), []
    for i in range(len(l)):
        _, meta = fuse(l[i], u[i])
        left[i] *= meta['L_multiplier']
        right[i] *= meta['U_multiplier']
        records.append(meta)
    return left, right, records


def check():
    rng = np.random.default_rng(44)
    for _ in range(100):
        l, u = rng.normal(size=(2, 128))
        fused, meta = fuse(l, u)
        assert fused@l >= -1e-9 and fused@u >= -1e-9
        if l@u >= 0:
            assert np.array_equal(fused, l+u)
        assert np.array_equal(fuse(l, np.zeros_like(l))[0], l)
        assert np.allclose(fuse(l, u, np.zeros((2, 4)), True)[0], fused)
    l = np.array([[1., 0.], [0., 1.]])
    u = np.array([[0., 1.], [-2., 0.]])
    a, b, _ = local_fuse(l, u)
    assert not np.allclose((a+b).sum(0), fuse(l.sum(0), u.sum(0))[0])
    return dict(passed=True, checks=['two-task descent', 'one-task identity',
        'nonconflict identity', 'zero-noise equivalence', 'local/global noncommutativity'])
