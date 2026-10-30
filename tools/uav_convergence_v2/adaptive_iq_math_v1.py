"""Norm-bounded task-specific RMS adaptation of recovered AirComp updates.

This is an operational adaptation inspired by adaptive OTA optimization, not
an implementation or theorem reproduction of Wang et al. (IEEE TSP2025).
"""
import numpy as np

SETTINGS = {
    'block_slow': dict(group=128, beta=.99, blend=1.),
    'block_blend': dict(group=128, beta=.9, blend=.5),
    'diagonal_slow': dict(group=1, beta=.99, blend=1.),
    'diagonal_blend': dict(group=1, beta=.9, blend=.5),
}


class RMS:
    def __init__(self, config, mode):
        self.config = config
        self.mode = mode
        self.states = {}
        self.context = None
        self.head = None

    def prepare(self, requests):
        key = tuple(sorted(requests))
        changed = self.context is not None and self.context != key
        if changed:
            if self.mode == 'combined_reset':
                self.states.clear()
            elif self.mode == 'iq_separate':
                self.states.pop('U', None)
        self.context = key

    def state_bytes(self):
        return sum(v['v'].nbytes+8 for v in self.states.values())

    def branch(self, x, variance, name, lr):
        x = np.asarray(x, dtype=float)
        group, beta, blend = [self.config[k] for k in ['group', 'beta', 'blend']]
        assert len(x) % group == 0 and lr > 0
        q = x/lr
        noise = np.asarray(variance)/lr**2
        value = np.maximum((q.reshape(-1, group)**2).mean(1)-noise.reshape(-1, group).mean(1), 0.)
        prev = self.states.get(name, dict(v=np.zeros_like(value, dtype=np.float32), count=0))
        count = prev['count']+1
        # The stored state really uses float32; its handover cost is not hypothetical.
        v = (beta*prev['v']+(1-beta)*value).astype(np.float32)
        self.states[name] = dict(v=v, count=count)
        second = v.astype(float)/(1-beta**count)
        floor = max(.01*float(second.mean()), 1e-30)
        denom = np.sqrt(second+noise.reshape(-1, group).mean(1)+floor)
        preconditioned = q/np.repeat(denom, group)
        radius = np.linalg.norm(x)
        norm = np.linalg.norm(preconditioned)
        if norm > 0:
            preconditioned *= radius/norm
        candidate = (1-blend)*x+blend*preconditioned
        norm = np.linalg.norm(candidate)
        if norm > 0:
            candidate *= radius/norm
        return candidate

    def apply(self, left, right, variance, lr):
        left = np.asarray(left, dtype=float)
        right = np.asarray(right, dtype=float)
        var = np.asarray(variance, dtype=float)
        assert var.shape == (2, left.size) and np.all(var >= 0)
        raw = left+right
        if self.mode == 'iq_separate':
            result = self.branch(left, var[0], 'L', lr)
            branches = 1
            if self.context:
                result += self.branch(right, var[1], 'U', lr)
                branches += 1
        else:
            result = self.branch(raw, var.sum(0), 'S', lr)
            branches = 1
        # Identical total step-norm budget for shared and separate adaptation.
        radius = float(np.linalg.norm(raw))
        norm = float(np.linalg.norm(result))
        if norm > 0:
            result *= radius/norm
        assert np.isfinite(result).all()
        return result, dict(raw_norm=radius, applied_norm=float(np.linalg.norm(result)),
            branches=branches, stored_state_bytes=self.state_bytes(),
            MAC_equivalents=int(64*left.size*branches+16*left.size),
            correction_norm=float(np.linalg.norm(result-raw)))


def checks():
    rng = np.random.default_rng(312)
    for config in SETTINGS.values():
        a, b = RMS(config, 'combined'), RMS(config, 'iq_separate')
        for _ in range(20):
            x = rng.normal(size=256)
            a.prepare([]); b.prepare([])
            ya, _ = a.apply(x, x*0, np.zeros((2, 256)), .0035)
            yb, _ = b.apply(x, x*0, np.zeros((2, 256)), .0035)
            assert np.array_equal(ya, yb), 'One-task policies must match.'
            assert np.isclose(np.linalg.norm(ya), np.linalg.norm(x))
        b.prepare([1]); b.apply(x, -x*.5, np.zeros((2, 256)), .0035)
        assert 'U' in b.states
        l = b.states['L']['v'].copy()
        b.prepare([2])
        assert 'U' not in b.states and np.array_equal(l, b.states['L']['v'])
        assert b.state_bytes() == l.nbytes+8
    return dict(passed=True, checks=['FL-only equality', 'step-norm preservation',
        'FU cohort reset retains FL state', 'serialized state bytes'])
