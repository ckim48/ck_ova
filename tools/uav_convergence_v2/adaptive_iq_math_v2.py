"""Preserve the registered RMS rule; handle the zero-LR cosine endpoint."""
import numpy as np
import adaptive_iq_math_v1 as original

SETTINGS = original.SETTINGS


class RMS(original.RMS):
    def apply(self, left, right, variance, lr):
        if lr == 0:
            raw = np.asarray(left)+np.asarray(right)
            assert not np.any(raw) and not np.any(variance)
            return raw, dict(raw_norm=0., applied_norm=0., branches=0,
                stored_state_bytes=self.state_bytes(), MAC_equivalents=0,
                correction_norm=0., zero_lr_endpoint=True)
        return super().apply(left, right, variance, lr)


def checks():
    result = original.checks()
    for config in SETTINGS.values():
        optimizer = RMS(config, 'iq_separate'); optimizer.prepare([])
        y, info = optimizer.apply(np.zeros(256), np.zeros(256), np.zeros((2, 256)), 0.)
        assert not np.any(y) and not optimizer.states and info['MAC_equivalents'] == 0
    result['checks'].append('zero-LR endpoint is an exact no-op')
    return result
