"""Count executed adapter work by data owner without changing model arithmetic."""
from collections import Counter
import math
import torch
from energy import operation_macs


def instrumented_engine(parent):
    class Engine(parent):
        def __init__(self, data, plan, payload):
            super().__init__(data, plan, payload)
            self.sample_owner = {int(i): u for u, ids in enumerate(plan['partitions']) for i in ids}
            self.device_counts = [Counter() for _ in range(10)]
            self.active_owners = []
            pixels, texts, apply = self.data.pixels, self.data.texts, self.model.apply

            def remember(ids):
                if self.work_started is not None:
                    self.active_owners = [self.sample_owner[int(i)] for i in ids]

            def counted_pixels(ids, device):
                remember(ids)
                return pixels(ids, device)

            def counted_texts(ids, device):
                remember([data.owners[int(i)] for i in ids])
                return texts(ids, device)

            def counted_apply(z, weights, modality):
                if self.work_started is not None:
                    assert len(self.active_owners) == len(z)
                    grad = torch.is_grad_enabled() and weights.requires_grad
                    key = modality + ('_grad_samples' if grad else '_no_grad_samples')
                    for user, count in Counter(self.active_owners).items():
                        self.device_counts[user][key] += count
                return apply(z, weights, modality)

            self.data.pixels, self.data.texts = counted_pixels, counted_texts
            self.model.apply = counted_apply

        def start_work(self):
            assert self.work_started is None
            self.device_before = [dict(c) for c in self.device_counts]
            super().start_work()

        def finish_work(self):
            result = super().finish_work()
            counts = [{k: v - old.get(k, 0) for k, v in current.items()}
                      for current, old in zip(self.device_counts, self.device_before)]
            macs = [operation_macs(c) for c in counts]
            assert math.isclose(sum(macs), operation_macs(result['adapter_operations']), abs_tol=1e-6)
            result.update(device_adapter_operations=counts, device_macs=macs,
                          device_MAC_scope='Executed adapter matrix products; gradient includes two backward products. Scenario energy model is stored in protocol.json.')
            self.work_started = None
            return result
    return Engine
