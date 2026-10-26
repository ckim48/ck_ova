"""Pre-training compatibility audit, NOT a FedWiper performance result.

Tests the official channel-doubling residual adapter parameterization; reports
an optimistic MAC sum-capacity lower bound, without weakening the baseline by
pretending an oversized payload can only be sent in one old prompt-sized slot.
"""
from pathlib import Path
import sys,json,math
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tools/overnight_energy_v2/full_system_baselines_v1'))
import run_memprune as run
import numpy as np


def main():
    plan=json.loads((run.core.SOURCE/'plan.json').read_text());c=plan['config']
    channels=np.load(run.core.SOURCE/'channels.npy');rows=[]
    for width in [512,768]:
        d=4*width*width+3*width # C->2C->C with both bias vectors.
        for adapters in [1,2]:
            for bits in [4,8,32]:
                choices=[]
                for head in range(10):
                    h=np.delete(channels[:,head],head)
                    # channels are magnitude / sqrt(full-band noise variance).
                    capacity=c['symbol_rate_hz']*np.log2(1+c['power_limit_w']*np.sum(h*h))
                    seconds=9*d*adapters*bits/capacity
                    choices.append((float(seconds),head))
                seconds,head=min(choices)
                rows.append(dict(channel_width=width,adapters=adapters,parameters_per_client=d*adapters,
                    bits=bits,optimistic_minimum_uplink_seconds=seconds,best_head=head,
                    minimum_legacy_slots=math.ceil(seconds/c['round_uplink_deadline_s'])))
    result=dict(status='payload_scope_audit_completed',not_a_learning_experiment=True,
        original_structure='official module/module.py ForgetLayer: C->2C->C, ReLU/residual/ReLU',
        original_source='https://github.com/grey1989/FedWiper/blob/master/module/module.py',
        old_prompt_parameters=20480,old_per_upload_deadline_s=c['round_uplink_deadline_s'],
        rows=rows,ranking_permitted=False,
        unresolved=['CLIP retrieval placement and head adaptation require validation; official release implements CNN classification/detection/segmentation',
          'affected federated shards must reset to pre-data initialization and retrain; no shared target-trained frozen backbone allowed',
          'large payloads require packetization with all airtime charged; exceeding one old slot must not alone be reported as FedWiper algorithm failure',
          'native prepared submodel ensemble needs its own retain-only replay reference'])
    run.core.write(run.OUT/'fedwiper_source_probe/payload_audit.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
