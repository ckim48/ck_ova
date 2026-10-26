"""Merge all fair FU61 radio cases before interpreting paid compression."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

root = Path('/home/wnlab/Documents/Codex/2026-10-20/files-pasted-by-the-user-standalone/outputs/ova_CK')
family = root/'results/overnight_energy_v2'; out = family/'fu_radio_summary_v23'; out.mkdir(exist_ok=True)
assert not (out/'results.json').exists()
paths = [family/'public_history_fu_v19/radio/results.json', family/'public_history_fu_v19b/results.json']
rows = sum([json.loads(p.read_text()) for p in paths], []); assert len(rows) == 58
certificate = json.loads((family/'transport_certificate_v20/fu_quality_v22.json').read_text())
assert certificate['checked_saved_candidates'] == sum(r['status'] == 'feasible_radio_only' for r in rows) == 48
assert all(x['status']['status'].startswith('completed') for x in certificate['observed_statuses'])
source = family/'public_history_fu_v19/screen'; plan = json.loads((source/'plan.json').read_text()); c = plan['config']
sys.path[:0] = [str(root), str(root/'tools/overnight_energy_v2')]
from phase_split_grid import make_inputs
with np.load(source/'vectors.npz') as data:
    learning, unlearning = [data[k].copy() for k in ['learning', 'unlearning']]
epsilon = .2*np.linalg.norm(make_inputs(learning, unlearning, c)[5].sum(axis=0))
selected = []; comparisons = []
for relative in [.2, .6]:
    for mode in ['hybrid', 'mixed_aircomp', 'phase_grid']:
        best = {}
        for rank in [0, 1, 3]:
            eligible = [r for r in rows if r['relative_tolerance'] == relative and r['mode'] == mode and r['rank'] == rank and r['status'] == 'feasible_radio_only']
            chosen = min(eligible, key=lambda x: x['energy_bound_j']); best[rank] = chosen
            selected.append({k: v for k, v in chosen.items() if k != 'heads'})
        for rank in [1, 3]:
            saving = best[0]['energy_bound_j']-best[rank]['energy_bound_j']
            # Two dense projection/reconstruction multiplies, each 2*d*rank FLOPs,
            # on each remote transmitter. Basis construction and head decoding
            # are deliberately excluded from this count, so it is only a floor.
            flop_floor = 4*learning.shape[1]*rank*(len(learning)-1)
            comparisons.append(dict(relative_tolerance=relative, mode=mode, rank=rank,
                baseline_j=best[0]['energy_bound_j'], candidate_j=best[rank]['energy_bound_j'],
                modeled_rf_upper_saving_j=saving, modeled_rf_upper_saving_fraction=saving/best[0]['energy_bound_j'],
                additional_total_network_cost_must_be_below_j=max(0., saving),
                transmitter_projection_flop_floor=flop_floor,
                break_even_j_per_counted_flop_if_positive=saving/flop_floor if saving > 0 else None,
                same_absolute_epsilon=best[0]['epsilon'] == best[rank]['epsilon'],
                selected_phase_fraction=best[rank]['analog_fraction']))
preview = []
for bits in [2, 4, 6, 8]:
    delta = make_inputs(learning, unlearning, dict(c, quantization_bits=bits))[3]
    floors = [float(np.linalg.norm(np.delete(delta, head, axis=0).sum(axis=0))) for head in range(len(learning))]
    preview.append(dict(bits=bits, epsilon=epsilon, per_head_quantization_bound_floor=floors,
                        conservative_bound_viable_heads=[i for i, v in enumerate(floors) if v < epsilon]))
result = dict(status='completed_all58_fair_radio_cases', at_kst=datetime.now().isoformat(timespec='seconds'),
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    input_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+[source/'vectors.npz', family/'transport_certificate_v20/fu_quality_v22.json']},
    selected=selected, compression_comparisons=comparisons, quantization_preview=preview,
    limits=['All54main and4minimum-dimension-phase-boundary cases included. Same actualCPUFU61 vectors, absolute tolerance/time/power/battery;48savedbestdesigns independently checked, not global optima.',
        'Rank3 savings are conditional modeled RF upper-bound changes; additional basis/projection/receiver/codec/header energy unmeasured. FLOP count excludes basis construction and receiver work, hence no total-energy claim.',
        'Original0.2:Hybrid rank3 saves about1%, Mixed about5%, phase rank3 costs slightly more. Loose0.6:all compressed candidates cost more. No universal compression or NOMA advantage.',
        'Some6bit/4bit quantizers have room in the original0.2 conservative error envelope.2bit floor exceeds this sufficient bound for all heads; this does not prove original physical/task infeasibility.',
        'No task-output/test-based selection, new inference, GPU work or automatic full-training promotion. V22 common-noise task pilot continues separately.'])
(out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k: result[k] for k in ['status', 'compression_comparisons', 'quantization_preview']}))
