"""User-requested digital FL then periodic FU; frozen parent training code stays intact."""
from pathlib import Path
import argparse, fcntl, hashlib, json, os, subprocess, sys, time, traceback
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT_TOOLS = ROOT/'tools/overnight_energy_v2/fourway_cap100_v1'
sys.path.insert(0, str(PARENT_TOOLS))
import prepare_campaign as parent
import runtime
import metrics
import numpy as np

OUT = ROOT/'results/overnight_energy_v2/sequential_digital_cap100_v1'
NAME = 'sequential_digital_periodic'
OPT = dict(parallel=False, crossmodal=True, period=5, digital=False,
           main=True, objective='cliperase_inspired')
LABEL = 'Sequential digital FL/FU (OFDMA)'
PARENT_PID = 197587
PARENT_START = '883304071'

def write(name, obj):
    parent.core.write(OUT/name, obj)

def status(**obj):
    write('status.json', dict(unix=time.time(), pid=os.getpid(), **obj))

def prepare():
    plan, requests = parent.prepare()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_file = OUT/'protocol.json'
    if protocol_file.exists():
        protocol = json.loads(protocol_file.read_text())
    else:
        hashes = dict(json.loads((parent.OUT/'protocol.json').read_text())['source_hashes'])
        hashes[str((parent.OUT/'protocol.json').relative_to(ROOT))] = parent.core.sha(parent.OUT/'protocol.json')
        for p in HERE.glob('*.py'):
            hashes[str(p.relative_to(ROOT))] = parent.core.sha(p)
        protocol = dict(version='sequential_digital_cap100_v1', registered_unix=time.time(),
            source_hashes=hashes, arms={NAME:OPT}, labels={NAME:LABEL},
            authorization='User requested replacing the headline parallel digital OFDMA comparator by separate digital FL and occasional FU, keeping the FL round count equal.',
            policy='Same causal physical-time request schedule as the other arms. Batch FU on the existing 5 normalized-frame grid (0.1024s), NOT every 5 FL rounds. FU transactions pause FL uploads; all 200 FL rounds and 45 five-step FU request procedures must finish.',
            scheduling_scope='Periodic FU is the chosen scheduling policy, not a fundamental OFDMA restriction. Different UAVs still transmit simultaneously on disjoint subcarriers within either a FL-only or FU-only transaction.',
            change_from_parallel='Only temporal separation/batching changes. Keep the prior digital baseline CLIPErase-inspired FU objective, Q6 codec and raw-update distortion budget. Do not degrade the FU objective to favor the proposed method.',
            radio='Use the existing integer-subcarrier digital radio for pure FL or pure FU. Never call the analog transport for this arm; no joint FL+FU packet.',
            common_settings=json.loads((parent.OUT/'protocol.json').read_text())['common_settings'],
            common_prefix='Same shared 20-round checkpoint and charged prefix as the existing comparison; this is an adaptation comparison from a common initial state, not separate digital pretraining.',
            network_tx_cap_per_transaction_j=0.10, total_fl_rounds=200, request_count=45,
            FU_steps_per_request=5, period_frames=5, period_seconds=5*parent.clip.stream.FRAME_S,
            main_metrics=['retained validation mR/loss', 'convergence curves and fixed target',
                          'cumulative TX energy upper bound', 'pure occupied uplink and service time'],
            time_scope='Pure uplink excludes local computation, decoding, downlink and control. Service time additionally includes scheduling waits.',
            quality_scope='Keep KL, worst-request KL, prediction disagreement, request latencies and MIA. Procedure completion is not a deletion-quality certificate.',
            comparison='Reuse unchanged completed proposed/separate-AirComp/AirComp+OFDMA arms from fourway_cap100_v1. Preserve its parallel digital arm as an additional control, not relabeled sequential.',
            evaluation='Reference/test data do not choose the policy, objective, codec or settings. Same reporting-only reference and fixed target as the parent.',
            execution='One new 200-round digital arm after the original fourway queue; same global GPU lock; no historical jobs resumed.',
            limitations='Single seed; ideal coding/CSI; existing common-prefix convention; the complete-system comparison contains objective differences between the analog serial baselines and proposed/digital arms.')
        write('protocol.json', protocol)
        write('requests.json', requests)
    assert protocol['arms'] == {NAME:OPT}
    assert json.loads((OUT/'requests.json').read_text()) == requests
    assert parent.common.base.CAP == protocol['network_tx_cap_per_transaction_j'] == .10
    for name, digest in protocol['source_hashes'].items():
        assert parent.core.sha(ROOT/name) == digest, name
    return plan, requests

class SequentialRadio:
    @staticmethod
    def analog(l, u, channels, config, seed):
        # The inherited runtime calls an interface named analog. The transport
        # inside this adapter is exclusively the frozen integer-subcarrier OFDMA.
        assert not (np.any(l) and np.any(u)), 'Sequential baseline received a joint packet'
        phase = 'FU' if np.any(u) else 'FL'
        vector, meta = parent.digital.analog(l, u, channels, config, seed)
        meta.update(physical_mode='digital_sequential_ofdma', phase=phase,
                    packet='Q6('+phase+' only); one payload per transmitting UAV',
                    FL_FU_temporally_overlapped=False)
        return vector, meta

def preflight():
    plan, requests = prepare()
    z = np.load(parent.clip.OUT/'fixed_training_updates.npz')
    l, u, channels = z['FL'], z['FU'], z['channels']
    zero = np.zeros_like(l)
    records = {}
    for phase, ll, uu in [('FL', l, zero), ('FU', zero, u)]:
        vec, meta = SequentialRadio.analog(ll, uu, channels, plan['config'], 500000)
        assert vec is not None and meta['status'] == 'feasible'
        assert meta['access'] == 'integer_subcarrier_OFDMA'
        assert meta['phase'] == phase and not meta['FL_FU_temporally_overlapped']
        assert meta['conditional_rmse'] <= meta['epsilon']*(1+2e-5)+1e-9
        assert meta['energy_bound_j'] <= .10*(1+2e-6)
        assert meta['max_power_w'] <= plan['config']['power_limit_w']*(1+2e-6)
        assert meta['max_user_energy_j'] <= plan['config']['available_energy_j']*(1+2e-6)
        assert meta['uplink_s'] <= plan['config']['round_uplink_deadline_s']+1e-12
        assert np.all(np.asarray(meta['subcarrier_counts']).sum(0) <= 128)
        assert np.isfinite(vec).all()
        records[phase] = meta
    queue = parent.clip.stream.Queue(requests, False, OPT['period'])
    t = 0.; fl = 0; kinds = []
    for _ in range(2000):
        kind, ids = queue.decision(t, fl)
        if kind == 'done': break
        if kind == 'wait': t = ids; continue
        assert kind in ['fl', 'fu']
        assert (kind == 'fu') == bool(ids)
        duration = records[kind.upper()]['uplink_s']
        end = t+duration
        if kind == 'fl': fl += 1
        queue.commit(ids, t, end)
        t = end; kinds.append(kind)
    else: raise AssertionError('Scheduler did not finish')
    assert fl == 180 and queue.metrics(t)['procedure_completed'] == 45
    assert all(r['steps'] == 5 and r['first_service_s'] >= r['arrival_s']-1e-10
               for r in queue.items.values())
    try:
        SequentialRadio.analog(l, u, channels, plan['config'], 0)
    except AssertionError:
        joint_rejected = True
    else:
        joint_rejected = False
    assert joint_rejected
    write('preflight.json', dict(passed=True, unix=time.time(), dimension=l.shape[1],
          records=records, joint_packet_rejected=joint_rejected,
          scheduler=dict(total_fl_rounds=20+fl, procedures=45, five_steps_each=True,
                         pure_FL_transactions=kinds.count('fl'), pure_FU_transactions=kinds.count('fu'),
                         no_joint_transactions=True, timing_only_fixed_update_replay=True)))
    write('queue_registered.json', dict(unix=time.time(), method=NAME, status='registered_after_preflight',
          after=dict(pid=PARENT_PID,start_ticks=PARENT_START,cmd_contains='fourway_cap100_v1/run_campaign.py'),
          GPU_lock='results/overnight_energy_v2/association_objective_gpu.lock'))
    print('PREFLIGHT_PASSED', flush=True)

def parent_alive():
    p = Path('/proc')/str(PARENT_PID)
    if not p.exists(): return False
    try:
        start = p.joinpath('stat').read_text().split()[21]
        cmd = p.joinpath('cmdline').read_bytes().replace(b'\0', b' ').decode()
    except FileNotFoundError:
        return False
    return start == PARENT_START and 'fourway_cap100_v1/run_campaign.py' in cmd

def main():
    plan, requests = prepare()
    own = (OUT/'launch.lock').open('a')
    fcntl.flock(own, fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'started.json').exists(), 'Duplicate sequential run'
    assert json.loads((OUT/'preflight.json').read_text())['passed']
    status(stage='waiting_for_parent_queue', method=NAME)
    while parent_alive(): time.sleep(10)
    assert (parent.OUT/'finished.json').exists(), 'Parent exited without finished marker'
    gpu = (ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a')
    status(stage='waiting_for_shared_gpu_lock', method=NAME)
    fcntl.flock(gpu, fcntl.LOCK_EX)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(), 'GPU occupied outside shared lock'
    prepare()
    write('started.json', dict(unix=time.time(),identity=parent.clip.stream.base.identity(os.getpid())))
    runtime.OUT = OUT
    runtime.write = write
    metrics.OUT = OUT; metrics.ARMS = {NAME:OPT}; metrics.LABELS = {NAME:LABEL}; metrics.write = write
    runtime.vectors = parent.clip.vectors
    runtime.radio = SequentialRadio
    status(stage='loading_model',method=NAME)
    from train import Engine
    from dataset import RetrievalData
    engine = Engine(RetrievalData(), dict(plan))
    channels = np.load(parent.core.SOURCE/'channels.npy')
    proxy_theta = parent.core.load_theta(ROOT/'results/overnight_energy_v2/proxy_objective_v2/proxies/analog/round_020.pt', engine.device)
    prefix = json.loads((parent.OUT/'common_prefix_cost.json').read_text()); write('common_prefix_cost.json', prefix)
    ref_theta = parent.core.load_theta(parent.core.SOURCE/'retrain/round_200.pt', engine.device)
    ref_eval, ref_probs = engine.evaluate(ref_theta, None)
    ref_mia, _ = engine.mia_scores(ref_theta)
    write('reference_metrics.json', dict(metrics=ref_eval,retained_mR_threshold=parent.THRESHOLD,reporting_only=True))
    write('reference_mia.json', ref_mia)
    start = time.time()
    write('queue_status.json', dict(current=NAME, finished=[], pending=[],failures=[]))
    try:
        runtime.run_arm(engine, plan, requests, proxy_theta, ref_probs, parent.THRESHOLD, channels, NAME, OPT, prefix)
    except Exception as e:
        parent.core.write(OUT/NAME/'failure.json',dict(unix=time.time(),error=repr(e)))
        raise
    finally:
        parent.core.write(OUT/NAME/'runtime_seconds.json',dict(started_unix=start,finished_unix=time.time(),elapsed_server_wall_seconds=time.time()-start,scope='Server runtime, not UAV end-to-end latency.'))
        metrics.report()
    prepare()
    write('queue_status.json', dict(current=None,finished=[NAME],pending=[],failures=[]))
    write('finished.json',dict(unix=time.time(),status=json.loads((OUT/NAME/'manifest.json').read_text())['status']))
    status(stage='completed',method=NAME)

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    try:
        preflight() if args.preflight else main()
    except Exception as e:
        status(stage='failed',error=repr(e));traceback.print_exc();raise
    finally:
        parent.common.base.close()
