"""Versioned full trajectories with matched data and decomposed delay metrics."""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPTS), str(ROOT), str(ROOT/'tools/overnight_energy_v2'),
               str(ROOT/'tools/overnight_energy_v2/initialization_v11'),
               str(ROOT/'tools/overnight_energy_v2/receiver_control_v1')]


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import torch
    import torch.nn.functional as F
    from clip_model import DeepPromptCLIP, projected_coupling
    from real_data import CIFAR100
    from data_schedule import dirichlet_partition, make_schedule
    from run_clip_federated import local_gradient, evaluate
    from transport import link_matrix
    from trajectory_transport import communicate

    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--method', choices=['proposed', 'aircomp_only'], required=True)
    p.add_argument('--max-rounds', type=int)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text()); c = plan['config']
    for name, expected in plan['source_sha256'].items():
        assert sha(ROOT/name) == expected, f'Source changed: {name}'
    for name, expected in plan['runner_sha256'].items():
        assert sha(SCRIPTS/name) == expected, f'Runner changed: {name}'
    out = args.output or args.plan.parent/args.method
    out.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU substitution')
    torch.set_num_threads(4); torch.manual_seed(c['seed']); torch.cuda.manual_seed_all(c['seed'])
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')
    checkpoint = ROOT/'checkpoints/clip-vit-base-patch16'
    dataset = CIFAR100(ROOT/'data/cifar-100-python', checkpoint)
    training_ids, heldout_ids = dataset.validation_split(c['seed'], c['validation_per_class'])
    order = np.random.default_rng(c['seed']+17001).permutation(heldout_ids)
    validation_ids = order[:c['evaluation_samples']]
    calibration_ids = order[c['evaluation_samples']:c['evaluation_samples']+64]
    assert not set(validation_ids).intersection(calibration_ids)
    partitions = dirichlet_partition(dataset.train_labels[training_ids], c['users'], c['dirichlet_alpha'], c['seed'])
    partitions = [training_ids[x] for x in partitions]
    schedule = json.loads((args.plan.parent/'records.json').read_text())
    assert sha(args.plan.parent/'records.json') == plan['records_sha256']
    assert sha(ROOT/plan['warm_checkpoints'][args.method]) == plan['warm_checkpoint_sha256'][args.method]
    assert sha(ROOT/plan['channel_source']) == plan['channel_source_sha256']
    with np.load(ROOT/plan['channel_source']) as z:
        channels = z['channels'].copy()
    forbidden=set(validation_ids)|set(calibration_ids)
    for rec in schedule:
        assert not forbidden.intersection(set().union(*map(set,rec['retained'])))
        assert not forbidden.intersection(set().union(*map(set,rec['forget'])))
    model = DeepPromptCLIP.from_local_checkpoint(checkpoint, prompt_layers=c['prompt_layers'],
        prompt_tokens=c['prompt_tokens'], checkpoint_layers=c['checkpoint_layers']).to(device).eval()
    text = dataset.text_inputs(device, c['prompt_tokens'])
    state = torch.load(ROOT/plan['warm_checkpoints'][args.method],map_location='cpu',weights_only=True)
    assert state['round'] == 65
    theta = state['theta'].to(device).detach().clone()
    assert theta.numel() == model.initial_vector.numel()
    torch.cuda.synchronize()
    manifest = dict(status='running', method=args.method, config=c, plan_sha256=sha(args.plan),
                    data_schedule_sha256=hashlib.sha256(json.dumps(schedule,sort_keys=True).encode()).hexdigest(),
                    validation_ids_sha256=hashlib.sha256(validation_ids.tobytes()).hexdigest(),
                    calibration_ids_sha256=hashlib.sha256(calibration_ids.tobytes()).hexdigest(),
                    window_scope=plan['scope'], warm_checkpoint_sha256=plan['warm_checkpoint_sha256'][args.method], dimension=len(theta), gpu=torch.cuda.get_device_name(0), timing_scope=plan['timing_scope'])
    write(out/'manifest.json', manifest); write(out/'schedule.json', schedule)
    np.savez_compressed(out/'partition.npz', validation_ids=validation_ids, calibration_ids=calibration_ids,
                        channels=channels, **{f'user_{i}':x for i,x in enumerate(partitions)})
    (out/'checkpoints').mkdir()
    prefix=ROOT/plan['prefix_folders'][args.method]
    assert sha(prefix/'rounds.jsonl')==plan['prefix_metrics_sha256'][args.method]
    prior=[json.loads(x) for x in (prefix/'rounds.jsonl').read_text().splitlines() if x.strip()]
    assert [x['round'] for x in prior]==list(range(61,66))
    initial = evaluate(model,theta,dataset,text,validation_ids,device,c['microbatch_size'])
    assert abs(initial['loss']-prior[-1]['validation']['loss'])<1e-5
    assert initial['accuracy']==prior[-1]['validation']['accuracy']
    write(out/'resume_evaluation.json',initial)
    write(out/'initial_evaluation.json',json.loads((prefix/'initial_evaluation.json').read_text()))
    (out/'rounds.jsonl').write_text((prefix/'rounds.jsonl').read_text())
    torch.save({'theta':theta.cpu(),'round':65},out/'checkpoints/round_065.pt')
    last=prior[-1]
    cum_e=last['cumulative_communication_energy_j'];cum_expected=last['cumulative_expected_communication_energy_j']
    cum_up=last['cumulative_uplink_s'];cum_local=last['cumulative_parallel_local_compute_s']
    cum_control=last['cumulative_calibration_compute_s'];cum_solver=last['cumulative_optimizer_compute_s']
    cum_requests=last['request_users_total'];delivered_requests=last['individual_request_deliveries_total']
    clock_start=time.perf_counter();prefix_wall=last['server_experiment_wall_s']
    probe=plan['fixed_forget_probe_ids']
    try:
        for rec in schedule[:args.max_rounds]:
            k = rec['round']; start = time.perf_counter()
            rate = c['learning_rate']*.5*(1+math.cos(math.pi*(k-1)/c['rounds']))
            learning = np.zeros((c['users'],len(theta)))
            unlearning = np.zeros_like(learning)
            retained_total = sum(map(len,rec['retained'])); forget_total = sum(map(len,rec['forget']))
            client_seconds = []; coupling = []
            for user, batch in enumerate(rec['batches']):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                if batch:
                    grad = local_gradient(model,theta,dataset,text,batch,device,c['microbatch_size'])
                    learning[user] = (-rate*len(rec['retained'][user])/retained_total*grad).cpu().numpy()
                forgotten = rec['forget'][user]
                if forgotten:
                    chosen = np.random.default_rng(c['seed']*100000+k*100+user).choice(forgotten,min(c['batch_size'],len(forgotten)),replace=False).tolist()
                    fg = local_gradient(model,theta,dataset,text,chosen,device,c['microbatch_size'])
                    if args.method == 'proposed' and batch:
                        pixels, labels = dataset.batch(batch[:c['microbatch_size']],device)
                        loss = lambda x: F.cross_entropy(model(pixels,**text,theta=x),labels)
                        fg, stat = projected_coupling(loss,theta,fg,model.visual_dimension,model.text_dimension,
                            rank=c['coupling_rank'],kappa_requested=c['coupling_kappa'],xi=c['coupling_xi'],
                            difference_step=c['coupling_difference_step'],seed=c['seed']+k*100+user)
                        coupling.append(dict(user=user,**stat))
                    unlearning[user] = (rate*len(forgotten)/forget_total*fg).cpu().numpy()
                torch.cuda.synchronize(); client_seconds.append(time.perf_counter()-t0)
            calibration_s = 0.; gradient = None
            if forget_total and args.method != 'aircomp_only':
                torch.cuda.synchronize(); t0 = time.perf_counter()
                gradient = local_gradient(model,theta,dataset,text,calibration_ids,device,c['microbatch_size']).cpu().numpy().astype(float)
                torch.cuda.synchronize(); calibration_s = time.perf_counter()-t0
            mode = 'hybrid' if args.method in ('proposed','hybrid_no_coupling') else args.method
            update, tx = communicate(learning,unlearning,channels,c,mode,c['seed']*100000+k,gradient)
            request_count = sum(bool(x) for x in rec['forget'])
            cum_requests += request_count
            if update is None:
                np.savez_compressed(out/f'failed_round_{k:03d}_updates.npz',learning=learning,unlearning=unlearning,channels=channels)
                write(out/'failure.json',dict(round=k,transmission=tx,requested_users=request_count,
                    completed_rounds=k-1,reason='No feasible physical design found; no update or request acknowledgement applied'))
                manifest.update(status='no_feasible_design_found',failed_round=k)
                break
            candidate = theta+torch.as_tensor(update,device=device,dtype=theta.dtype)
            if not torch.isfinite(candidate).all() or candidate[-1] > 20:
                raise RuntimeError('Model diverged; no clipping or silent ideal-link fallback')
            theta = candidate.detach()
            cum_e += tx['energy_bound_j']; cum_expected += tx['expected_energy_j']; cum_up += tx['uplink_s']
            cum_local += max(client_seconds); cum_control += calibration_s; cum_solver += tx['optimizer_wall_s']
            delivered_requests += tx['individual_requests_delivered']
            # Offline evaluation overhead is measured but not charged as deployment delay.
            before_eval = time.perf_counter()
            val = evaluate(model,theta,dataset,text,validation_ids,device,c['microbatch_size'])
            forget_eval = None
            if forget_total:
                ids = sorted(set().union(*map(set,rec['forget'])))
                selected = np.random.default_rng(c['seed']+k).permutation(ids)[:200].tolist()
                forget_eval = evaluate(model,theta,dataset,text,selected,device,c['microbatch_size'])
            persistent_forget = evaluate(model,theta,dataset,text,probe,device,c['microbatch_size'])
            eval_s = time.perf_counter()-before_eval
            row = dict(round=k,validation=val,current_request_forget=forget_eval,persistent_forget_evaluation=persistent_forget,learning_rate=rate,
                client_compute_s=client_seconds,parallel_local_compute_s=max(client_seconds),
                calibration_compute_s=calibration_s,optimizer_compute_s=tx['optimizer_wall_s'],
                uplink_s=tx['uplink_s'],evaluation_wall_s=eval_s,round_simulator_wall_s=time.perf_counter()-start,
                cumulative_communication_energy_j=cum_e,cumulative_expected_communication_energy_j=cum_expected,
                cumulative_uplink_s=cum_up,cumulative_parallel_local_compute_s=cum_local,
                cumulative_calibration_compute_s=cum_control,cumulative_optimizer_compute_s=cum_solver,
                cumulative_delay_without_optimizer_or_overhead_s=cum_up+cum_local+cum_control,
                cumulative_delay_with_optimizer_without_overhead_s=cum_up+cum_local+cum_control+cum_solver,
                server_experiment_wall_s=prefix_wall+time.perf_counter()-clock_start,
                request_users_total=cum_requests,individual_request_deliveries_total=delivered_requests,
                requested_samples=len(rec['deleted']),coupling=coupling,transmission=tx)
            with (out/'rounds.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            torch.save({'theta':theta.cpu(),'round':k},out/f'checkpoints/round_{k:03d}.pt')
            write(out/'progress.json',{key:row[key] for key in ['round','validation','cumulative_communication_energy_j','cumulative_uplink_s','cumulative_delay_with_optimizer_without_overhead_s','request_users_total','individual_request_deliveries_total']})
            print(json.dumps(dict(method=args.method,round=k,loss=val['loss'],accuracy=val['accuracy'],energy_j=cum_e,wall_s=row['server_experiment_wall_s'])),flush=True)
        else:
            manifest['status'] = 'window_completed'
    except Exception as error:
        manifest.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write(out/'manifest.json',manifest)


if __name__ == '__main__':
    main()
