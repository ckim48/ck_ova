"""Audit one-step FL sample coefficient history; not an unlearning result."""
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True)
    root=p.parse_args().project.resolve();suite=root/'results/comparisons_seed0_solver_v2'
    c=json.loads((suite/'config.json').read_text());source=suite/'coupled_hybrid/schedule.json'
    schedule=json.loads(source.read_text());masses=defaultdict(float);events=[];total_mass=0.
    expected=json.loads((suite/'suite_metadata.json').read_text())['source_sha256'];assert all(sha(root/n)==h for n,h in expected.items())
    for rec in schedule:
        rate=c['learning_rate']*.5*(1+math.cos(math.pi*(rec['round']-1)/c['rounds']))
        count=sum(map(len,rec['forget']))
        if count:
            clients=[]
            for user,ids in enumerate(rec['forget']):
                if not ids:continue
                weights=np.array([masses[int(i)] for i in ids]);assert np.all(weights>0)
                original=rate*len(ids)/count;history=float(weights.sum())
                clients.append({'user':user,'samples':len(ids),'original_eta_times_beta':original,
                                'sum_historical_sgd_coefficients':history,
                                'history_mass_over_original_coefficient':history/original,
                                'historical_sample_coefficient_min':float(weights.min()),
                                'historical_sample_coefficient_median':float(np.median(weights)),
                                'historical_sample_coefficient_max':float(weights.max())})
            events.append({'round':rec['round'],'learning_rate':rate,'clients':clients,
                           'total_history_mass_over_original_eta':sum(x['sum_historical_sgd_coefficients'] for x in clients)/rate})
        denominator=sum(map(len,rec['retained']))
        for retained,batch in zip(rec['retained'],rec['batches']):
            assert batch and set(batch)<=set(retained)
            alpha=len(retained)/denominator;coefficient=rate*alpha/len(batch)
            for i in batch:masses[int(i)]+=coefficient
            total_mass+=rate*alpha
    assert abs(sum(masses.values())-total_mass)<1e-10
    result={'at_kst':datetime.now().isoformat(timespec='seconds'),'status':'coefficient_history_diagnostic_only',
            'definition':'c_z(k) = sum over prior sampled learning steps of eta_L(s)*alpha_u(s)/batch_size_u(s)',
            'events':events,'total_learning_coefficient_mass':total_mass,
            'candidate':'A separate history-weighted first-order update sum_z c_z * grad f(theta_current;z), with local weighted minibatch estimator. It is not implemented or validated by this coefficient audit.',
            'source_sha256':expected,'schedule_sha256':sha(source),'script_sha256':sha(__file__),
            'limits':['A coefficient ratio is not a vector norm ratio: historical/current gradients, per-sample weights, nonlinear paths and cross-modal preconditioning differ.',
                      'The current eta_U*beta gradient-ascent rule is not claimed equivalent to a historical-gradient inverse.',
                      'Historical coefficients motivate a first-order candidate; they do not prove a larger update improves retraining similarity or retention.',
                      'No final model/test predictions were used to tune coefficients; all values derive from the frozen SGD schedule.',
                      'No training/inference/radio optimization, GPU promotion or queued-source change. Any future method requires actual weighted-gradient CPU/task checks and equal comparator treatment.']}
    out=root/'results/overnight_energy_v2/history_mass_audit_v1';out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
