"""Read-only source audit for the explicitly authorized two proposed seed0 pilots.

Does not launch training, choose a clock, or change any existing experiment.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
from gallery import prepare_plan

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'results/overnight_energy_v2/korea_joint_seed0_pilot_v2'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=True)


def main():
    old = ROOT / 'results/overnight_energy_v2/korea_mission_v1'
    protocol = json.loads((old / 'protocol.json').read_text())
    for name, expected in protocol['source_hashes'].items():
        assert sha(ROOT / name) == expected, name
    report = dict(version=HERE.name, prepared_unix=time.time(), clock_policy='pending_user_clarification',
                  launch_authorized_scope='Proposed only, RSITMD seed0 and RSICD seed0; preserve all previous results.',
                  original_sources_verified=True, sources=protocol['source_hashes'], datasets={})
    for dataset in ('RSITMD', 'RSICD'):
        source = ROOT / 'results/overnight_energy_v2' / ('retrieval_long200_v3' if dataset == 'RSITMD' else 'rsicd_cap100_v1')
        reqsource = source if dataset == 'RSICD' else ROOT / 'results/overnight_energy_v2/fourway_cap100_v1'
        raw = json.loads((source / 'plan.json').read_text())
        requests = json.loads((reqsource / 'requests.json').read_text())
        plan, check = prepare_plan(raw, requests)
        files = dict(initial20=source/'proposed/round_020.pt',
                     proxy20=source/'proxy20/final.pt', reference200=source/'reference200/final.pt',
                     prefix_cost=source/'common_prefix_cost.json')
        if dataset == 'RSITMD':
            files.update(proxy20=ROOT/'results/overnight_energy_v2/proxy_objective_v2/proxies/analog/round_020.pt',
                         reference200=source/'retrain/round_200.pt', prefix_cost=reqsource/'common_prefix_cost.json')
        check['source_files'] = {str(p.relative_to(ROOT)): sha(p) for p in [source/'plan.json', reqsource/'requests.json', *files.values()]}
        check['checkpoint_paths'] = {k: str(p.relative_to(ROOT)) for k, p in files.items()}
        check['seed'] = 0
        check['method'] = 'proposed_fu_guard_cliperase'
        dest = OUT / 'preparation' / dataset.lower()
        write(dest/'plan.json', plan)
        write(dest/'original_requests.json', requests)
        write(dest/'coverage_check.json', check)
        report['datasets'][dataset] = check
    write(OUT/'preparation.json', report)
    print(json.dumps(dict(prepared=True, training_launched=False,
                         galleries={k: {x: v[x] for x in ('gallery_count', 'previously_missing_count')} for k,v in report['datasets'].items()})))


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
