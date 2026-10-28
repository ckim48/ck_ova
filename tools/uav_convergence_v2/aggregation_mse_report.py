"""Audit and export the actual MSE-budget/energy sweep without rescaling data."""
import copy
import math
from pathlib import Path

import aggregation_mse_v1 as study
import proposed_pks_report as current

ROOT, read, write, sha = study.ROOT, study.read, study.write, study.sha
prior, plots = current.prior, current.plots


def source(case, dataset, method):
    value = next(v for v in study.VALUES if study.case_id(v) == case)
    return study.source(value, dataset, method)


def collect():
    assert read(study.OUT / 'campaign_finished.json')['passed']
    rows, audits = [], []
    sources = prior.source, plots.source
    prior.source = plots.source = source
    hashes = {}
    def digest(path):
        key = str(path.resolve())
        if key not in hashes:
            hashes[key] = sha(path)
        prior.SOURCES[str(path.relative_to(ROOT))] = hashes[key]
        return hashes[key]
    try:
        for tau in study.VALUES:
            for ds in study.DATASETS:
                baseline = study.source(study.BASE, ds, 'gate_select16')
                base_settings = prior.load(baseline.parents[1] / 'settings.json')
                for method in study.METHODS:
                    arm = study.source(tau, ds, method)
                    settings = prior.load(arm.parents[1] / 'settings.json')
                    config = copy.deepcopy(settings['config'])
                    assert config.pop('relative_tolerance') == math.sqrt(tau)
                    base_config = copy.deepcopy(base_settings['config'])
                    base_config.pop('relative_tolerance')
                    assert config == base_config
                    assert settings['energy_model'] == base_settings['energy_model']
                    plan = copy.deepcopy(prior.load(arm.parent / 'plan.json'))
                    plan['config']['relative_tolerance'] = .2
                    assert plan == prior.load(baseline.parent / 'plan.json')
                    for name in ['requests.json', 'initial20/final.pt', 'reference200/final.pt', 'feature_cache/features.pt']:
                        assert digest(arm.parent / name) == digest(baseline.parent / name), (tau, ds, method, name)
                    for name in ['deployment.json', 'mission_spec.json']:
                        assert digest(arm.parents[1] / name) == digest(baseline.parents[1] / name)
                    row = plots.ledger(study.case_id(tau), ds, method)
                    assert row['FL_rounds'] == 200 and row['FU_cycles'] == 225
                    tx = [prior.load(f) for f in sorted((arm / 'transactions').glob('*.json'))]
                    ratios, conditional_mse, mse_bounds = [], [], []
                    for t in tx:
                        radio = t['radio']
                        mse, bound = radio['conditional_rmse']**2, radio['epsilon']**2
                        assert mse <= bound*(1+6e-5)+1e-18
                        assert radio['max_power_w'] <= config['power_limit_w']*(1+2e-6)
                        ratios.append(mse/bound)
                        conditional_mse.append(mse)
                        mse_bounds.append(bound)
                        if method == 'gate_select16' and radio['uplink_s'] > 0:
                            norm = radio['FU_norm_bound'] if radio['fu_noise_guard_active'] else radio['FL_norm_bound']
                            expected = max(config['absolute_tolerance_floor'], math.sqrt(tau)*norm)
                            assert math.isclose(radio['epsilon'], expected, rel_tol=1e-9, abs_tol=1e-14)
                        if method in ['iscc', 'salun']:
                            assert mse == 0.
                    row.update(mse_threshold=tau, relative_rmse_coefficient=math.sqrt(tau),
                               max_conditional_mse=max(conditional_mse), max_absolute_mse_budget=max(mse_bounds),
                               max_mse_budget_utilization=max(ratios),
                               numerical_floor_transactions=sum(t['radio']['epsilon'] <= 1e-8*(1+1e-12) for t in tx),
                               uplink_s=sum(t['radio']['uplink_s'] for t in tx),
                               downlink_s=sum(t['timing']['downlink_s'] for t in tx),
                               origin='reused_baseline' if tau == study.BASE else 'fresh_mse_execution')
                    rows.append(row)
                    audits.append(dict(dataset=ds, method=method, threshold=tau, source=row['source'],
                                       shared_inputs_verified=True, complete_workload=True, costs_recomputed=True,
                                       conditional_squared_error_within_budget=True, hard_power_limit_verified=True))
    finally:
        prior.source, plots.source = sources
    for ds in study.DATASETS:
        for method in ['iscc', 'salun']:
            group = [r for r in rows if r['dataset'] == ds and r['method'] == method]
            for metric in ['total_energy_j', 'delay_s', 'retained_mR_percent', 'retained_loss', 'forget_KL']:
                assert all(math.isclose(r[metric], group[0][metric], rel_tol=0, abs_tol=1e-8) for r in group)
    return rows, dict(passed=True, arms=40, fresh_arms=32, reused_arms=8, rows=audits,
                      digital_control_cost_and_quality_unchanged=True,
                      definition='Dimensionless squared relative RMSE budget; original method-specific norm bounds and FU guard retained.')


def render(dest):
    protocol = study.register()
    rows, audit = collect()
    name = '09_aggregation_mse_energy_1x2'
    fig, axes = prior.chart.style.canvas('Total energy (kJ)')
    panels = []
    for ax, ds in zip(axes, study.DATASETS):
        values = []
        for i, (method, label) in enumerate(study.METHODS.items()):
            y = [next(r['total_energy_j']*.001 for r in rows if (r['mse_threshold'], r['dataset'], r['method']) == (tau, ds, method)) for tau in study.VALUES]
            prior.chart.style.curve(ax, study.VALUES, y, i, label).set_markevery(1)
            values += y
            prior.SERIES.append(dict(figure=name, dataset=ds, method=method, kind='line', axis='aggregation_mse',
                                     x=study.VALUES, y=y, metric='total_energy_j', scale=.001,
                                     normalized_relative_budget=True))
        prior.chart.regular_axis(ax, 'x', study.VALUES, 3)
        ax.set_xlabel('Aggregation MSE threshold')
        panels.append(values)
    prior.chart.finish_axes(axes, panels)
    prior.save(fig, axes, name, 'Aggregation MSE threshold versus total energy')
    write(dest / 'data/mse_metrics.json', rows)
    prior.write_csv(dest / 'data/mse_metrics.csv', rows)
    write(dest / 'data/mse_audit.json', audit)
    write(dest / 'data/mse_protocol.json', protocol)
    prior.SOURCES[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    prior.SOURCES[str(Path(study.__file__).relative_to(ROOT))] = sha(Path(study.__file__))
    return audit
