"""Shrink map panels another20%, while using the charts' unscaled12pt type."""
import importlib.util
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import downlink_bandwidth_v1 as study

ROOT, read, write, sha = study.ROOT, study.read, study.write, study.sha
SOURCE = ROOT / 'results/paper_figures_20261026/mission_map_1x2'
PREVIOUS_SCALE = .72
RELATIVE_SCALE = .8
SCALE = PREVIOUS_SCALE * RELATIVE_SCALE


def pdf_size(path):
    info = subprocess.check_output(['pdfinfo', str(path)], text=True)
    match = re.search(r'Page size:\s+([\d.]+) x ([\d.]+) pts', info)
    assert match, info
    return [float(match[1]), float(match[2])]


def render(dest):
    # Keep the user-selected Yeouido map when rebuilding the chart gallery.
    current = ROOT / 'final_final_charts/yeouido_map'
    if (current/'manifest.json').exists() and (current/'previous_daejeon/manifest.json').exists():
        import yeouido_mission_map_v1
        return yeouido_mission_map_v1.install(dest)
    import proposed_pks_report as chart
    chart.prior.chart.configure()
    reference, panels = chart.prior.chart.style.canvas('')
    reference_font = panels[0].get_xticklabels()[0].get_fontproperties().copy()
    assert reference_font.get_size_in_points() == 12.
    plt.close(reference)
    old = read(SOURCE.with_suffix('.json'))
    captured = {}

    def restyle(fig, axes):
        chart.prior.chart.configure()
        plt.rcParams['savefig.bbox'] = None
        fig.canvas.draw()
        original_size = fig.get_size_inches().copy()
        original_panels = [np.array([ax.get_position().width, ax.get_position().height])*original_size for ax in axes]
        side = float(original_panels[0][0] * SCALE)
        assert all(np.allclose(p, original_panels[0]) for p in original_panels)
        assert np.isclose(original_panels[0][0], original_panels[0][1])
        # Preserve physical type size; reserve space around the smaller square maps.
        left, gap, right, bottom, top = .47, .59, .19, .33, .17
        width, height = left + 2*side + gap + right, bottom + side + top
        fig.set_size_inches(width, height, forward=True)
        for index, ax in enumerate(axes):
            ax.set_position([(left + index*(side+gap))/width, bottom/height, side/width, side/height])
            ax.tick_params(axis='both', labelsize=12, direction='out', pad=3, length=3, width=.9)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontproperties(reference_font)
            for spine in ax.spines.values():
                spine.set_linewidth(.9)
            for line in ax.lines:
                line.set_linewidth(line.get_linewidth()*SCALE)
                line.set_markersize(line.get_markersize()*SCALE)
            for collection in ax.collections:
                if hasattr(collection, 'get_sizes'):
                    collection.set_sizes(collection.get_sizes()*SCALE**2)
                collection.set_linewidths(collection.get_linewidths()*SCALE)
            for patch in ax.patches:
                patch.set_linewidth(patch.get_linewidth()*SCALE)
            for text in ax.texts:
                if text.get_text():
                    text.set_fontproperties(reference_font)
                arrow = getattr(text, 'arrow_patch', None)
                if arrow is not None:
                    arrow.set_linewidth(arrow.get_linewidth()*SCALE)
                    arrow_scale = arrow.get_mutation_scale()*SCALE
                    arrow.set_mutation_scale(arrow_scale)
                    text.arrowprops['mutation_scale'] = arrow_scale
                    arrow.set_path_effects([pe.withStroke(linewidth=3.4*SCALE, foreground='white')])
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        labels = [text for ax in axes for text in ax.get_xticklabels()+ax.get_yticklabels()+ax.texts if text.get_text()]
        assert len(labels) == 25
        assert all(text.get_fontsize() == reference_font.get_size_in_points() for text in labels)
        assert all(fig.bbox.contains(*text.get_window_extent(renderer).min) and
                   fig.bbox.contains(*text.get_window_extent(renderer).max) for text in labels)
        bounds = [[ax.get_position().width*width, ax.get_position().height*height] for ax in axes]
        captured.update(figure_size_inches=[width, height],
                        original_map_panel_size_inches=[p.tolist() for p in original_panels],
                        previous_map_panel_size_inches=[(p*PREVIOUS_SCALE).tolist() for p in original_panels],
                        map_panel_size_inches=bounds, tick_fontsize_pt=12., visit_fontsize_pt=12.,
                        font_family=reference_font.get_family(), labels_inside_canvas=True,
                        tick_values=[0, 500, 1000, 1500, 2000], output_dpi=350)

    old_argv = sys.argv[:]
    with tempfile.TemporaryDirectory(prefix='ova_compact_map_') as tmp:
        try:
            renderer_path = ROOT / 'tools/plot_mission_map_visible_clusters.py'
            transactions = (ROOT / old['mission_timeline_source']).parent / 'transactions'
            sys.argv = [str(renderer_path), '--global-iot', '--mode', 'iot', '--transactions', str(transactions), '--output', tmp]
            spec = importlib.util.spec_from_file_location('compact_map_source', renderer_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            code = module.source
            needle = "    for extension in ['png', 'pdf', 'svg']:"
            assert code.count(needle) == 1 and code.count('dpi=450') == 1
            code = code.replace(needle, '    restyle_map(fig, axes)\n' + needle).replace('dpi=450', 'dpi=350')
            with plt.rc_context():
                exec(compile(code, str(ROOT / 'tools/plot_mission_map.py'), 'exec'),
                     dict(__name__='__main__', __file__=str(ROOT / 'tools/plot_mission_map.py'), restyle_map=restyle))
        finally:
            sys.argv = old_argv
        produced = read(Path(tmp) / 'mission_map_1x2.json')
        geometry_keys = ['trajectory_sha256', 'extent_epsg3857', 'snapshot_records',
                         'centroid_route_epsg3857', 'formation_positions_epsg3857',
                         'display_positions_epsg3857', 'per_uav_paths_epsg3857', 'route_arrows']
        assert all(produced[k] == old[k] for k in geometry_keys)
        dest.mkdir(parents=True, exist_ok=True)
        name = '06_mission_map_1x2'
        for extension in ['png', 'pdf', 'svg']:
            shutil.copyfile(Path(tmp) / ('mission_map_1x2.' + extension), dest / (name + '.' + extension))
    rendered_size = pdf_size(dest / (name + '.pdf'))
    assert np.allclose(rendered_size, np.array(captured['figure_size_inches'])*72, atol=.001)
    sources = [SOURCE.with_suffix('.json'), SOURCE.with_suffix('.svg'),
               ROOT / 'tools/plot_mission_map.py', ROOT / 'tools/plot_mission_map_reference.py',
               ROOT / 'tools/plot_mission_map_cluster.py', renderer_path,
               ROOT / 'tools/iot_sensitivity_v1/selected_pair.py', ROOT / 'tools/plot_event_mission_reference_style.py']
    audit = dict(passed=True, linear_scale=SCALE, previous_linear_scale=PREVIOUS_SCALE,
                 scale_relative_to_previous=RELATIVE_SCALE, scale_scope='Map axes in physical inches; text stays12pt. Margins adjusted for label clearance.',
                 geometry_and_snapshot_heads_unchanged=True, geometry_keys_verified=geometry_keys,
                 original_svg_content_preserved=False, vector_pdf=True,
                 page_size_points=rendered_size, source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sources},
                 original_snapshot_run='results/iot_uav_remoteclip_v4_event_mission/capera_seed0/proposed_fu_guard_cliperase',
                 snapshot_scope='Preserved earlier scenario snapshots; not new P-KS bandwidth-run head decisions.', **captured)
    write(dest / 'data/mission_map.json', dict(old, compact_variant=audit, figure_size_inches=captured['figure_size_inches']))
    figure = dict(name=name, title='Satellite / IoT–UAV map: smaller panels with matching12pt labels',
                  kind='map', axes=[], linear_scale=SCALE, physical_size_inches=captured['figure_size_inches'],
                  files=[dict(file=name+'.'+ext, sha256=sha(dest/(name+'.'+ext))) for ext in ['png', 'pdf', 'svg']])
    return figure, audit
