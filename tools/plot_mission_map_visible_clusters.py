"""Larger map, visible snapshot enclosures and five directed transition arrows.

Reuse the registered map renderer and geometry. Actual head snapshots are
validated against the completion-driven timeline, not the old 60-second slots.
"""
from pathlib import Path
import importlib.util

path = Path(__file__).with_name('plot_mission_map_cluster.py')
spec = importlib.util.spec_from_file_location('visible_cluster_source', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
source = module.source


def replace(old, new, count=1):
    global source
    assert source.count(old) == count, (old, source.count(old))
    source = source.replace(old, new, count)


replace("    args = parser.parse_args()",
        "    parser.add_argument('--output', type=Path, default=ROOT/'results/paper_figures_20261026')\n"
        "    args = parser.parse_args()\n"
        "    assert args.transactions is not None, 'Use completed mission transactions for actual heads'\n"
        "    timeline_path = args.transactions.parent/'mission_timeline.json'\n"
        "    mission_timeline = read_json(timeline_path)")
replace("            assert stop['start_s'] <= record['start_s'] < record['end_s'] <= stop['end_s']",
        "            active_site = mission_timeline['sites'][site]\n"
        "            assert active_site['site'] == site\n"
        "            assert active_site['collection_end_s'] <= record['start_s'] < record['end_s'] <= active_site['departure_s']")
replace('All four sites require actual transactions', 'All five sites require actual transactions')
replace('figsize=(9.4, 4.85)', 'figsize=(10.4, 5.35)')
replace("facecolor=(.22,.46,.68,.075),edgecolor=(.22,.46,.68,.35),\n                                 linewidth=.8,linestyle=(0,(4,3))",
        "facecolor=(.16,.43,.72,.18),edgecolor=(.12,.35,.62,.88),\n                                 linewidth=1.25,linestyle=(0,(4,2.5))")
replace("color='#28608b',alpha=.23,lw=.65", "color='#28608b',alpha=.25,lw=.7")
replace("marker='*',s=70,facecolors='#3a78b5',edgecolors='#14202a',alpha=.65,\n                           linewidths=.5",
        "marker='*',s=88,facecolors='#286bb3',edgecolors='#12365a',alpha=.9,\n                           linewidths=.6")
replace("marker='*',s=105,facecolors='#df6249',edgecolors='#30201b',alpha=.9,\n                           linewidths=.55",
        "marker='*',s=125,facecolors='#df6249',edgecolors='#30201b',alpha=.95,\n                           linewidths=.65")
replace("route_color = '#245c82'", "route_color = '#174c77'")
replace("ax.plot(*route.T, color='white', lw=2.8", "ax.plot(*route.T, color='white', lw=4.2")
replace("lw=.8 if snapshots else 1.1", "lw=1.4 if snapshots else 1.4")
replace("""    for a, b in zip(route, route[1:]):
        ax.annotate('', xy=a+.60*(b-a), xytext=a+.46*(b-a),
                    arrowprops=dict(arrowstyle='-|>', color=route_color,
                                    lw=1.1, mutation_scale=8), zorder=6)""",
        """    route_arrows = []
    for leg, (a, b) in enumerate(zip(route, route[1:])):
        tip, tail = a+.67*(b-a), a+.33*(b-a)
        arrow = ax.annotate('', xy=tip, xytext=tail,
                            arrowprops=dict(arrowstyle='-|>', color=route_color,
                                            lw=1.6, mutation_scale=17), zorder=6)
        arrow.arrow_patch.set_path_effects([pe.withStroke(linewidth=3.4, foreground='white')])
        route_arrows.append(dict(from_visit=leg+1, to_visit=(leg+1)%5+1,
                                 tail_epsg3857=tail.tolist(), tip_epsg3857=tip.tolist()))
    assert len(route_arrows) == 5""")
replace("ha='center',va='center',fontsize=10,\n                    color='#234f72'",
        "ha='center',va='center',fontsize=11,\n                    color='#234f72'")
replace("    OUT.mkdir(parents=True, exist_ok=True)",
        "    OUT = args.output\n    OUT.mkdir(parents=True, exist_ok=True)")
replace("    stem=stem.with_name(stem.name+'_cluster_motion')", "    stem=OUT/'mission_map_1x2'")
replace("    stem.with_suffix('.json').write_text",
        "    metadata.update(visual_variant='visible_clusters_actual_event_mission',figure_size_inches=[10.4,5.35],cluster_fill_alpha=.18,cluster_edge_alpha=.88,route_arrows=route_arrows,route_arrow_semantics='Direction of centroid transitions 1-2-3-4-5-1; actual ten-UAV paths remain separately drawn',mission_timeline_source=str(timeline_path.relative_to(ROOT)),mission_timeline_sha256=hashlib.sha256(timeline_path.read_bytes()).hexdigest(),plot_style='Larger map; translucent blue enclosures and directed centroid route; five numeric visit labels and numeric endpoint ticks only',source_renderer='tools/plot_mission_map_visible_clusters.py')\n"
        "    stem.with_suffix('.json').write_text")

if __name__ == '__main__':
    exec(compile(source, str(module.module.source_path), 'exec'),
         dict(__name__='__main__', __file__=str(module.module.source_path)))
