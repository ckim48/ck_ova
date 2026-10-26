"""IEEE-style satellite / swarm-centroid figure using recorded geography.

Run with --download once to cache Esri imagery; subsequent runs are offline.
The trajectory and mission specification are read without changing simulations.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import urllib.parse
import urllib.request

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MISSION = ROOT / 'results/overnight_energy_v2/korea_mission_v1'
MAP = ROOT / 'data/korea_mission_daejeon_v1'
OUT = ROOT / 'results/figures'
SERVICE = 'https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer'
R = 6378137.0


def mercator(lat, lon):
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    return np.column_stack((R * np.radians(lon), R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))))


def read_json(path):
    return json.loads(path.read_text())


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=45) as response:
        data = json.load(response)
    if 'error' in data:
        raise RuntimeError(data['error'])
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--mode', choices=['snapshots', 'schematic', 'iot'], default='snapshots')
    parser.add_argument('--deployment', type=Path, default=ROOT/'results/iot_uav_deployment_v1')
    parser.add_argument('--transactions', type=Path,
                        help='Actual transaction records used to select the first transmitted update per site.')
    args = parser.parse_args()
    spec = read_json(MISSION / 'mission_spec.json')
    with (MISSION / 'trajectory.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert {int(row['uav']) for row in rows} == set(range(10))
    points = mercator([r['latitude'] for r in rows], [r['longitude'] for r in rows])
    scale = 1 / math.cos(math.radians(spec['projection']['lat0']))
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    width = np.ptp(points[:, 0]) + 360 * scale
    height = width * 1280 / 1800
    bbox = [center[0] - width / 2, center[1] - height / 2,
            center[0] + width / 2, center[1] + height / 2]
    cache = MAP / 'satellite_mission_1x2'
    image_path = cache.with_suffix('.jpg')
    meta_path = cache.with_suffix('.json')
    if args.download:
        service_meta = fetch_json(SERVICE + '?f=pjson')
        params = dict(f='json', bbox=','.join(map(str, bbox)), bboxSR=3857,
                      imageSR=3857, size='1800,1280', format='jpg', dpi=180)
        url = SERVICE + '/export?' + urllib.parse.urlencode(params)
        exported = fetch_json(url)
        with urllib.request.urlopen(exported['href'], timeout=45) as response:
            image_path.write_bytes(response.read())
        meta_path.write_text(json.dumps(dict(
            service=SERVICE, export_url=url, export=exported,
            copyright=service_meta['copyrightText'], requested_bbox=bbox,
            image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
        ), indent=2))
    if not image_path.exists() or not meta_path.exists():
        raise SystemExit('Satellite cache missing: run this script with --download.')
    imagery = read_json(meta_path)
    assert np.allclose(imagery['requested_bbox'], bbox, rtol=0, atol=.01)
    assert hashlib.sha256(image_path.read_bytes()).hexdigest() == imagery['image_sha256']
    extent = imagery['export']['extent']
    limits = [extent['xmin'], extent['xmax'], extent['ymin'], extent['ymax']]
    image = Image.open(image_path)
    assert image.width >= 1200 and image.height >= 800
    osm = read_json(MAP / 'osm_snapshot.json')

    plt.rcParams.update({'font.family': 'serif', 'font.size': 8,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    clean_snapshots = args.mode in ['snapshots', 'iot']
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.54 if clean_snapshots else 2.86))
    fig.subplots_adjust(left=.012, right=.988, top=.985,
                        bottom=.015 if clean_snapshots else .12, wspace=.045)
    axes[0].imshow(image, extent=limits, origin='upper', zorder=0)
    ax = axes[1]
    ax.set_facecolor('#f8faf7')
    for element in osm['elements']:
        geometry = element.get('geometry')
        if element['type'] != 'way' or not geometry:
            continue
        tags = element.get('tags', {})
        xy = mercator([g['lat'] for g in geometry], [g['lon'] for g in geometry])
        if (xy[:, 0].max() < limits[0] or xy[:, 0].min() > limits[1]
                or xy[:, 1].max() < limits[2] or xy[:, 1].min() > limits[3]):
            continue
        closed = np.linalg.norm(xy[0] - xy[-1]) < 1
        water = tags.get('natural') == 'water' or tags.get('waterway') == 'riverbank'
        leisure = tags.get('leisure') in {'park', 'garden', 'nature_reserve', 'pitch'}
        if closed and (water or leisure):
            ax.add_patch(Polygon(xy, facecolor='#cbdfe7' if water else '#e2ebda',
                                 edgecolor='#a9c5cf' if water else '#c7d5bd', lw=.35, zorder=1))
        if tags.get('waterway') in {'river', 'stream', 'canal'}:
            ax.plot(*xy.T, color='#98bfce', lw=2.0, zorder=1)
        road = tags.get('highway')
        if road:
            road_width = {'primary': 1.7, 'secondary': 1.4, 'tertiary': 1.1,
                          'residential': .8, 'service': .55}.get(road, .35)
            ax.plot(*xy.T, color='#b8c1ba', lw=road_width + .5, zorder=2)
            ax.plot(*xy.T, color='white', lw=road_width, zorder=2)

    deployment = None
    if args.mode == 'iot':
        deployment = read_json(args.deployment/'deployment.json')
        spec = read_json(args.deployment/'mission_spec.json')
        with (args.deployment/'trajectory.csv').open() as handle:
            rows = list(csv.DictReader(handle))
    # Keep recorded centroids and route; schematic markers communicate a swarm
    # rather than claiming that the simulated two-row formation changed.
    footprints, centers = [], []
    p = spec['projection']
    for stop in spec['stops']:
        formation = np.array(stop['positions_xy_m'])
        lat = p['lat0'] + np.degrees(formation[:, 1] / p['R_m'])
        lon = p['lon0'] + np.degrees(formation[:, 0] / (p['R_m'] * math.cos(math.radians(p['lat0']))))
        projected = mercator(lat, lon)
        footprints.append(projected)
        centers.append(projected.mean(axis=0))
    centers = np.array(centers)
    snapshots = []
    if args.mode == 'snapshots' or (deployment and args.transactions):
        if args.transactions is None:
            args.transactions = ROOT/'results/overnight_energy_v2/uav_single_model_two_dataset_v1/rsitmd_seed0/proposed_fu_guard_cliperase/transactions'
        first_by_site = {}
        for path in sorted(args.transactions.glob('*.json')):
            record = read_json(path)
            if record['radio']['status'] != 'feasible' or record['radio']['uplink_s'] <= 0:
                continue
            site = int(record['site'])
            if site not in first_by_site or record['start_s'] < first_by_site[site][1]['start_s']:
                first_by_site[site] = (path, record)
        assert set(first_by_site) == set(range(len(spec['stops']))), 'All four sites require actual transactions'
        for site, stop in enumerate(spec['stops']):
            path, record = first_by_site[site]
            head = int(record['radio']['head'])
            assert 0 <= head < 10
            assert stop['start_s'] <= record['start_s'] < record['end_s'] <= stop['end_s']
            # The prescribed positions are constant throughout each hover window.
            hover_rows = [r for r in rows if r['phase'] == 'inspection_hover' and int(r['site']) == site]
            assert {int(r['uav']) for r in hover_rows} == set(range(10))
            for row in hover_rows:
                assert np.allclose([float(row['east_m']), float(row['north_m'])],
                                   stop['positions_xy_m'][int(row['uav'])], atol=1e-6)
            snapshots.append(dict(site=site, time_s=record['start_s'], end_s=record['end_s'],
                                  head_uav_zero_based=head, display_head='U'+str(head+1),
                                  source=str(path.relative_to(ROOT)),
                                  source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                  selection='Earliest feasible positive-duration transmission at this site',
                                  kind=record['kind'], sequence=record['sequence'],
                                  positions_xy_m=stop['positions_xy_m']))
    elif deployment:
        snapshots = [dict(site=s['site'],time_s=s['time_s'],head_uav_zero_based=s['head_zero_based'],
                          display_head='U'+str(s['head_zero_based']+1),
                          head_kind='nominal_packet_planning_head_not_FL_FU_result',
                          positions_xy_m=s['uav_positions_xy_m']) for s in deployment['sites']]
    display_footprints = []
    for index, center in enumerate(centers):
        rng = np.random.default_rng(480 + index)
        angles = np.concatenate([
            np.arange(3) * 2*np.pi/3 + .28,
            np.arange(7) * 2*np.pi/7,
        ]) + index * .47 + rng.uniform(-.13, .13, 10)
        radii = np.concatenate([np.full(3, 43.), np.full(7, 94.)])
        radii *= rng.uniform(.88, 1.12, 10)
        offsets = np.column_stack([np.cos(angles), np.sin(angles)]) * radii[:, None]
        offsets -= offsets.mean(axis=0)
        display_footprints.append(center + offsets * scale)
    if clean_snapshots:
        display_footprints = footprints
    if deployment:
        for s, footprint in zip(deployment['sites'],footprints):
            ground=np.array(s['iot_positions_xy_m'])
            lat=p['lat0']+np.degrees(ground[:,1]/p['R_m'])
            lon=p['lon0']+np.degrees(ground[:,0]/(p['R_m']*math.cos(math.radians(p['lat0']))))
            projected_ground=mercator(lat,lon)
            for node,uav in zip(projected_ground,s['association_zero_based']):
                ax.plot([node[0],footprint[uav,0]],[node[1],footprint[uav,1]],
                        color='#5d9d98',lw=.3,alpha=.5,zorder=3)
            ax.scatter(*projected_ground.T,s=5,facecolors='#91aaa8',edgecolors='white',
                       linewidths=.25,zorder=5)
    route = np.vstack([centers, centers[:1]])
    route_color = '#163a50'
    ax.plot(*route.T, color='white', lw=2.8, solid_capstyle='round', zorder=4)
    ax.plot(*route.T, color=route_color, lw=.8 if snapshots else 1.1,
            linestyle=(0, (3, 3)) if snapshots else '-', solid_capstyle='round', zorder=4)
    for a, b in zip(route, route[1:]):
        ax.annotate('', xy=a+.60*(b-a), xytext=a+.46*(b-a),
                    arrowprops=dict(arrowstyle='-|>', color=route_color,
                                    lw=1.1, mutation_scale=8), zorder=6)
    for number, (footprint, center) in enumerate(zip(display_footprints, centers), 1):
        if snapshots:
            shot = snapshots[number-1]
            head = shot['head_uav_zero_based']
            if deployment:
                ax.scatter(*footprint.T,marker='*',s=65,facecolors='#278c83',edgecolors='white',
                           linewidths=.55,zorder=7)
                ax.scatter(*footprint[head],marker='*',s=110,facecolors='#e6a13d',edgecolors='#895608',
                           linewidths=.6,zorder=9)
                continue
            ax.scatter(*footprint.T, s=49, facecolors='#e9f4f1', edgecolors='#4d9690',
                       linewidths=.65, zorder=7)
            ax.scatter(*footprint[head], s=85, facecolors='white', edgecolors='#cf942e',
                       linewidths=.85, zorder=8)
            ax.scatter(*footprint[head], s=49, facecolors='#fff1d9', edgecolors='#cf942e',
                       linewidths=.65, zorder=9)
            for uav, xy in enumerate(footprint):
                ax.annotate(str(uav+1), xy=xy, xytext=(0, -.15),
                            textcoords='offset points',
                            color='#8e5910' if uav == head else '#205e5a',
                            fontsize=4.6, fontfamily='DejaVu Sans',
                            ha='center', va='center', fontweight='normal', zorder=10)
            continue
        ax.scatter(*footprint.T, s=19, facecolors='#64ada6', edgecolors='white',
                   linewidths=.8, zorder=7)
        ax.text(*center, str(number), color=route_color, ha='center', va='center',
                fontsize=7, fontweight='bold', zorder=8,
                bbox=dict(boxstyle='circle,pad=.22', facecolor='white',
                          edgecolor=route_color, linewidth=.55))

    for panel in axes:
        panel.set_xlim(limits[:2]); panel.set_ylim(limits[2:])
        panel.set_aspect('equal'); panel.set_xticks([]); panel.set_yticks([])
        for spine in panel.spines.values():
            spine.set_linewidth(.55); spine.set_color('#252525')
        if clean_snapshots:
            continue
        color = 'white' if panel is axes[0] else '#222222'
        stroke = [pe.withStroke(linewidth=2, foreground='#333333')] if panel is axes[0] else []
        arrow = panel.annotate('N', xy=(.06, .95), xytext=(.06, .82), xycoords='axes fraction',
                              ha='center', va='center', color=color, fontsize=8,
                              arrowprops=dict(arrowstyle='-|>', color=color, lw=.9))
        arrow.set_path_effects(stroke)
        x = limits[0] + .065 * (limits[1]-limits[0])
        y = limits[2] + .06 * (limits[3]-limits[2])
        length = 200 * scale
        line, = panel.plot([x, x+length], [y, y], color=color, lw=1.6, zorder=12)
        line.set_path_effects(stroke)
        text = panel.text(x+length/2, y+.018*(limits[3]-limits[2]), '200 m',
                          color=color, ha='center', fontsize=7, zorder=12)
        text.set_path_effects(stroke)
    if not clean_snapshots:
        axes[0].set_xlabel('(a) Satellite imagery', fontsize=10, labelpad=6)
        axes[1].set_xlabel('(b) Swarm trajectory', fontsize=10, labelpad=6)
    elif not deployment:
        labels = [t.get_text() for panel in axes for t in panel.texts if t.get_text()]
        assert sorted(labels) == sorted([str(i) for i in range(1,11)]*4)
    if deployment:
        handles=[Line2D([],[],marker='o',linestyle='none',markersize=3.3,color='#91aaa8',label='IoT'),
                 Line2D([],[],marker='*',linestyle='none',markersize=6.5,color='#278c83',label='UAV'),
                 Line2D([],[],marker='*',linestyle='none',markersize=7,color='#e6a13d',markeredgecolor='#895608',markeredgewidth=.5,label='Head')]
        ax.legend(handles=handles,loc='upper right',ncol=3,frameon=True,
                  facecolor='white',edgecolor='none',framealpha=.95,
                  fontsize=6.5,handlelength=.8,handletextpad=.45,columnspacing=1.2,borderpad=.6)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / ('mission_satellite_uav_snapshots_1x2' if snapshots else 'mission_satellite_swarm_cluster_1x2')
    if deployment:
        stem=OUT/('mission_satellite_iot_uav_actual_heads_1x2' if args.transactions else 'mission_satellite_iot_uav_planned_1x2')
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(stem.with_suffix('.' + extension), dpi=450, facecolor='white')
        if extension == 'svg':
            svg = stem.with_suffix('.svg')
            svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)
    metadata = dict(
        layout='1 x 2', uav_count=10, mission_stops=len(spec['stops']),
        trajectory_source=str(((args.deployment if deployment else MISSION)/'trajectory.csv').relative_to(ROOT)),
        trajectory_sha256=hashlib.sha256(((args.deployment if deployment else MISSION)/'trajectory.csv').read_bytes()).hexdigest(),
        mission_semantics=spec['semantics'], imagery=imagery,
        osm_attribution='© OpenStreetMap contributors',
        osm_copyright_url='https://www.openstreetmap.org/copyright',
        extent_epsg3857=limits, route_offsets_applied=False,
        representation=('Four time-separated simulation snapshots with actual recorded formations and selected heads' if snapshots else 'Recorded swarm-centroid route; schematic dispersed circular UAV markers at each stop'),
        marker_positions_are_schematic=not bool(snapshots),
        marker_layout=('Exact prescribed hover positions, validated against trajectory.csv; not real-world flight telemetry' if snapshots else 'Ten dots per stop, staggered inner and outer radii with deterministic jitter; recentered to the recorded centroid. These are not measured or simulated individual positions.'),
        snapshot_records=snapshots,
        id_convention='Figure U1–U10 correspond to logged user indices 0–9',
        head_semantics='Head of the selected individual transmission, not a fixed head for the whole region; different sites show the same ten UAVs at different times',
        centroid_route_epsg3857=route.tolist(),
        formation_positions_epsg3857=[f.tolist() for f in footprints],
        display_positions_epsg3857=[f.tolist() for f in display_footprints],
        plot_style=('IEEE double-column width; UAV IDs only, sans-serif numerals; no explanatory text, panel captions, north labels or scale labels' if clean_snapshots else 'IEEE double-column width; no footer captions or individual UAV legend'),
    )
    if deployment:
        metadata.update(representation='IoT circles, optimized UAV stars and selected head; actual ground coordinates are synthetic scenario inputs.',
                        deployment_source=str((args.deployment/'deployment.json').relative_to(ROOT)),
                        deployment_sha256=hashlib.sha256((args.deployment/'deployment.json').read_bytes()).hexdigest(),
                        head_source='actual FL/FU transactions' if args.transactions else 'nominal packet planning objective; not a learned FL/FU result',
                        marker_layout='UAV coordinates are optimizer outputs, IoT coordinates are predeclared synthetic demand; no decorative position jitter',
                        plot_style='IEEE double-column width; IoT/UAV/head legend only, no explanatory labels')
    stem.with_suffix('.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print('Created:', stem.with_suffix('.png'))


if __name__ == '__main__':
    main()
