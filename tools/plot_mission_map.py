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
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.86))
    fig.subplots_adjust(left=.012, right=.988, top=.985, bottom=.12, wspace=.045)
    axes[0].imshow(image, extent=limits, origin='upper', zorder=0)
    ax = axes[1]
    ax.set_facecolor('#fafbf9')
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
            ax.add_patch(Polygon(xy, facecolor='#dcebf0' if water else '#edf1e8',
                                 edgecolor='#c7dbe1' if water else '#e0e6da', lw=.35, zorder=1))
        if tags.get('waterway') in {'river', 'stream', 'canal'}:
            ax.plot(*xy.T, color='#d0e3e9', lw=1.7, zorder=1)
        road = tags.get('highway')
        if road:
            road_width = {'primary': 1.7, 'secondary': 1.4, 'tertiary': 1.1,
                          'residential': .8, 'service': .55}.get(road, .35)
            ax.plot(*xy.T, color='#e0e3df', lw=road_width + .4, zorder=2)
            ax.plot(*xy.T, color='white', lw=road_width, zorder=2)

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
    route = np.vstack([centers, centers[:1]])
    route_color = '#163a50'
    ax.plot(*route.T, color='white', lw=2.8, solid_capstyle='round', zorder=4)
    ax.plot(*route.T, color=route_color, lw=1.1, solid_capstyle='round', zorder=4)
    for a, b in zip(route, route[1:]):
        ax.annotate('', xy=a+.60*(b-a), xytext=a+.46*(b-a),
                    arrowprops=dict(arrowstyle='-|>', color=route_color,
                                    lw=1.1, mutation_scale=8), zorder=6)
    for number, (footprint, center) in enumerate(zip(display_footprints, centers), 1):
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
    axes[0].set_xlabel('(a) Satellite imagery', fontsize=10, labelpad=6)
    axes[1].set_xlabel('(b) Swarm trajectory', fontsize=10, labelpad=6)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / 'mission_satellite_swarm_cluster_1x2'
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(stem.with_suffix('.' + extension), dpi=450, facecolor='white')
    plt.close(fig)
    metadata = dict(
        layout='1 x 2', uav_count=10, mission_stops=len(spec['stops']),
        trajectory_source=str((MISSION/'trajectory.csv').relative_to(ROOT)),
        trajectory_sha256=hashlib.sha256((MISSION/'trajectory.csv').read_bytes()).hexdigest(),
        mission_semantics=spec['semantics'], imagery=imagery,
        osm_attribution='© OpenStreetMap contributors',
        osm_copyright_url='https://www.openstreetmap.org/copyright',
        extent_epsg3857=limits, route_offsets_applied=False,
        representation='Recorded swarm-centroid route; schematic dispersed circular UAV markers at each stop',
        marker_positions_are_schematic=True,
        marker_layout='Ten dots per stop, staggered inner and outer radii with deterministic jitter; recentered to the recorded centroid. These are not measured or simulated individual positions.',
        centroid_route_epsg3857=route.tolist(),
        formation_positions_epsg3857=[f.tolist() for f in footprints],
        display_positions_epsg3857=[f.tolist() for f in display_footprints],
        plot_style='IEEE double-column width; no footer captions or individual UAV legend',
    )
    stem.with_suffix('.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print('Created:', stem.with_suffix('.png'))


if __name__ == '__main__':
    main()
