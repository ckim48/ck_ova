"""Yeouido map and synthetic IoT-driven geometry; no FL/FU training claims.

The existing placement solver is reused on new OSM demand. Plot heads are its
nominal packet-planning heads, not head decisions from the Daejeon experiments.
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as PolygonPath
from matplotlib.patches import Polygon, Ellipse
import matplotlib.patheffects as pe
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT/'data/korea_mission_yeouido_v1'
OUT = ROOT/'final_final_charts/yeouido_map'
NAME = 'mission_map_1x2'
SOLVER = ROOT/'tools/iot_uav_deployment_v3_global_iot/deployment.py'
SIDE = 2.599607547169811
FIGSIZE = (6.449215094339622, 3.099607547169811)
SEED = 48
GEOMETRY_REVISION = 'park_multipolygons_with_holes_v2'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_solver():
    spec = importlib.util.spec_from_file_location('yeouido_placement_solver', SOLVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MapGeometry:
    def __init__(self):
        self.imagery = read(CACHE/'satellite_1x2.json')
        assert self.imagery['image_sha256'] == sha(CACHE/'satellite_1x2.jpg')
        self.lat, self.lon = self.imagery['center_latlon']
        self.scale = np.cos(np.radians(self.lat))
        self.center = self.mercator(np.array([[self.lat, self.lon]]))[0]
        extent = self.imagery['export']['extent']
        actual = np.array([extent[k] for k in ['xmin','ymin','xmax','ymax']])
        assert np.allclose(actual, self.imagery['requested_bbox'], rtol=0, atol=.001)
        assert np.allclose((actual[[0,1]]-self.center)*self.scale, [-1000,-1000])
        assert np.allclose((actual[[2,3]]-self.center)*self.scale, [1000,1000])
        self.osm = read(CACHE/'osm_snapshot.json')
        self.parks, self.park_regions, self.water, self.water_holes = [], [], [], []
        for e in self.osm['elements']:
            t = e.get('tags', {})
            park = t.get('leisure') in ['park', 'garden', 'pitch', 'playground']
            water = t.get('natural') == 'water'
            if not (park or water):
                continue
            outer, holes = [], []
            for ring, inner in self.rings(e):
                xy = self.xy(ring)
                if water:
                    (self.water_holes if inner else self.water).append(xy)
                else:
                    (holes if inner else outer).append(xy)
                    if not inner:
                        self.parks.append(xy)
            if park and outer:
                self.park_regions.append((outer, holes))
        assert self.parks and self.water

    @staticmethod
    def mercator(latlon):
        lat, lon = np.asarray(latlon).T
        return 6378137.*np.column_stack([np.radians(lon), np.log(np.tan(np.pi/4+np.radians(lat)/2))])

    def xy(self, geometry):
        values = [[g['lat'], g['lon']] for g in geometry]
        return (self.mercator(values)-self.center)*self.scale

    def latlon(self, xy):
        x, y = (np.asarray(xy)/self.scale+self.center).T/6378137.
        return np.column_stack([np.degrees(2*np.arctan(np.exp(y))-np.pi/2), np.degrees(x)])

    @staticmethod
    def rings(element):
        if element['type'] == 'way':
            g = element.get('geometry', [])
            return [(g, False)] if len(g)>3 and g[0]==g[-1] else []
        rings = []
        for role in ['outer', 'inner']:
            parts = [list(m['geometry']) for m in element.get('members', [])
                     if m.get('role', 'outer')==role and m.get('geometry')]
            while parts:
                ring = parts.pop(0)
                while ring[-1] != ring[0]:
                    found = False
                    for i, part in enumerate(parts):
                        if ring[-1] == part[0]:
                            ring += part[1:]; parts.pop(i); found=True; break
                        if ring[-1] == part[-1]:
                            ring += part[-2::-1]; parts.pop(i); found=True; break
                    if not found:
                        break
                if len(ring)>3 and ring[0]==ring[-1]:
                    rings.append((ring, role=='inner'))
        return rings

    @staticmethod
    def inside(points, polygons):
        answer = np.zeros(len(points), dtype=bool)
        for polygon in polygons:
            answer |= PolygonPath(polygon).contains_points(points, radius=.1)
        return answer

    def on_water(self, points):
        return self.inside(points, self.water) & ~self.inside(points, self.water_holes)

    def in_parks(self, points):
        answer = np.zeros(len(points), dtype=bool)
        for outer, holes in self.park_regions:
            answer |= self.inside(points, outer) & ~self.inside(points, holes)
        return answer


def prepare(geo):
    path = OUT/'deployment.json'
    if path.exists():
        previous = read(path)
        assert previous['osm_sha256'] == sha(CACHE/'osm_snapshot.json')
        assert previous['center_latlon'] == [geo.lat, geo.lon]
        if previous.get('geometry_revision') == GEOMETRY_REVISION:
            return previous
    solver = load_solver()
    rng = np.random.default_rng(SEED)
    candidates, ids = [], []
    for e in geo.osm['elements']:
        t, g = e.get('tags', {}), e.get('geometry', [])
        if e['type']!='way' or t.get('highway') not in ['footway', 'path', 'cycleway', 'pedestrian', 'service'] or len(g)<2:
            continue
        if t.get('bridge') in ['yes', 'viaduct'] or t.get('tunnel')=='yes' or t.get('access') in ['no', 'private']:
            continue
        xy = geo.xy(g)
        for a, b in zip(xy, xy[1:]):
            if np.any(np.maximum(a,b)<-900) or np.any(np.minimum(a,b)>900):
                continue
            points = np.linspace(a, b, max(2, int(np.ceil(np.linalg.norm(b-a)/8))+1))
            points = points[np.all(abs(points)<=900, axis=1)]
            candidates.extend(points); ids.extend([e['id']]*len(points))
    candidates, ids = np.asarray(candidates), np.asarray(ids)
    valid = geo.in_parks(candidates) & ~geo.on_water(candidates)
    candidates, ids = candidates[valid], ids[valid]
    _, unique = np.unique(np.round(candidates, 1), axis=0, return_index=True)
    candidates, ids = candidates[unique], ids[unique]
    cells = np.clip(((candidates+900)/180).astype(int), 0, 9)
    cell_ids = cells[:,0]*10+cells[:,1]
    pools = {int(k): list(rng.permutation(np.flatnonzero(cell_ids==k))) for k in np.unique(cell_ids)}
    chosen = []
    while len(chosen)<200:
        added = 0
        for cell in rng.permutation(list(pools)):
            while pools[cell]:
                i = pools[cell].pop()
                if not chosen or np.linalg.norm(candidates[chosen]-candidates[i], axis=1).min()>=18:
                    chosen.append(i); added+=1; break
            if len(chosen)==200:
                break
        assert added, 'Insufficient park-path samples with 18 m separation'
    ground = candidates[chosen]
    bits = rng.choice([250000., 500000., 750000., 1000000.], 200)
    weights, relay = solver.coefficients(bits)
    # Farthest-point initialization depends only on the fixed device distribution.
    centers = [ground[np.argmin(ground[:,1])]]
    for _ in range(4):
        d2 = np.min(np.sum((ground[:,None]-np.asarray(centers)[None])**2, axis=-1), axis=1)
        centers.append(ground[d2.argmax()])
    centers = np.asarray(centers); slots = np.repeat(np.arange(5), 40); previous=None
    for iteration in range(100):
        cost = weights[:,None]*np.sum((ground[:,None]-centers[None])**2, axis=-1)
        rows, cols = linear_sum_assignment(cost[:,slots])
        assignment = np.empty(200, int); assignment[rows]=slots[cols]
        centers = np.array([np.average(ground[assignment==k],axis=0,weights=weights[assignment==k]) for k in range(5)])
        if previous is not None and np.array_equal(assignment, previous):
            break
        previous=assignment.copy()
    routes = [(0,)+p for p in itertools.permutations(range(1,5))]
    order = min(routes, key=lambda r: np.linalg.norm(np.roll(centers[list(r)],-1,axis=0)-centers[list(r)],axis=1).sum())
    region_ids = [np.flatnonzero(assignment==k) for k in order]
    devices = np.array([ground[i] for i in region_ids]); payloads=np.array([bits[i] for i in region_ids])
    centers = centers[list(order)]
    # A dispersed initial swarm avoids enforcing a one/two-row formation.
    angle = np.arange(10)*np.pi*(3-np.sqrt(5))
    radius = np.sqrt((np.arange(10)+.5)/10)*130
    offsets = np.column_stack([np.cos(angle),np.sin(angle)])*radius[:,None]
    offsets -= offsets.mean(axis=0)
    initial = centers[:,None]+offsets
    original_constraints = solver.constraints
    def bounded_constraints(x, *args):
        return np.concatenate([original_constraints(x, *args), ((900-abs(x))/100).ravel()])
    solver.constraints = bounded_constraints
    assert np.max(abs(initial)) < 900
    write(OUT/'registered_inputs.json', dict(seed=SEED, center_latlon=[geo.lat, geo.lon],
        mission_extent_xy_m=[-900,900,-900,900], view_extent_xy_m=[-1000,1000,-1000,1000],
        devices_xy_m=devices.tolist(), payload_bits=payloads.tolist(), initial_uav_xy_m=initial.tolist(),
        region_global_device_ids=[i.tolist() for i in region_ids],
        global_device_osm_way_ids=ids[chosen].tolist(), occupied_10x10_cells=len(np.unique(cell_ids[chosen])),
        distribution='200 synthetic sensors stratified on mapped park paths before deriving five balanced collection zones',
        source_sha256=sha(Path(__file__)), solver_sha256=sha(SOLVER)))
    print('Optimizing 10 UAVs at five Yeouido collection zones...', flush=True)
    positions, association, heads, optimization = solver.optimize(initial, devices, payloads)
    assert any(row['position_step_accepted'] for row in optimization['trace'])
    data = dict(location='Yeouido, Seoul, South Korea', center_latlon=[geo.lat,geo.lon],
        geometry_revision=GEOMETRY_REVISION,
        mission_side_m=1800, displayed_map_side_m=2000, altitude_m=50,
        uav_count=10, site_count=5, iot_count=200, parameters=solver.PARAMETERS,
        optimizer_initialization='Dispersed phyllotactic swarm, same UAV identity through all sites',
        iot_positions_xy_m=devices.tolist(), iot_payload_bits=payloads.tolist(),
        uav_positions_xy_m=positions.tolist(), association_zero_based=association.tolist(),
        head_zero_based=heads.tolist(), head_kind='nominal_packet_planning_head_not_FL_FU_experiment_result',
        optimization=optimization, osm_sha256=sha(CACHE/'osm_snapshot.json'),
        semantics='New synthetic geometry on real Yeouido map coordinates; not measured flights or sensor deployments. Existing Daejeon FL/FU results are not relabeled as Yeouido results.')
    write(path, data)
    return data


def verify(geo, data):
    solver = load_solver(); p=data['parameters']
    x=np.array(data['uav_positions_xy_m']); devices=np.array(data['iot_positions_xy_m'])
    a=np.array(data['association_zero_based']); bits=np.array(data['iot_payload_bits']); heads=np.array(data['head_zero_based'])
    assert x.shape==(5,10,2) and devices.shape==(5,40,2)
    assert np.max(abs(x))<=900+1e-5 and np.max(abs(devices))<=900
    assert not geo.on_water(devices.reshape(-1,2)).any()
    assert geo.in_parks(devices.reshape(-1,2)).all()
    minimum=float('inf')
    for start,end in zip(x,np.roll(x,-1,axis=0)):
        for f in np.linspace(0,1,2001):
            pts=(1-f)*start+f*end; distances=np.linalg.norm(pts[:,None]-pts[None],axis=-1)
            minimum=min(minimum,float(distances[~np.eye(10,dtype=bool)].min()))
    assert minimum>=22-1e-5
    peak_power=0.
    for index in range(5):
        assert np.array_equal(np.bincount(a[index],minlength=10),np.full(10,4))
        d2=np.sum((x[index,a[index]]-devices[index])**2,axis=1)+50**2
        gain=1e-6/d2; noise=solver.noise(1e5)
        power=np.expm1(np.log(2)*bits[index]/(2.5*1e5))*noise/gain
        assert power.max()<=.1+1e-9
        assert np.allclose(2.5*1e5*np.log2(1+power*gain/noise),bits[index])
        distances=np.linalg.norm(x[index,:,None]-x[index,None,:],axis=-1)
        assert int(np.sum(distances**2,axis=0).argmin())==int(heads[index])
        peak_power=max(peak_power,float(power.max()))
    result=dict(passed=True, no_FL_FU_retraining=True, iot_not_on_water=True,
        sensors_on_mapped_park_paths=True, satellite_and_overlay_registered=True, all_200_devices_retained=True,
        ten_uav_identities_preserved=True, five_collection_zones=True,
        no_duplicate_UAVs_at_a_snapshot=True, nominal_head_candidates_checked=50,
        minimum_continuous_separation_m=data['optimization']['minimum_flight_separation_m'],
        independently_sampled_minimum_separation_m=minimum, peak_iot_power_W=peak_power,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),SOLVER,CACHE/'osm_snapshot.json',CACHE/'satellite_1x2.jpg']})
    write(OUT/'verification.json',result)
    return result


def render(geo, data):
    import proposed_pks_report as reference
    reference.setup()
    plt.rcParams.update({'savefig.bbox':None, 'hatch.linewidth':.55})
    fig=plt.figure(figsize=FIGSIZE,facecolor='white')
    axes=[fig.add_axes([(.47+i*(SIDE+.59))/FIGSIZE[0],.33/FIGSIZE[1],SIDE/FIGSIZE[0],SIDE/FIGSIZE[1]]) for i in range(2)]
    limits=[-1000,1000,-1000,1000]
    axes[0].imshow(Image.open(CACHE/'satellite_1x2.jpg'),extent=limits,origin='upper',zorder=0)
    ax=axes[1];ax.set_facecolor('#f8faf7')
    for polygon in geo.water:
        ax.add_patch(Polygon(polygon,facecolor='#cbdfe7',edgecolor='#a9c5cf',lw=.25,zorder=1))
    for polygon in geo.water_holes:
        ax.add_patch(Polygon(polygon,facecolor='#f8faf7',edgecolor='none',zorder=1.1))
    # Render large park relations with their excluded urban interiors first.
    def region_area(region):
        return sum(abs(np.dot(p[:,0],np.roll(p[:,1],1))-np.dot(p[:,1],np.roll(p[:,0],1)))/2 for p in region[0])
    for outer,holes in sorted(geo.park_regions,key=region_area,reverse=True):
        for polygon in outer:
            ax.add_patch(Polygon(polygon,facecolor='#e2ebda',edgecolor='#c7d5bd',lw=.25,zorder=1.2))
        for polygon in holes:
            ax.add_patch(Polygon(polygon,facecolor='#f8faf7',edgecolor='#c7d5bd',lw=.25,zorder=1.2))
    for e in geo.osm['elements']:
        t,g=e.get('tags',{}),e.get('geometry',[])
        if e['type']!='way' or len(g)<2 or 'highway' not in t:
            continue
        xy=geo.xy(g)
        if np.any(xy.max(axis=0)<-1000) or np.any(xy.min(axis=0)>1000):
            continue
        width={'primary':.85,'secondary':.7,'tertiary':.55,'residential':.4,'service':.28}.get(t['highway'],.18)
        ax.plot(*xy.T,color='#84919a',lw=width+.25,alpha=.85,zorder=2)
        ax.plot(*xy.T,color='white',lw=width,zorder=2)
    positions=np.array(data['uav_positions_xy_m']);devices=np.array(data['iot_positions_xy_m'])
    heads=np.array(data['head_zero_based']);centers=positions.mean(axis=1)
    scale=.576
    for uav in range(10):
        path=np.vstack([positions[:,uav],positions[0,uav]])
        ax.plot(*path.T,color='#28608b',alpha=.25,lw=.7*scale,zorder=3.5)
    for site,(points,center) in enumerate(zip(positions,centers)):
        values,vectors=np.linalg.eigh(np.cov((points-center).T));values=np.maximum(values,1.)
        local=(points-center)@vectors;multiplier=np.sqrt(np.max(np.sum(local**2/values,axis=1)))
        radii=np.sqrt(values)*multiplier+25;angle=np.degrees(np.arctan2(vectors[1,1],vectors[0,1]))
        ax.add_patch(Ellipse(center,width=2*radii[1],height=2*radii[0],angle=angle,
            facecolor=(.16,.43,.72,.18),edgecolor=(.12,.35,.62,.88),lw=1.25*scale,ls=(0,(4,2.5)),zorder=2.8))
        ax.scatter(*devices[site].T,s=9*scale**2,facecolors='#f3c77f',edgecolors='#ae9265',alpha=.75,lw=.25*scale,zorder=5)
        ax.scatter(*points.T,marker='*',s=88*scale**2,facecolors='#286bb3',edgecolors='#12365a',alpha=.9,lw=.6*scale,zorder=8)
        ax.scatter(*points[heads[site]],marker='*',s=125*scale**2,facecolors='#df6249',edgecolors='#30201b',alpha=.95,lw=.65*scale,zorder=9)
        label_y=np.clip(points[:,1].max()+65,-900,900)
        ax.text(center[0],label_y,str(site+1),ha='center',va='center',fontsize=12,color='#234f72',zorder=12,
                bbox=dict(boxstyle='circle,pad=.2',facecolor='white',edgecolor='#6e92ac',lw=.65*scale,alpha=.9))
    route=np.vstack([centers,centers[0]])
    ax.plot(*route.T,color='white',lw=4.2*scale,zorder=5.5)
    ax.plot(*route.T,color='#174c77',lw=1.4*scale,ls=(0,(4,3)),zorder=6)
    for a,b in zip(route,route[1:]):
        arrow=ax.annotate('',xy=a+.67*(b-a),xytext=a+.33*(b-a),zorder=7,
            arrowprops=dict(arrowstyle='-|>',color='#174c77',lw=1.6*scale,mutation_scale=17*scale))
        arrow.arrow_patch.set_path_effects([pe.withStroke(linewidth=3.4*scale,foreground='white')])
    for axis in axes:
        axis.set_xlim(-1000,1000);axis.set_ylim(-1000,1000);axis.set_aspect('equal')
        for dim in ['x','y']:
            getattr(axis,dim+'axis').set_major_locator(FixedLocator(np.arange(-1000,1001,500)))
            getattr(axis,dim+'axis').set_major_formatter(FuncFormatter(lambda v,_: f'{v+1000:.0f}'))
        axis.tick_params(labelsize=12,direction='out',pad=3,length=3,width=.9)
        for spine in axis.spines.values():spine.set_linewidth(.9)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for axis in axes:
        for text in [*axis.get_xticklabels(),*axis.get_yticklabels(),*axis.texts]:
            if not text.get_text():continue
            box=text.get_window_extent(renderer)
            assert box.x0>=0 and box.y0>=0 and box.x1<=fig.bbox.width and box.y1<=fig.bbox.height
    files=[]
    for ext in ['pdf','png','svg']:
        path=OUT/(NAME+'.'+ext);fig.savefig(path,dpi=350,bbox_inches=None)
        if ext=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        files.append(dict(file=path.name,sha256=sha(path)))
    plt.close(fig)
    metadata=dict(location=data['location'],layout='1 x 2',figure_size_inches=list(FIGSIZE),map_panel_size_inches=[SIDE,SIDE],
        tick_fontsize_pt=12,visit_fontsize_pt=12,tick_values=[0,500,1000,1500,2000],mission_side_m=1800,displayed_map_side_m=2000,
        view_padding_m=100,center_latlon=[geo.lat,geo.lon],imagery=geo.imagery,
        osm_attribution='© OpenStreetMap contributors',osm_copyright_url='https://www.openstreetmap.org/copyright',
        uav_count=10,mission_stops=5,iot_count=200,head_semantics=data['head_kind'],
        deployment_source=str((OUT/'deployment.json').relative_to(ROOT)),deployment_sha256=sha(OUT/'deployment.json'),
        cluster_semantics='Translucent enclosure of a planned ten-UAV collection snapshot; not radio coverage',
        path_semantics='Ten identity-preserving piecewise-linear planned paths across five stops and return',
        scope=data['semantics'],files=files)
    write(OUT/'manifest.json',metadata)
    return metadata


def install(dest):
    """Install the published scenario in a rebuilt gallery without retraining."""
    metadata=read(OUT/'manifest.json')
    for ext in ['pdf','png','svg']:
        shutil.copyfile(OUT/(NAME+'.'+ext),dest/('06_mission_map_1x2.'+ext))
    audit=read(OUT/'previous_daejeon/manifest.json')['compact_variant']
    audit.update(location_revision='yeouido_v1',geometry_and_snapshot_heads_unchanged=False,
        geometry_keys_verified=[],new_geometry_verified=True,
        original_snapshot_run=None,snapshot_scope=metadata['scope'],
        geometry_checks=str((OUT/'verification.json').relative_to(ROOT)),
        source_sha256=read(OUT/'verification.json')['source_sha256'])
    write(dest/'data/mission_map.json',dict(metadata,compact_variant=audit))
    figure=dict(name='06_mission_map_1x2',title='Yeouido satellite imagery and planned IoT–UAV deployment',
        kind='map',axes=[],linear_scale=.576,physical_size_inches=list(FIGSIZE),location=metadata['location'],
        scope='New Yeouido geometry illustration; the FL/FU performance tables and curves retain their Daejeon conditions.',
        files=[dict(file='06_'+f['file'],sha256=f['sha256']) for f in metadata['files']])
    return figure,audit


def publish(metadata):
    base=ROOT/'final_final_charts';dest=base/'downlink_bandwidth';alias=base/'figures'
    archive=OUT/'previous_daejeon'
    if not archive.exists():
        archive.mkdir(parents=True)
        for ext in ['pdf','png','svg']:
            shutil.copyfile(dest/('06_mission_map_1x2.'+ext),archive/(NAME+'.'+ext))
        shutil.copyfile(dest/'data/mission_map.json',archive/'manifest.json')
    figure,map_audit=install(dest)
    for ext in ['pdf','png','svg']:
        shutil.copyfile(OUT/(NAME+'.'+ext),alias/(NAME+'.'+ext))
    audit_path=dest/'data/audit.json';audit=read(audit_path);audit['map']=map_audit;write(audit_path,audit)
    path=dest/'data/manifest.json';document=read(path)
    old=next(f for f in document['figures'] if f['name']=='06_mission_map_1x2')
    old.update(figure)
    document['source_sha256'][str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    compact=Path(__file__).with_name('compact_mission_map.py')
    document['source_sha256'][str(compact.relative_to(ROOT))]=sha(compact)
    # The preceding rank/FU layout pass changed only those category plots;
    # carry its recorded current typography source into this gallery manifest.
    typography_key='tools/uav_convergence_v2/paper_typography_v1.py'
    typography_source=read(base/'data_typography_v1.json')['source_sha256'][typography_key]
    assert typography_source==sha(ROOT/typography_key)
    if document['source_sha256'].get(typography_key)!=typography_source:
        document.setdefault('previous_plot_source_sha256',{})[typography_key]=document['source_sha256'][typography_key]
        document['source_sha256'][typography_key]=typography_source
    write(path,document)
    caption=('Simulation scenario in Yeouido, Seoul, South Korea: satellite imagery (left) and synthetic IoT deployment '
        'with planned UAV positions and trajectories (right). Circles denote IoT devices, blue stars denote UAVs, '
        'and orange stars denote nominal planning heads. Numbers indicate five sequential collection stops of the same ten UAVs. '
        'The 1800 × 1800 m mission region has 100 m map padding on each side. The geometry is newly planned for Yeouido; '
        'the existing FL/FU performance results use the previous Daejeon scenario. Imagery: '+str(metadata['imagery']['copyright'])+
        '. Map data: © OpenStreetMap contributors.\n')
    (OUT/'caption.txt').write_text(caption);(alias/(NAME+'_caption.txt')).write_text(caption)
    (OUT/'README.md').write_text('# Yeouido mission map\n\n'
        '[PDF](mission_map_1x2.pdf) · [PNG](mission_map_1x2.png) · [SVG](mission_map_1x2.svg)\n\n'+caption+'\n'
        'The 200 synthetic sensors are stratified on OSM park paths before deriving five balanced 40-device collection zones. '
        'UAV positions, associations and nominal packet heads are then optimized using the existing placement solver. '
        'Heads are planning outputs, not replayed or claimed FL/FU AirComp head decisions. Straight-line transitions are '
        'checked for 22 m inter-UAV separation, but building clearance, terrain and flight permissions are not modeled. '
        'This is a simulation map, not an operational flight plan. No model-quality or resource results have been rerun.\n\n'
        'The old map is retained in `previous_daejeon/`. The new paper alias is `../figures/mission_map_1x2.pdf`. '
        'Both square panels retain the prior physical dimensions and 12 pt ticks.\n\n'
        'OSM source and license: [OpenStreetMap contributors](https://www.openstreetmap.org/copyright), ODbL. '
        'Cached imagery metadata and OSM query are in `data/korea_mission_yeouido_v1` at repository root.\n\n'
        'Reproduce: `.venv/bin/python tools/uav_convergence_v2/yeouido_mission_map_v1.py --publish`.\n')
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Yeouido mission map</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:0 20px}img{width:100%}</style>'
        '<h1>Yeouido, Seoul</h1><a href="mission_map_1x2.pdf"><img src="mission_map_1x2.png"></a>'
        '<p><a href="README.md">Scenario and sources</a> · <a href="previous_daejeon/mission_map_1x2.pdf">Previous Daejeon map</a></p>')
    for target in [alias/'README.md',dest/'README.md']:
        text=target.read_text();marker='<!-- yeouido-map -->';text=text.split(marker)[0].rstrip()
        target.write_text(text+'\n\n'+marker+'\n\nThe current mission map now illustrates Yeouido, Seoul. '
            'See [the map folder](../yeouido_map/README.md) for the new geometry and sources. '
            'Existing FL/FU result tables and curves remain based on Daejeon; they have not been relabeled or rerun.\n')
    tgcn=dest/'data/tgcn_manifest.json';document=read(tgcn)
    for key in ['original_figures','original_figures_before_typography']:
        if key=='original_figures':
            for f in document.get(key,{}):
                if f.startswith('06_mission_map_1x2.'):
                    document[key][f]=sha(dest/f)
    write(tgcn,document)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--publish',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    geo=MapGeometry();data=prepare(geo);checks=verify(geo,data);metadata=render(geo,data)
    if args.publish:publish(metadata)
    print(json.dumps(dict(passed=checks['passed'],location=data['location'],files=metadata['files'],optimization=data['optimization']),indent=2),flush=True)


if __name__=='__main__':main()
