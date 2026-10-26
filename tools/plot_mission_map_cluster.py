"""Translucent five-snapshot clusters with ten recorded per-UAV paths.

Rendering only: preserves all global IoT, optimized UAV and head coordinates.
"""
from pathlib import Path
import importlib.util
import sys

path=Path(__file__).with_name('plot_mission_map_reference.py')
if '--global-iot' not in sys.argv:sys.argv.append('--global-iot')
if '--mode' not in sys.argv:sys.argv.extend(['--mode','iot'])
spec=importlib.util.spec_from_file_location('cluster_reference_style',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
source=module.source


def replace(old,new,count=1):
    global source
    assert source.count(old)==count,(old,source.count(old))
    source=source.replace(old,new,count)


replace('from matplotlib.patches import Polygon','from matplotlib.patches import Polygon, Ellipse')
replace('figsize=(9.4, 5.5)','figsize=(9.4, 4.85)')
replace('fig.subplots_adjust(left=.065, right=.97, top=.87, bottom=.16, wspace=.22)',
        'fig.subplots_adjust(left=.055, right=.975, top=.98, bottom=.08, wspace=.12)')
replace("s=12,facecolors='#f3c77f',edgecolors='#ae9265',\n                       linewidths=.35", "s=9,facecolors='#f3c77f',edgecolors='#ae9265',alpha=.75,\n                       linewidths=.25")
replace("marker='*',s=100,facecolors='#3a78b5',edgecolors='#14202a',\n                           linewidths=.65", "marker='*',s=70,facecolors='#3a78b5',edgecolors='#14202a',alpha=.65,\n                           linewidths=.5")
replace("                ax.scatter(*footprint[head],marker='o',s=330,facecolors='none',edgecolors='#c14a38',linewidths=.8,linestyle=(0,(4,3)),zorder=8)\n",'')
replace("marker='*',s=150,facecolors='#df6249',edgecolors='#30201b',\n                           linewidths=.7", "marker='*',s=105,facecolors='#df6249',edgecolors='#30201b',alpha=.9,\n                           linewidths=.55")
replace("            panel.set_xlabel('X Coordinate (m)', fontsize=11, labelpad=4)\n            panel.set_ylabel('Y Coordinate (m)', fontsize=11, labelpad=4)\n",'')
old="""        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.525,.98),ncol=3,frameon=False,
                   fontsize=12,handlelength=1.1,handletextpad=.5,columnspacing=1.8)
        for panel,label in zip(axes,['(a) Satellite imagery','(b) IoT–UAV deployment']):
            box=panel.get_position()
            fig.text((box.x0+box.x1)/2,.035,label,ha='center',va='bottom',fontsize=11,color='black')"""
replace(old,'')
insert="""    if deployment:
        for footprint,center in zip(footprints,centers):
            values,vectors=np.linalg.eigh(np.cov((footprint-center).T))
            values=np.maximum(values,1.)
            local=(footprint-center)@vectors
            multiplier=np.sqrt(np.max(np.sum(local**2/values,axis=1)))
            radii=np.sqrt(values)*multiplier+25*scale
            angle=np.degrees(np.arctan2(vectors[1,1],vectors[0,1]))
            ax.add_patch(Ellipse(center,width=2*radii[1],height=2*radii[0],angle=angle,
                                 facecolor=(.22,.46,.68,.075),edgecolor=(.22,.46,.68,.35),
                                 linewidth=.8,linestyle=(0,(4,3)),zorder=2.8))
        for uav in range(10):
            waypoints=np.array([footprint[uav] for footprint in footprints])
            waypoints=np.vstack([waypoints,waypoints[:1]])
            ax.plot(*waypoints.T,color='#28608b',alpha=.23,lw=.65,zorder=3.5)
        for step,(footprint,center) in enumerate(zip(footprints,centers),1):
            label_y=np.clip(footprint[:,1].max()+45*scale,limits[2]+50*scale,limits[3]-50*scale)
            ax.text(center[0],label_y,str(step),ha='center',va='center',fontsize=10,
                    color='#234f72',zorder=12,
                    bbox=dict(boxstyle='circle,pad=.2',facecolor='white',edgecolor='#6e92ac',linewidth=.65,alpha=.9))
"""
replace('    for number, (footprint, center) in enumerate(zip(display_footprints, centers), 1):',
        insert+'    for number, (footprint, center) in enumerate(zip(display_footprints, centers), 1):')
replace("    for extension in ['png', 'pdf', 'svg']:",
        "    stem=stem.with_name(stem.name+'_cluster_motion')\n    assert sorted(t.get_text() for panel in axes for t in panel.texts if t.get_text())==['1','2','3','4','5']\n    for extension in ['png', 'pdf', 'svg']:")
replace("    stem.with_suffix('.json').write_text",
        "    metadata.update(visual_variant='global_iot_five_snapshot_cluster_motion',figure_size_inches=[9.4,4.85],head_ring='Not drawn',cluster_semantics='Translucent visual enclosure of each recorded ten-UAV snapshot; not radio coverage',path_semantics='Ten separate identity-preserving piecewise-linear UAV trajectories through the five snapshots and return',snapshot_visit_order=[1,2,3,4,5],per_uav_paths_epsg3857=[np.vstack([[f[u] for f in footprints],footprints[0][u]]).tolist() for u in range(10)],plot_style='Translucent clusters, faint individual UAV paths, five numeric visit labels and numeric endpoint ticks only; no textual legend or axis/panel captions')\n    stem.with_suffix('.json').write_text")

if __name__=='__main__':
    exec(compile(source,str(module.source_path),'exec'),dict(__name__='__main__',__file__=str(module.source_path)))
