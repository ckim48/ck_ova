"""Larger paper-style visual variant; reuse the registered map's exact geometry.

Only plot styling/output names are replaced. No model, head or position changes.
Run: python tools/plot_mission_map_reference.py --mode iot
"""
from pathlib import Path
import sys

source_path=Path(__file__).with_name('plot_mission_map.py')
source=source_path.read_text()


def replace(old,new,count=1):
    global source
    assert source.count(old)==count,(old,source.count(old))
    source=source.replace(old,new,count)


replace("fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.54 if clean_snapshots else 2.86))", "fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))")
replace("fig.subplots_adjust(left=.012, right=.988, top=.985,\n                        bottom=.015 if clean_snapshots else .12, wspace=.045)",
        "fig.subplots_adjust(left=.065, right=.985, top=.84, bottom=.20, wspace=.22)")
replace("'font.size': 8", "'font.size': 10")
replace("color='#b8c1ba', lw=road_width + .5", "color='#84919a', lw=road_width + .5")
replace("color='#5d9d98',lw=.3,alpha=.5", "color='#487997',lw=.4,alpha=.45")
replace("s=5,facecolors='#91aaa8',edgecolors='white',\n                       linewidths=.25", "s=12,facecolors='#f5a623',edgecolors='#352a1b',\n                       linewidths=.35")
replace("marker='*',s=65,facecolors='#278c83',edgecolors='white',\n                           linewidths=.55", "marker='*',s=100,facecolors='#3a78b5',edgecolors='#14202a',\n                           linewidths=.65")
replace("ax.scatter(*footprint[head],marker='*',s=110,facecolors='#e6a13d',edgecolors='#895608',\n                           linewidths=.6,zorder=9)",
        "ax.scatter(*footprint[head],marker='o',s=330,facecolors='none',edgecolors='#c14a38',linewidths=.8,linestyle=(0,(4,3)),zorder=8)\n                ax.scatter(*footprint[head],marker='*',s=150,facecolors='#df6249',edgecolors='#30201b',\n                           linewidths=.7,zorder=9)")
replace("route_color = '#163a50'", "route_color = '#245c82'")
replace("if clean_snapshots:\n            continue", "if clean_snapshots:\n            from matplotlib.ticker import FixedLocator, FuncFormatter\n            tick_x = np.arange(0, (limits[1]-limits[0])/scale, 300.)\n            tick_y = np.arange(0, (limits[3]-limits[2])/scale, 300.)\n            panel.xaxis.set_major_locator(FixedLocator(limits[0]+tick_x*scale))\n            panel.yaxis.set_major_locator(FixedLocator(limits[2]+tick_y*scale))\n            panel.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{(value-limits[0])/scale:.0f}'))\n            panel.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{(value-limits[2])/scale:.0f}'))\n            panel.set_xlabel('X Coordinate (m)', fontsize=11, labelpad=4)\n            panel.set_ylabel('Y Coordinate (m)', fontsize=11, labelpad=4)\n            panel.tick_params(axis='both',labelsize=10,direction='in',length=3,width=.65,pad=3)\n            continue")
old="""        handles=[Line2D([],[],marker='o',linestyle='none',markersize=3.3,color='#91aaa8',label='IoT'),
                 Line2D([],[],marker='*',linestyle='none',markersize=6.5,color='#278c83',label='UAV'),
                 Line2D([],[],marker='*',linestyle='none',markersize=7,color='#e6a13d',markeredgecolor='#895608',markeredgewidth=.5,label='Head')]
        ax.legend(handles=handles,loc='upper right',ncol=3,frameon=True,
                  facecolor='white',edgecolor='none',framealpha=.95,
                  fontsize=6.5,handlelength=.8,handletextpad=.45,columnspacing=1.2,borderpad=.6)"""
new="""        handles=[Line2D([],[],marker='*',linestyle='none',markersize=12,color='#df6249',markeredgecolor='#30201b',markeredgewidth=.7,label='Head'),
                 Line2D([],[],marker='*',linestyle='none',markersize=11,color='#3a78b5',markeredgecolor='#14202a',markeredgewidth=.65,label='UAV'),
                 Line2D([],[],marker='o',linestyle='none',markersize=5.5,color='#f5a623',markeredgecolor='#352a1b',markeredgewidth=.5,label='IoT')]
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.525,.98),ncol=3,frameon=False,
                   fontsize=12,handlelength=1.1,handletextpad=.5,columnspacing=1.8)
        for panel,label in zip(axes,['(a) Satellite imagery','(b) IoT–UAV deployment']):
            box=panel.get_position()
            fig.text((box.x0+box.x1)/2,.035,label,ha='center',va='bottom',fontsize=11,color='black')"""
replace(old,new)
replace("    for extension in ['png', 'pdf', 'svg']:", "    stem=stem.with_name(stem.name+'_reference_large')\n    for extension in ['png', 'pdf', 'svg']:")
replace("    stem.with_suffix('.json').write_text", "    metadata.update(visual_variant='reference_large',figure_size_inches=[9.4,4.2],axis_units='Local ground metres from displayed lower-left corner, using the mission latitude scale',head_ring='Marker emphasis only, not a communication coverage radius',palette=dict(iot='#f5a623',uav='#3a78b5',head='#df6249'),plot_style='Larger serif paper layout, black marker outlines, shared top legend and metric axes')\n    stem.with_suffix('.json').write_text")

if __name__=='__main__':
    if '--mode' not in sys.argv:sys.argv.extend(['--mode','iot'])
    exec(compile(source,str(source_path),'exec'),dict(__name__='__main__',__file__=str(source_path)))
