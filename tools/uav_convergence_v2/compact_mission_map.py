"""Reduce the previous 90%-size map by another 20%, retaining vector content."""
from pathlib import Path
import re
import subprocess

import downlink_bandwidth_v1 as study

ROOT, read, write, sha = study.ROOT, study.read, study.write, study.sha
SOURCE = ROOT / 'results/paper_figures_20261026/mission_map_1x2'
PREVIOUS_SCALE = .9
RELATIVE_SCALE = .8
SCALE = PREVIOUS_SCALE * RELATIVE_SCALE


def pdf_size(path):
    info = subprocess.check_output(['pdfinfo', str(path)], text=True)
    match = re.search(r'Page size:\s+([\d.]+) x ([\d.]+) pts', info)
    assert match, info
    return [float(match[1]), float(match[2])]


def render(dest):
    dest.mkdir(parents=True, exist_ok=True)
    name = '06_mission_map_1x2'
    target = dest / name
    old = read(SOURCE.with_suffix('.json'))
    before = pdf_size(SOURCE.with_suffix('.pdf'))
    after = [value * SCALE for value in before]
    # pdfwrite retains vector paths/text; scale the whole page, including fonts.
    subprocess.run(['gs', '-q', '-dSAFER', '-dBATCH', '-dNOPAUSE', '-sDEVICE=pdfwrite',
                    '-dCompatibilityLevel=1.4', '-dFIXEDMEDIA', '-dPDFFitPage',
                    '-dAutoRotatePages=/None', f'-dDEVICEWIDTHPOINTS={after[0]}',
                    f'-dDEVICEHEIGHTPOINTS={after[1]}',
                    '-sOutputFile=' + str(target.with_suffix('.pdf')), '-f',
                    str(SOURCE.with_suffix('.pdf'))], check=True)
    rendered_size = pdf_size(target.with_suffix('.pdf'))
    # Ghostscript writes the PDF MediaBox to two decimal places (points).
    assert all(abs(a-b) <= .0051 for a, b in zip(rendered_size, after))
    # Keep every original SVG element and its viewBox; only physical dimensions change.
    svg = SOURCE.with_suffix('.svg').read_text()
    match = re.search(r'<svg\b[^>]*\bwidth="([\d.]+)pt" height="([\d.]+)pt"', svg)
    assert match
    width, height = map(float, match.groups())
    old_dimensions = f'width="{match[1]}pt" height="{match[2]}pt"'
    new_dimensions = f'width="{width*SCALE:.8g}pt" height="{height*SCALE:.8g}pt"'
    assert svg.count(old_dimensions) == 1
    target.with_suffix('.svg').write_text(svg.replace(old_dimensions, new_dimensions, 1))
    subprocess.run(['pdftocairo', '-png', '-singlefile', '-r', '450',
                    str(target.with_suffix('.pdf')), str(target)], check=True)
    source_hashes = {str(SOURCE.with_suffix(ext).relative_to(ROOT)): sha(SOURCE.with_suffix(ext))
                     for ext in ['.pdf', '.svg', '.json']}
    audit = dict(passed=True, linear_scale=SCALE, original_page_size_points=before,
                 previous_linear_scale=PREVIOUS_SCALE, scale_relative_to_previous=RELATIVE_SCALE,
                 previous_page_size_points=[v*PREVIOUS_SCALE for v in before],
                 page_size_points=after, rendered_pdf_page_size_points=rendered_size,
                 original_svg_content_preserved=True,
                 geometry_and_snapshot_heads_unchanged=True, vector_pdf=True,
                 source_sha256=source_hashes,
                 original_snapshot_run='results/iot_uav_remoteclip_v4_event_mission/capera_seed0/proposed_fu_guard_cliperase',
                 snapshot_scope='Preserved requested map from the earlier scenario illustration; not claimed as a new P-KS bandwidth-run head selection.')
    metadata = dict(old, compact_variant=audit,
                    figure_size_inches=[v*SCALE for v in old['figure_size_inches']])
    write(dest / 'data/mission_map.json', metadata)
    files = [dict(file=target.with_suffix(ext).name, sha256=sha(target.with_suffix(ext)))
             for ext in ['.png', '.pdf', '.svg']]
    figure = dict(name=name, title='Satellite / IoT–UAV map: 20% smaller than the previous release',
                  kind='map', axes=[], files=files, linear_scale=SCALE,
                  physical_size_inches=metadata['figure_size_inches'])
    return figure, audit
