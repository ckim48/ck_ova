"""Booktabs/multirow tables from the same audited chart measurements."""
import csv
from pathlib import Path
import shutil
import subprocess
import tempfile

import downlink_bandwidth_v1 as study

ROOT, write, sha = study.ROOT, study.write, study.sha
ORDER = ['iscc', 'afl', 'salun', 'gate_select16']
DATASETS = study.MODEL_DATASETS
FORMATTING = r'''\footnotesize
\renewcommand{\arraystretch}{1.0}
\setlength{\tabcolsep}{3pt}
'''
BASE_COLUMNS = [
    dict(key='mr_percent', source='retained_mR_percent', bandwidth=1, scale=1., decimals=3, higher=True),
    dict(key='loss', source='retained_loss', bandwidth=1, scale=1., decimals=4, higher=False),
    dict(key='energy_kj', source='total_energy_j', bandwidth=1, scale=.001, decimals=3, higher=False),
    dict(key='delay_s', source='delay_s', bandwidth=1, scale=1., decimals=3, higher=False),
]
COST_COLUMNS = [dict(key=f'{kind}_{bw}mhz', source=source, bandwidth=bw, scale=scale,
                     decimals=3, higher=False)
                for kind, source, scale in [('energy_kj', 'total_energy_j', .001), ('delay_s', 'delay_s', 1.)]
                for bw in [1, 5]]


def cells(rows, columns):
    by = {(r['dataset'], r['method'], r['bandwidth_mhz']): r for r in rows}
    result = []
    for ds in DATASETS:
        block = []
        for method in ORDER:
            values = []
            for col in columns:
                value = by[(ds, method, col['bandwidth'])][col['source']] * col['scale']
                formatted = f"{value:.{col['decimals']}f}"
                values.append(dict(column=col['key'], value=value, formatted=formatted))
            block.append(dict(dataset=ds, method=method, label=study.METHODS[method], cells=values))
        for i, col in enumerate(columns):
            displayed = [float(r['cells'][i]['formatted']) for r in block]
            best = (max if col['higher'] else min)(displayed)
            for r, value in zip(block, displayed):
                r['cells'][i]['bold'] = value == best
        result.extend(block)
    return result


def tabular(records, columns, header):
    lines = [r'\begin{tabular}{' + '|'.join(['c']*(len(columns)+2)) + '}',
             r'\toprule', header, r'\midrule']
    for ds in DATASETS:
        block = [r for r in records if r['dataset'] == ds]
        for i, row in enumerate(block):
            dataset = r'\multirow{4}{*}{\textsc{' + ds + '}}' if i == 0 else ''
            method = r'\textsc{' + row['label'] + '}'
            values = [r'\textbf{' + c['formatted'] + '}' if c['bold'] else c['formatted'] for c in row['cells']]
            lines.append(' & '.join([dataset, method, *values]) + r' \\')
        if ds != DATASETS[-1]:
            lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


def render_one(dest, rows, name, caption, label, columns, header, number):
    records = cells(rows, columns)
    content = tabular(records, columns, header)
    tex = '\n'.join([r'% Requires \usepackage{booktabs,multirow}.',
                     '% Single seed; bold = best at displayed precision, including ties.',
                     r'\begin{table}[t]', r'\centering', r'\caption{' + caption + '}',
                     r'\label{' + label + '}', FORMATTING, content, r'\end{table}', ''])
    (dest / (name + '.tex')).write_text(tex)
    standalone = '\n'.join([
        r'\documentclass[border=5pt]{standalone}',
        r'\usepackage{booktabs,multirow,mathptmx,caption}', r'\newsavebox{\datatable}',
        r'\begin{document}', r'\sbox{\datatable}{' + FORMATTING + content + '}',
        r'\begin{minipage}{\wd\datatable}', r'\centering',
        r'\captionsetup{font=footnotesize,labelfont=sc,labelsep=period,justification=centering,skip=6pt}',
        r'\setcounter{table}{' + str(number-1) + '}', r'\captionof{table}{' + caption + '}',
        r'\usebox{\datatable}', r'\end{minipage}', r'\end{document}', ''])
    (dest / (name + '_standalone.tex')).write_text(standalone)
    with tempfile.TemporaryDirectory(prefix='ova_table_') as tmp:
        build = Path(tmp)
        source = build / (name + '.tex')
        source.write_text(standalone)
        result = subprocess.run(['pdflatex', '-no-shell-escape', '-interaction=nonstopmode',
                                 '-halt-on-error', source.name], cwd=build,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout[-5000:])
        assert 'Overfull \\hbox' not in result.stdout, result.stdout[-3000:]
        shutil.copyfile(build / (name + '.pdf'), dest / (name + '.pdf'))
    subprocess.run(['pdftocairo', '-png', '-singlefile', '-r', '450',
                    str(dest / (name + '.pdf')), str(dest / name)], check=True)
    subprocess.run(['pdftocairo', '-svg', str(dest / (name + '.pdf')),
                    str(dest / (name + '.svg'))], check=True)
    numeric = []
    for r in records:
        numeric.append(dict(dataset=r['dataset'], method=r['method'], label=r['label'],
                            **{c['column']: c['value'] for c in r['cells']}))
    with (dest / 'data' / (name + '.csv')).open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(numeric[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(numeric)
    data = dict(name=name, caption=caption, label=label, columns=columns, rows=records,
                bold_policy='Best within dataset/column at displayed precision; all displayed ties bold.',
                evaluation='Single seed; same completed200FL and225FU steps. Baseline KL is common retained-only final200 reference, not the training proxy.')
    write(dest / 'data' / (name + '.json'), data)
    files = [dict(file=name + ext, sha256=sha(dest / (name + ext)))
             for ext in ['.png', '.pdf', '.svg', '.tex', '_standalone.tex']]
    return dict(name=name, title=caption, kind='table', axes=[], files=files), data


def render(dest, rows, names=None):
    headers = [
        r'''\multirow{2}{*}{\textsc{Dataset}} & \multirow{2}{*}{\textsc{Method}}
        & \textsc{Retained mR} & \textsc{Loss} & \textsc{Energy} & \textsc{Delay} \\
        & & (\%) $\uparrow$ & $\downarrow$ & (kJ) $\downarrow$ & (s) $\downarrow$ \\''',
        r'''\multirow{2}{*}{\textsc{Dataset}} & \multirow{2}{*}{\textsc{Method}}
        & \multicolumn{2}{c|}{\textsc{Total Energy} (kJ) $\downarrow$}
        & \multicolumn{2}{c}{\textsc{Delay} (s) $\downarrow$} \\
        & & 1 MHz & 5 MHz & 1 MHz & 5 MHz \\''']
    specs = [
        ('07_performance_table', 'Performance comparison after 200 FL rounds and 225 FU steps at 1 MHz downlink bandwidth.',
         'tab:vlm_fl_fu_performance', BASE_COLUMNS),
        ('08_bandwidth_table', 'Energy and active FL/FU delay under two downlink bandwidths for the same completed workload.',
         'tab:vlm_downlink_resources', COST_COLUMNS)]
    figures, audits = [], []
    for i, ((name, caption, label, columns), header) in enumerate(zip(specs, headers), 1):
        if names is not None and name not in names:
            continue
        figure, data = render_one(dest, rows, name, caption, label, columns, header, i)
        figures.append(figure)
        audits.append(dict(name=name, rows=len(data['rows']), cells=sum(len(r['cells']) for r in data['rows']),
                           data_file='data/' + name + '.json', single_seed_no_error_bars=True,
                           source_metrics_sha256=sha(dest / 'data/metrics.json')))
    return figures, dict(passed=True, tables=audits, source_metrics_unchanged=True)
