"""One qualitative figure with both datasets and selected successes/misses."""
import copy
import hashlib
from pathlib import Path
import textwrap

from PIL import Image

from tgcn_success_examples_v2 import excerpt


PICKS = {
    'CapERA': [(2096, 'Urban fire'), (2431, 'Harvesting'),
               (6635, 'Swimming'), (2165, 'Flood')],
    'AirScape': [(7, 'Elephants'), (154, 'Horses and water'),
                 (234, 'Road junction'), (105, 'Pagoda')],
}


def render(report, study):
    root, dest, prior = report.ROOT, report.DEST, report.prior
    read, write, sha = report.read, report.write, report.sha
    methods = report.METHODS
    assert read(study.OUT/'finished.json')['passed']
    protocol = copy.deepcopy(study.register())
    protocol.update(
        display_selection_version='combined_mixed_v3', display_only=True,
        chosen_caption_ids={ds: [cid for cid, _ in picks] for ds, picks in PICKS.items()},
        illustrations='Post-hoc illustrative subset: three Proposed top1 successes and one miss per dataset. Successes are selected from the previous diverse examples; misses are the original median-rank-gap disadvantage selected by the frozen inference protocol. This deliberately chosen 3:1 mix is not a measured success rate or a representative sample. Full original query ranks and aggregate metrics remain unchanged.',
        text_display='Excerpts of original native queries only, with visible ellipses for omitted repeated camera-perspective text or long endings. Inference used the original complete text. Editorial scene labels are not generated captions.',
        layout='One figure, CapERA above AirScape, four rows per dataset, one shared method header. Each method shows its actual top1 retrieved video, visualized by its fourth of eight sampled frames. GT rank below it is the rank of the annotated paired video in the full gallery, not the rank of the displayed top1 result. Green indicates the paired video was retrieved first, red otherwise. Other semantically plausible videos may also match the query.',
    )
    protocol['source_sha256'][str(Path(__file__).relative_to(root))] = sha(Path(__file__))
    original_examples = read(study.OUT/'examples.json')
    all_examples, provenance, layout = [], [], []
    width, height = 8.8, 8.65
    margin, query_width, query_gap = .14, 2.05, .12
    image_start = margin+query_width+query_gap
    pitch = (width-margin-image_start)/5
    centers = [image_start+(i+.5)*pitch for i in range(5)]
    fig = prior.chart.plt.figure(figsize=(width, height), facecolor='white')
    fig.text(.5, 1-.04/height, 'Selected retrieval examples | 650 kJ',
             ha='center', va='top', fontsize=12)
    fig.text(margin/width, 1-.36/height, 'Text query', ha='left', va='top', fontsize=10.5)
    for center, label in zip(centers, ['Ground truth', *methods.values()]):
        fig.text(center/width, 1-.36/height, label, ha='center', va='top', fontsize=10.5)
    row_height, query_artists = .92, []
    for block, ds in enumerate(report.DATASETS):
        data, _ = study.data_for(ds)
        rows = read(study.OUT/(ds.lower()+'_query_ranks.json'))
        by_id = {r['caption_id']: r for r in rows}
        block_top = height-.70-block*3.96
        fig.text(margin/width, block_top/height, f'({chr(97+block)}) {ds}',
                 ha='left', va='top', fontsize=10.5, fontweight='bold')
        line = prior.chart.plt.Line2D([margin/width, 1-margin/width],
                    [(block_top+.04)/height]*2, transform=fig.transFigure,
                    color='#555555', linewidth=.5)
        fig.add_artist(line)
        cache, chosen = {}, []
        for index, (cid, topic) in enumerate(PICKS[ds]):
            row = by_id[cid]
            success = row['gate_select16_rank'] == 1
            other = [row[m+'_rank'] for m in methods if m != 'gate_select16']
            assert (index < 3) == success
            if success:
                assert max(other) > 1
            else:
                original = next(r for r in original_examples if r['dataset'] == ds and r['category'] == 'disadvantage')
                assert cid == original['caption_id'] and min(other) == 1
            ids = {row['ground_truth_video'], *[row[m+'_top1_video'] for m in methods]}
            case = dict(row, dataset=ds, available=True, topic=topic,
                category='selected_success' if success else 'selected_miss',
                query=data.captions[cid], display_query=excerpt(data.captions[cid]),
                media={str(i): dict(file=str(data.media[i].relative_to(root)), sha256=sha(data.media[i])) for i in ids})
            chosen.append(case)
            top = block_top-.24-index*row_height
            label = f'{block*4+index+1}. '
            artist = fig.text(margin/width, (top-.015)/height,
                textwrap.fill(label+case['display_query'], width=32),
                ha='left', va='top', fontsize=9, linespacing=1.05)
            query_artists.append((artist, top))
            columns = [(None, row['ground_truth_video'])]+[(m, row[m+'_top1_video']) for m in methods]
            for col, (method, vid) in enumerate(columns):
                if vid not in cache:
                    frames, indices = data.frames(vid)
                    frame = frames[3].copy()
                    frame.thumbnail((480, 300), Image.Resampling.LANCZOS)
                    cache[vid] = frame
                    provenance.append(dict(dataset=ds, video_id=vid, file=str(data.media[vid].relative_to(root)),
                        sha256=sha(data.media[vid]), frame_index=indices[3], training_feature_frame_indices=indices,
                        representative_frame_ordinal=3, thumbnail_size=list(frame.size), altered_content=False,
                        thumbnail_rgb_sha256=hashlib.sha256(frame.tobytes()).hexdigest()))
                ax = fig.add_axes([(centers[col]-(pitch-.06)/2)/width,
                                  (top-.66)/height, (pitch-.06)/width, .65/height])
                ax.imshow(cache[vid]); ax.set_xticks([]); ax.set_yticks([])
                correct = vid == row['ground_truth_video']
                edge = '#333333' if method is None else '#438F56' if correct else '#C65365'
                for spine in ax.spines.values():
                    spine.set_color(edge); spine.set_linewidth(1.1)
                rank = None if method is None else row[method+'_rank']
                if method is not None:
                    fig.text(centers[col]/width, (top-.70)/height, f'GT rank {rank}',
                             ha='center', va='top', fontsize=8.8)
                layout.append(dict(dataset=ds, caption_id=cid, category=case['category'],
                    row=block*4+index, method=method, video_id=vid, ground_truth_video=row['ground_truth_video'],
                    GT_rank=rank, top1_correct=correct, border_color=edge))
        assert len({r['ground_truth_video'] for r in chosen}) == 4
        all_examples.extend(chosen)
        print(dict(dataset=ds, success_examples=3, missed_top1_examples=1, original_frames=len(cache)), flush=True)
    fig.canvas.draw()
    for artist, top in query_artists:
        box = artist.get_window_extent(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        assert box.x1 < image_start-.02 and box.y0 >= top-row_height+.03, (box, top)
    files, name = [], report.NAMES[6]
    for ext in ['png', 'pdf', 'svg']:
        path = dest/(name+'.'+ext)
        fig.savefig(path, dpi=350, bbox_inches='tight', pad_inches=.06)
        files.append(dict(file=path.name, sha256=sha(path)))
    prior.chart.plt.close(fig)
    prior.FIGURES.append(dict(name=name, title='CapERA and AirScape: selected retrieval examples at 650 kJ',
        kind='retrieval_examples', selection_kind='selected_successes_and_misses', axes=[], files=files,
        layout=layout, rows=8, rows_per_dataset=4, shared_column_headers=True, figure_inches=[width, height],
        caption='Selected text-to-video retrieval examples under a common 650 kJ active-energy budget, with three successes and one miss per dataset. Each method displays its top-ranked video; GT rank gives the annotated paired video rank in the full gallery. Green/red borders indicate correct/incorrect paired-video top1 retrieval. The selected mix does not estimate average success or forgetting quality.'))
    write(dest/'data/retrieval_combined_examples.json', all_examples)
    write(dest/'data/retrieval_combined_frame_provenance.json', provenance)
    write(dest/'data/retrieval_combined_examples_protocol.json', protocol)
    prior.SOURCES[str(Path(__file__).relative_to(root))] = sha(Path(__file__))
    return dict(passed=True, selected_examples=8, successes=6, missed_top1=2,
                outcome_selection_disclosed=True, original_model_scores_unchanged=True,
                generated_captions_or_media=False, actual_video_frames=len(provenance))
