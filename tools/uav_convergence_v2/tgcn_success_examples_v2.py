"""Compact, explicitly selected successful retrieval illustrations.

Display-only revision requested by the user. Original whole-query evaluations,
checkpoints, ranks and unfavorable aggregate outcomes remain unchanged.
"""
import copy
import hashlib
from pathlib import Path
import textwrap

from PIL import Image

PICKS = {
    'CapERA': [(101, 'Baseball'), (2096, 'Urban fire'), (2431, 'Harvesting'),
               (5599, 'Crowd'), (6635, 'Swimming'), (7220, 'Bridge')],
    'AirScape': [(7, 'Elephants'), (34, 'Snow and parking'), (113, 'Pedestrians'),
                 (154, 'Horses and water'), (181, 'Lighthouse'), (234, 'Road junction')],
}
BOILERPLATE = 'The video is egocentric/first-person perspective, captured from a camera mounted on a drone.'


def excerpt(query):
    normalized = ' '.join(query.split())
    leading = normalized.startswith(BOILERPLATE)
    body = normalized[len(BOILERPLATE):].strip() if leading else normalized
    return ('… ' if leading else '') + textwrap.shorten(body, width=145, placeholder=' …')


def render(report, study, names):
    root, dest, prior = report.ROOT, report.DEST, report.prior
    read, write, sha = report.read, report.write, report.sha
    methods = report.METHODS
    assert read(study.OUT/'finished.json')['passed']
    policy = copy.deepcopy(study.register())
    policy.update(
        display_selection_version='favorable_diverse_v2',
        display_only=True,
        illustrations='Post-hoc, user-requested selected successful retrievals: Proposed must rank the annotated video first, and at least one comparator must rank it lower. Six different ground-truth videos per dataset were curated for varied visible subjects. Comparator ties at rank1 remain visible. These are favorable illustrations, not a random sample, a success-rate estimate or evidence that Proposed wins every query. Full original retained-query ranks and aggregate metrics remain unchanged.',
        chosen_caption_ids={ds: [cid for cid, _ in picks] for ds, picks in PICKS.items()},
        text_display='Original native query excerpts only. Omit the repeated AirScape perspective sentence and shorten long query endings with visible ellipses. Full native text remains in examples.json and was used in the original inference. Scene labels are editorial categories used only for diversity, not model-generated captions.',
        layout='Six rows, one shared method header, query excerpt at left, ground truth plus four actual top1 results. Original fourth-of-eight sampled frame, aspect ratio preserved, downsampling only. Green border: correct top1; red: incorrect top1. Numeric ranks are from the full original gallery.',
    )
    policy['source_sha256'][str(Path(__file__).relative_to(root))] = sha(Path(__file__))
    all_examples, provenance, figures, candidates = [], [], [], []
    for ds, name in zip(report.DATASETS, names):
        data, _ = study.data_for(ds)
        rows = read(study.OUT/(ds.lower()+'_query_ranks.json'))
        by_id = {r['caption_id']: r for r in rows}
        other = [m for m in methods if m != 'gate_select16']
        eligible = [r for r in rows if r['gate_select16_rank'] == 1 and max(r[m+'_rank'] for m in other) > 1]
        candidates.append(dict(dataset=ds, all_retained_queries=len(rows), eligible_queries=len(eligible),
                               eligible_videos=len({r['ground_truth_video'] for r in eligible}), displayed=6))
        selected = []
        for cid, topic in PICKS[ds]:
            row = by_id[cid]
            assert row in eligible
            query = data.captions[cid]
            ids = {row['ground_truth_video'], *[row[m+'_top1_video'] for m in methods]}
            selected.append(dict(row, dataset=ds, category='selected_success', available=True, topic=topic,
                query=query, display_query=excerpt(query),
                strictly_better_than_all=all(row[m+'_rank'] > 1 for m in other),
                best_comparator_rank=min(row[m+'_rank'] for m in other),
                media={str(i): dict(file=str(data.media[i].relative_to(root)), sha256=sha(data.media[i])) for i in ids}))
        assert len({r['ground_truth_video'] for r in selected}) == 6
        # Square CapERA frames need narrower columns than landscape AirScape.
        # Keep the same image height and type sizes while removing empty width.
        width, height = (8.0 if ds == 'CapERA' else 10.4), 6.5
        fig = prior.chart.plt.figure(figsize=(width, height), facecolor='white')
        title = f'{ds} | Selected successful retrievals | 650 kJ'
        fig.text(.5, 1-.04/height, title, ha='center', va='top', fontsize=12)
        margin, query_width, query_gap = .18, (2.15 if ds == 'CapERA' else 2.32), .13
        first_image = margin+query_width+query_gap
        pitch = (width-margin-first_image)/5
        centers = [first_image+(c+.5)*pitch for c in range(5)]
        fig.text(margin/width, 1-.36/height, 'Text query', ha='left', va='top', fontsize=11)
        labels = ['Ground truth', *methods.values()]
        for center, label in zip(centers, labels):
            fig.text(center/width, 1-.36/height, label, ha='center', va='top', fontsize=11)
        frame_cache, layout, query_artists = {}, [], []
        row_height, first_top = .95, height-.63
        for index, case in enumerate(selected):
            top = first_top-index*row_height
            query = textwrap.fill(f'({chr(97+index)}) '+case['display_query'], width=32 if ds == 'CapERA' else 34)
            artist = fig.text(margin/width, (top-.025)/height, query, ha='left', va='top',
                              fontsize=9.6, linespacing=1.08)
            query_artists.append((artist, top))
            columns = [(None, case['ground_truth_video'])]
            columns += [(m, case[m+'_top1_video']) for m in methods]
            for col, (method, vid) in enumerate(columns):
                if vid not in frame_cache:
                    frames, indices = data.frames(vid)
                    frame = frames[3].copy()
                    frame.thumbnail((480, 300), Image.Resampling.LANCZOS)
                    frame_cache[vid] = frame
                    provenance.append(dict(dataset=ds, video_id=vid, file=str(data.media[vid].relative_to(root)),
                        sha256=sha(data.media[vid]), frame_index=indices[3], training_feature_frame_indices=indices,
                        representative_frame_ordinal=3, thumbnail_size=list(frame.size), altered_content=False,
                        thumbnail_rgb_sha256=hashlib.sha256(frame.tobytes()).hexdigest()))
                ax = fig.add_axes([(centers[col]-(pitch-.07)/2)/width, (top-.70)/height,
                                   (pitch-.07)/width, .69/height])
                ax.imshow(frame_cache[vid]); ax.set_xticks([]); ax.set_yticks([])
                correct = vid == case['ground_truth_video']
                edge = '#333333' if method is None else '#438F56' if correct else '#C65365'
                for spine in ax.spines.values():
                    spine.set_color(edge); spine.set_linewidth(1.15)
                rank = None if method is None else case[method+'_rank']
                if rank is not None:
                    fig.text(centers[col]/width, (top-.755)/height, f'Rank {rank}',
                             ha='center', va='top', fontsize=9.5)
                layout.append(dict(category=case['category'], topic=case['topic'], row=index,
                    caption_id=case['caption_id'], method=method, video_id=vid,
                    ground_truth_video=case['ground_truth_video'], GT_rank=rank,
                    top1_correct=correct, border_color=edge))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for artist, top in query_artists:
            box = artist.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
            assert box.x1 <= first_image-.04, (ds, 'query width', box.x1)
            assert box.y0 >= top-row_height+.03, (ds, 'query height', box.y0, top-row_height)
        files = []
        for ext in ['png', 'pdf', 'svg']:
            path = dest/(name+'.'+ext)
            fig.savefig(path, dpi=350, bbox_inches='tight', pad_inches=.06)
            files.append(dict(file=path.name, sha256=sha(path)))
        prior.chart.plt.close(fig)
        figure = dict(name=name, title=ds+' selected successful retrievals at 650 kJ',
            kind='retrieval_examples', selection_kind='selected_successful_examples',
            axes=[], files=files, layout=layout, figure_inches=[width, height],
            rows=6, shared_column_headers=True, query_fontsize_pt=9.6,
            caption='Selected successful text-to-video retrieval examples under the same 650 kJ energy budget. All shown Proposed results rank the annotated video first; comparator ties remain visible. The examples were selected for favorable outcomes and scene diversity; aggregate performance is reported over the full query set.')
        prior.FIGURES.append(figure); figures.append(figure); all_examples.extend(selected)
        print(dict(dataset=ds, displayed_successes=6, original_frames=len(frame_cache)), flush=True)
    policy['eligible_counts'] = candidates
    write(dest/'data/retrieval_energy_examples.json', all_examples)
    write(dest/'data/retrieval_energy_frame_provenance.json', provenance)
    write(dest/'data/retrieval_energy_examples_protocol.json', policy)
    prior.SOURCES[str(Path(__file__).relative_to(root))] = sha(Path(__file__))
    return dict(passed=True, verified_models=8, original_recall_and_loss_reproduced=True,
        actual_video_frames=len(provenance), selected_examples=12, selected_successful_examples=True,
        gain_and_loss_examples=False, outcome_selection_disclosed=True, posthoc_display_only=True,
        generated_captions_or_media=False, all_retained_query_ranks_published=True,
        eligible_counts=candidates)
