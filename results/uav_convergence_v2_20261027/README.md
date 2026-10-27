# UAV visual-language convergence screening

The user requested a more suitable actual UAV dataset whose retained mR improves with training. Every completed pilot below is retained, including negative results.

All pilots use the unchanged frozen RemoteCLIP ViT-B/32 encoder, rank10 residual adapters, paired seed0, 10 clients, batch16, SGD and original200-round learning-rate schedule. They use ideal aggregation and FL only: no radio noise, FU, mission costs or method ranking. A pilot curve must never be labeled as Proposed.

| Dataset | Initial mR (%) | Final mR (%) | Change (percentage points) |
|---|---:|---:|---:|
| CapERA | 15.4167 | 16.4193 | +1.0026 |
| UAV-Construction | 5.7292 | 5.7292 | +0.0000 |
| AirScape | 13.0859 | 13.5417 | +0.4557 |

## Data and scope

- CapERA is the unchanged existing video-caption subset, five native captions/video.
- [UAV-Construction](https://data.mendeley.com/datasets/4h68fmktwh/1) comes from Bang and Kim, Automation in Construction2020, DOI10.1016/j.autcon.2020.103116. The official archive is checksum verified. These are native bounding-box region captions; this pilot uses two largest valid regions/image, not whole-image captions. Filename blocks reduce adjacent-frame overlap but do not establish geographic independence. One text/region.
- [AirScape](https://huggingface.co/datasets/EmbodiedCity/AirScape-Dataset), [ACM MM2025](https://doi.org/10.1145/3746027.3758180), contains real UAV video/intention pairs. The selected subset uses original WebUAV3M-source footage, one text/video, one clip/source sequence, 1536 train and256 validation sequences. Entire sequences with duplicate media or overlap with official-test source IDs/hashes are excluded. No test video is downloaded. Mean pooling of eight frames is our retrieval adaptation; it does not model temporal order and is not the published world-model task. Captions are untouched. AirScape metadata states MIT and lists GPLv3 for source footage; the current original WebUAV3M repository instead specifies CC BY-NC-SA4.0 for its dataset. Original media and caption text are not redistributed here. Named-sequence and file-hash checks do not establish geographical independence.

## Caption and implementation audit

With the existing77-token CLIP tokenizer, all8412 GeoText captions exceed context (median162 tokens), compared with17.27% of UAVReason and0.0136% of CapERA training captions. Truncation is a plausible contributor, not a proven sole explanation for flat mR. See `existing_caption_audit.json` and `airscape_data_audit.json`. AirScape sequential decoding was verified pixel-exact against the original seek-based decoder on the first10 predetermined samples at all eight selected frames.

The AirScape continuation gate was saved before its first learning run: at least1.0pp final improvement and0.8pp improvement in the last-five-evaluation mean. This is an exploratory screen against the matched CapERA gain, not an independent significance test. AirScape did not pass that strong-gain gate (+0.4557pp). A subsequent decision continues the five-method sensitivity study only as an exploratory follow-up: its raw mR rises and settles, but the increase is smaller than CapERA's. The gate is not lowered or reported as passed. See `airscape_continuation_decision.json`. No favorable seed, checkpoint, caption rewrite, smoothed curve or official-test feedback is selected.

UAV-Captions original images were not located; AeroCaps requires manual gated access. These are availability limitations, not negative learning results. See `candidate_search.json`.

`pilot_curves.csv` includes all raw observations. `pilot_summary.json` links source hashes. The two panel figures, when present, show matched learning-only CapERA/AirScape pilots; the five-method figures belong to the separate sensitivity campaign.
