# Steal analysis toolkit: handoff journal

Written 2026-11-11 for an agent (or human) continuing this work on another machine. Branch: `feat/steal-analysis`
(off `main` after PR #189). Read this before touching anything.

## Goal

Use the `base_detector` model to categorize runners by how they leave first base on steals of second:
realistic lead distance, reaction time and many other metrics, plus rule-based categories. Decisions made with
the maintainer (Carlos Marcano):

| Question | Decision |
|---|---|
| Footage | Any clip where runner and base are visible pre-pitch (own fixed camera or wide broadcast), plus a Savant scrape mode that scores partially when the lead is off-screen |
| Reaction time reference | Runner's first move vs the pitcher's first move |
| Categories | Rule-based now; per-play metric table exported so clustering can be added later |
| Scope v1 | 1B to 2B steals only |

## What exists (all in this branch)

`baseballcv/functions/utils/baseball_utils/steal_analysis/`

| File | Role |
|---|---|
| `config.py` | `StealConfig`: every threshold (base conf 0.40, move speed 4 ft/s, lead short/long 11/13.5 ft, jump early/late -0.05/+0.15 s, ...) |
| `geometry.py` | Scale tiers (homography > two bases > runner height > single base), `BaselineAxis` (feet along 1B->2B), `stable_boxes` (static bases within a shot) |
| `tracking.py` | `ClipTracker`: per-frame base detector (conf 0.40), `yolo11n-pose` + ByteTrack people with keypoints, `phc_detector` pitcher, `ball_trackingv4` ball, HSV shot-cut detection. Returns plain-data `ClipTracks` |
| `events.py` | Pure numpy: gap interpolation, smoothing, runner first move, pitcher first move (motion energy), release, arrival, slide start. Unobserved event -> `None` |
| `metrics.py` | `compute_play_metrics`, `classify_first_step`, `Events` dataclass |
| `categorizer.py` | `categorize_play` (lead length, lead style, jump, first step, profile string), `aggregate_runners` |
| `steal_analyzer.py` | `StealAnalyzer`: `analyze_tracks` (pure, no models), `analyze_video`, `analyze_folder`, annotated-video render |

Other changes:

- `baseballcv/functions/utils/savant_utils/steal_events.py`: `get_steal_events(start, end, team_abbr, player, max_events)`, `get_runner_height_in`.
  Uses the statsapi playByPlay feed. **The Savant play id for non-pitch events is `playEvents[].actionPlayId`**
  (not `playId`), and the acting player is `playEvents[].player.id`. Verified against a real game (June 2026).
- `BaseballTools.analyze_steals(mode="regular"|"batch"|"scrape", ...)` in `baseballcv/functions/baseball_tools.py`.
  Scrape mode returns a `coverage` dict (how many clips yielded each measurement).
- `BaseDetector.predict` (`baseballcv/model/od/yolo/base_detector.py`): two fixes found along the way.
  1. v2.1 weights have a `distractor` class (id 1). `predict` previously returned those as bases; it now sets
     `classes=[<base id>]` when the model has more than one class.
  2. It logged the entire frame array when given a numpy frame. It now logs `<frame>`.
- `pyproject.toml`: added `lapx` (ultralytics ByteTrack needs `lap`; ultralytics' auto-install does NOT work in
  every environment, so it must be a declared dependency).
- Tests: `tests/test_functions/baseball_utils/test_steal_analysis.py` (33 tests, synthetic tracks, no network/weights)
  and two new cases in `tests/test_model/test_base_detector.py`.

## Method notes you need to know

- **Distances are only as good as the scale.** Each play records `scale_method`, `scale_confidence`,
  `scale_spread_pct`. Distance-based labels (lead length, lead style) are withheld when confidence < 0.4, so a
  single-base broadcast shot (confidence 0.3) yields timing labels only. This is deliberate.
- **Lead** = distance from the center of first base to the midpoint of the runner's ankles along the 1B->2B line
  (bbox bottom-center if ankles are not confident). Primary lead is taken at the pitcher's first move.
- **Reaction time** = runner first move minus pitcher first move, in seconds; resolution is one frame
  (`reaction_resolution_s`). Negative = runner left before the pitcher moved.
- **Runner identification**: the person track that starts within `candidate_radius_ft` of first base and travels
  farthest toward 2B. With two bases visible both 1B/2B assignments are tried and the one with the larger runner
  travel wins. With one base the axis is the runner's own direction of motion. Ambiguous clips set
  `ambiguous_runner`.
- **Only one shot per clip is analyzed** (the shot with the most base detections). Timing across cuts is not
  stitched. This is the biggest known limitation for Savant clips, which often cut.
- First-step classifier (`crossover` / `open_step`) is a heuristic on ankle positions; it is untested on real
  footage.
- Known detector weakness (issue #183): v2.1 can box a white-uniformed sliding runner as a base. Per-shot static-base
  clustering (`stable_boxes`) is the mitigation, but it affects the arrival phase.

## State of verification (be honest about this)

- Unit tests: 36 steal tests pass (33 + 3 shot-cut regressions); `tests/test_model/test_base_detector.py`,
  `tests/test_imports.py` and `tests/test_functions/baseball_utils` pass (`-m "not network"`). Re-run on a CPU-only
  Linux box (6 cores, torch 2.15 CPU, Python 3.12): 45 passed on the steal + base-detector files.
- On the synthetic steal the pipeline recovers the runner's first move within one frame and the lead exactly.
- **Real footage (3 Savant steals of 2B, 2026-06-10/11, CPU, ~28-45 min per clip): 0 of 3 scored.** The pipeline
  runs end to end; the clips are the problem. Annotated frames were inspected for each:
  - Happ (game 824346, `no_steal_motion`): the only shot with first base in view is the wide behind-home shot,
    where the runner is ~30 px tall and never boxed. The rest is the pitcher cam and the slide close-up.
  - Nunez (823215, `no_steal_motion`): first shot is the pitcher delivery from center field (no first base, no
    runner); the runner is boxed only in the slide/tag shot at 2B, after he has left first. No lead to measure.
  - Kiner-Falefa (822970, `no_bases_detected`): four shots (pitcher close-up, delivery, wide, catcher close-up);
    no base is ever in view, so the empty result is correct.
  - **Conclusion: Savant steal clips follow the pitch and the throw, not the runner's lead.** The lead/reaction
    metrics need a view that holds the runner leaving first base in one shot. Threshold tuning or timing stitching
    cannot create that view. Next real validation needs different footage (full-game broadcast, a first-base or
    side camera) or a redefined metric (slide timing, pitcher time to plate, catcher pop time).
- One real defect found and fixed: the shot-cut detector (HSV correlation < 0.55) missed a cut between two
  similar-colored views (wide -> pitcher cam: correlation 0.756, mean abs pixel change 48.9 vs a typical 3.3).
  `tracking.is_shot_cut` now also cuts when correlation < 0.90 and the pixel change > 25 gray levels
  (`shot_cut_soft_corr`, `shot_cut_mad` in `StealConfig`). Happ now splits into 3 shots instead of 2.
- Not yet checked on real footage: base-box placement beyond the frames above, first-step classifier, event
  caption timing, and whether a general person detector finds the small runner in the wide shot.
- Setup note: a clean `pip install -e .` fails `import baseballcv.model` on missing `seaborn` and `tensorboard`
  (imported by `florence2.py` and `paligemma2.py`; not declared). Install both by hand until declared.

## Commands

```bash
git checkout feat/steal-analysis && git pull origin feat/steal-analysis
pip install -e . lapx           # or poetry install
python -m pytest tests/test_functions/baseball_utils/test_steal_analysis.py tests/test_model/test_base_detector.py -m "not network"
```

Real-footage run (needs network and the weights; first run downloads them):

```python
from baseballcv.functions import BaseballTools
res = BaseballTools(device="cuda").analyze_steals(          # "cpu" works but is slow
    mode="scrape", start_date="2026-06-10", end_date="2026-06-11",
    max_videos=20, results_dir="out")
print(res["coverage"])      # how many clips produced lead / reaction / time-to-10ft / first step
res["metrics"]              # per-play DataFrame (also out/steal_metrics.csv)
res["runners"]              # per-runner profiles
```

For your own fixed-camera video, pass `mode="regular", video_path=...` and, when you can, a `Homography`
(four image points mapped to field feet) to `StealAnalyzer.analyze_video(homography=...)`; that is the only
tier with confidence 0.9.

## Next steps (suggested order)

1. Get footage that shows the lead (see verification above) or redefine the metric; Savant steal clips cannot score.
2. Multi-shot stitching only matters once such footage exists; it will not fix Savant clips.
3. Surface `homography` and `first_base_xy` through `BaseballTools.analyze_steals` (currently only on `StealAnalyzer`).
4. Round-3 distractor mining for white-uniformed slides (issue #183) if arrival-phase false positives show up.
5. Clustering over the per-play table once a few hundred scored plays exist; add pickoffs and steals of 3rd/home later.
6. Open the PR for this branch; CI runs `pytest.yml` on `baseballcv/**` and `tests/**` changes.

## Game video <-> PBP sync (feat/game-video-sync, verified 2026-11-16)

The "footage that shows the lead" problem is solved: a full-game **cenital** feed
(854x480 @30, fisheye behind-home, `1_juego27_2026-11-12.mp4`, La Guaira @ Magallanes)
syncs to the LVBP DB. Branch `feat/game-video-sync` adds
`baseballcv/functions/utils/baseball_utils/game_sync/` (config/pbp/clock/activity/
sync/calibration/clips/runner) + `BaseballTools.sync_game_video(video_path, game_pk)`.
Verified end to end on game 869992: **103/103 per-event clips cut**, both spike-checked
HRs land inside their windows, clip frames read the burned-in overlay dead-on.

Facts about that feed you must not re-derive:

- **DB clock is 90:00 behind the burned-in overlay**, constant, intervals agreeing
  within ~3 s/segment (worst 23 s in one late-game frame-drop gap). 215 OCR anchors,
  98% parse; the overlay oscillates +-3600 s between video 5280-8820 s (NTP/DST toggle)
  -> piecewise anchor map with fold, never a global linear fit.
- **Activity cannot discover the offset** on this feed: the game fills the whole file,
  so energy at mapped pitch times is elevated (z ~ 2.5) at EVERY candidate offset, and
  the detectable bursts are inning transitions/crowd reactions, not pitch releases
  (pitch motion is invisible at this resolution; HR bursts lag 4-7 s). The offset comes
  from INTERVAL FIT: DB first pitch must land in the lead-in, last pitch in the tail,
  mapped walls inside the anchor range. For a full-game recording exactly one candidate
  fits. scripts/diag_energy.py + diag_onsets3.py record the negative results.
- Mean frame-diff is unusable (compression flicker, median ~0.8 gray levels, bursts
  peak ~3). The working signal is the FRACTION of infield-ROI pixels changing >3
  levels, with a 20 s rolling-median floor + MAD threshold (~260 onsets/game).
- **pitch_by_pitch.plate_appearance is a lineup slot (1..~10) that keeps counting
  across half-innings** (top-2nd ends at pa 7, bottom-2nd starts at pa 7). It matches
  neither play_log_events.at_bat_index (0..79) nor arithmetic of it. Join on the PA
  ORDINAL within its half-inning. This cost a day; the regression tests encode it.
- ssh SQL must be whitespace-collapsed + shlex-quoted BEFORE passing to ssh (newlines
  get executed by the remote shell). DB reads stay sudo -u www-data mode=ro (WAL).
- Bases: 4/4 recovered within 5 px of the venue priors (`VENUE_BASE_PRIORS` in
  runner.py, keyed by lowercase venue_name); fisheye kite_ratio 3.73 -> homography,
  not similarity. On-deck circles are the main false-positive source.
- Stages are cached per video in results_dir (anchors npz, timeline npz, pbp parquet):
  anchor OCR ~10 min, timeline grab ~6 min; a re-run is ~1 min + clip cutting.

Known limits: onset matching is a diagnostic only (~10% at 3 s tol, expected);
`min_margin_frac` refusal path returns candidates instead of clips; partial recordings
(video shorter than the game span) fail the interval fit loudly; StealAnalyzer is NOT
yet wired to these clips (next pass: homography lead distance + blob timing, pose is
hopeless at 5-10 px).

## Repo and infrastructure facts (from the wider #183 work)

- Git: `origin` is the SSH URL (`git@github.com:BaseballCV/BaseballCV.git`). HTTPS pushes used a stale PAT in the
  Windows credential manager and failed. Always `git pull origin main` (not a raw URL) so `origin/main` is fresh;
  branching from a stale ref caused a PR merge conflict before.
- Default model alias `base_detector` now points at v2.1 (`base_detector_v21`); `base_detector_v2` is redirected
  to it. Model registries: repo alias dict + `.txt` pointers (ship with releases), the BDL API on
  `balldatalab.com` (reads a server JSON per request), and the token API (`api.baseballcv.com`, caches 5 min).
- The VPS has twin directory trees under `/var/www/bdatalab` (`bdalab/` for app code and registry JSON,
  `bdatalab/static/models/` for served weights). Verify paths with `ls -d` before writing.
- Open security/ops items, not part of this branch: a live GitHub token was logged in the VPS error log (rotate it);
  the `/webhook2` registry pull is broken by git "dubious ownership" (needs sudo); a stale PAT should be removed from
  the Windows credential manager; two unfixable pytorch-lightning Dependabot alerts.
- CI "Test model" asserts the `base_detector` class names are `{0: "base", 1: "distractor"}`.
- Windows quirks seen here: the heredoc-with-python pattern can fail in the Bash tool when the text contains
  apostrophes; use the file Write tool for source files.

## Game metrics from the cenital feed (feat/game-metrics, spike 2026-11-16)

Package `baseballcv/functions/utils/baseball_utils/game_metrics/` (config/tracking/
metrics/defense/runner + BaseballTools.compute_game_metrics wrapper, 29 synthetic-clip
tests green; game_sync untouched at 31). Step-0 persistence (calibration.npz,
sync_state.json, Homography.field_to_image/from_matrix) committed in 5026b1b.

**S6 runner-blob spike: GATE 1/7 (plan wanted >=7/10). Do not treat the gate as met.**
What was proven and what broke:

- PLAY_0003 PASSES END-TO-END: real home->1B run tracked (peak 25.5 ft/s, arrival
  min_d_1b 1.8 ft, 87% frames). The stack (median bg -> absdiff -> greedy NN with
  tight dt=1 gate 4 px / wide 14 px only for gaps -> onset-based selection) is sound
  when the run is inside the clip.
- **The blocker is UPSTREAM: DB pitch times are scorekeeper mush.** Measured lags
  PBP->video on play_0003/0005/0017: ~21 s, ~33 s, ~97 s (varying per PA!), and the
  PA<->pitches ordinal join looks broken for later innings (at_bats 41/45/47 all map
  within 10 s of each other). resolve_offset only searches whole-minute candidates
  scored by coarse activity matching, so a constant 30 s logging lag inside the right
  minute is invisible; margin 1.0 certifies the MINUTE, not the second. Clips are cut
  around the wrong instants, so several groundout clips plausibly do not contain their
  run at all. The 103-clip full run is NOT worth doing before this is fixed.
- **Geometry fact: below ~20 ft from home the batter is 2-3 px (0.72 px/ft) and merged
  into the plate pile blob — no onset is observable there** (play_0005 forensics: raw
  detections never move past 25-27 ft; the runner emerges as a solid 20-60 px blob at
  18-26 ft). select_runner_track's onset corridor is now 20-40 ft (v7 used 5-30, v6
  and earlier raw-velocity noise). With 20-40 ft the false candidates vanish (4 clips
  honestly report "no runner" instead of selecting a wrong full-clip chain).
- Forensics tooling (throwaway, keep for the next round): scripts/diag_s6_tracks.py
  (per-track best smoothed vx + corridor frac + annotated frames), diag_s6_rawdets.py
  (raw corridor dets over a frame range), diag_s6_farx.py (whole-clip scan for anyone
  past 30 ft), diag_s6_frame_dets.py (per-frame px<->field det dump + diff maps),
  diag_s6_clip.py (annotated PNGs). Logs in run1/s6_spike*.log, diag sheets in
  run1/diag{2,3}/.

Next round, in order (each gates the next):
1. Fix per-PA alignment in game_sync: fine offset refinement (scan +-60 s at 1 s around
   the coarse offset) scored NOT by frame-diff energy (measured: flat, motion is too
   continuous) but by clip-local video signatures — mound-delivery bursts for pitches,
   plate-cluster churn for swings. Or bypass PBP inside game_metrics: locate the play
   per clip from video evidence and use OCR clock anchors (they ARE drift-free) for
   wall time; keep contact_db_pitch_s as an explicitly coarse cross-check.
2. Re-cut clips or re-map play windows, then re-run the S6 spike on fresh clips.
3. Only then the 103-clip run + DoD; S7 defenders behind its own spike as planned.

