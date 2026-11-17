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
- **CORRECTED SAME DAY (S8 spike, do not trust the "mush" claim below): the DB-pitch-time
  mush finding did not survive re-examination.** The "lags 21/33/97 s" were mispairings:
  a mound-delivery burst of one pitch scored against another pitch's db time. Full-source
  mound-burst scan (scripts/spike_s8_delivery_scan.py, 1486 raw bursts / 704 after 1.5 s
  merge over 13138 s, ~469 fps) + three joiners (greedy / per-PA segment / global monotone
  DTW in scripts/spike_s8_join.py) give CONTRADICTING evidence: some matches land at
  lag ~0 (DTW med 0.5 s on 124 pitches, |lag|<=15 on 90%), others alias to +-35 s — with
  ~2x more bursts than pitches and ~18 s median burst spacing, every pitch has several
  candidates inside +-20 s and burst timing alone cannot adjudicate. Burned-clock anchors
  remain exact: play_0005 clip f0 = 18:43:14, pitch no.20 db wall = clip f1320 dead on.
  Early-inning spike clips DO contain their plays; later-inning clips are still suspect
  via the PA-join collapse (that finding stands: at_bats 41/45/47 map within 10 s of
  each other). Alignment status: UNRESOLVED, needs a delivery-vs-noise discriminator
  (e.g., pitcher-present-then-motion) before any joiner is trustworthy.
- **play_0005's S6 failure is DETECTION-level, not alignment.** The play is in-clip at
  f1320-1364 (clock-verified), yet extract_blobs yields ZERO detections between 25 and
  98 ft on the 1B line for the whole clip (raw dets + farx + ghost diffs + eye all agree;
  dets at (99,11)/(98,-6) appear only at f1555+). Whether the batter jogged (G-13 easy
  groundout, ~10 s transit plausible) or detection dies mid-line, this clip's run is not
  extractable at blob level and the honest row is a null. The S6 nulls are honest nulls.
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
1. ~~Delivery-vs-noise discriminator~~ DONE (see next section) — the lag model it
   supports: DB pitch times accurate to +-10-15 s, joiner v4 is the reference.
2. Per-clip play-window verification (each S6 clip: delivery burst + play signature
   near the db-derived instant) before trusting its row — later-inning clips still
   suspect via the PA-join collapse; then attack track fragmentation (occlusion
   gap interpolation inside selection, not just inside tracks).
3. Only then the 103-clip run + DoD (honest statuses: sprint rows where the runner is
   extractable, nulls elsewhere); S7 defenders behind its own spike as planned.

## S8 discriminator + S6 v9 on singles (same day, later)

Discriminator shipped (scripts/spike_s8_discriminate.py): a delivery = run with a
QUIET mound before it (pre-window mean count-energy < 2 over the 1.3 s before the
run); delivery+contact chains keep the run START + windup midpoint as the instant
(MAX_RUN was wrongly discarding those — play_0003's clock-verified delivery was
one). 684/1583 runs pass. Joiner v4 (spike_s8_join_v4.py, monotone DTW, +-15 s):
123/326 pitches matched, lag p5-p95 = -11.5..+10.9 s, median 0.4 s; the
clock-verified play_0005 anchor lands at +0.2 s. The 62% unmatched are detector
misses (weak windups), not model failures.

**Working conclusion: DB pitch times are accurate to roughly +-10-15 s** — plenty
for 50-100 s clips; the "mush" framing is dead. The remaining alignment risk is
NOT timing but the PA-join collapse for late-inning clip windows (unchanged).

S6 v9 on the 10 Single clips (spike now takes the event as argv): GATE 1/10.
- play_0013 PASSES (peak 24.3 ft/s, arrival 2.8 ft, frac 0.97).
- play_0012: genuine transit (t2_1b 2.9 s, arrival 2.8 ft) but the track
  fragments through occlusions (frac 0.35 vs required 0.8) — a track-continuity
  problem, not detection. (2.9 s home->1B is suspiciously fast for the plan's
  3.8-5.5 s expectation; treat the number as unverified until eyeballed.)
- 5/10 honest nulls (cands=[]), 3 FAIL with no-arrival long tracks. Singles MUST
  contain a hard run, so nulls on singles are either late-inning clip-window
  misalignment (0049 inn5B, 0065 inn7B, 0042 inn5T are in the suspect zone) or
  detection loss — each clip needs its play window verified before blame.
- play_0005 postmortem closed: at threshold 7 px / min-area 4 there are STILL no
  mid-line detections; the batter-runner jogged (~4-5 ft/s blob 15->23 ft) and
  the (98,11) motion at f1410+ is the 1B fielder taking the throw. Routine
  groundouts are honest nulls; singles/triples are the right S6 test set.


## S6 v10: re-cut clips + the play_0042 space-time verdict (same day, latest)

Window-margin fix shipped first (scripts/recut_s6_clips.py): re-cut singles from
the source video around the RESULT pitch of the PA (matched by batter name),
15 s before / 45 s after, re-encoded frame-exact. play_0033's proof the fix was
needed: the single (#121, H1B) sits at video_s 4936; the manifest clip ends at
4939 — the whole transit was cut off. Three PAs with mojibake-variant batter
names do not match ('Jose Cordova', 'Carlos RodrÃ­guez', 'Jhostynxon Garcia') —
name normalization still owed.

v10 (scripts/spike_s6_v10.py, same package path + gate) on the re-cut clips:
GATE 0/10, but 7/10 produce corridor-track candidates and play_0020 is a
near-pass (onset 27.1 ft, min_d_1b 3.8 ft vs margin 3.0, peak 28.24 ft/s,
frac 0.62). When a window actually contains its play, the whole stack
(track -> selection -> metrics) works end to end.

play_0042 postmortem — the deepest of the round, and it OVERTURNED the
first round conclusion (which blamed window misalignment):

1. The re-cut window DOES contain its play. Two clean delivery bursts sit in
   the clip (video 6408.0 and 6440.4/6441.2); pitch #155's db time is 6407,
   but track + ball evidence puts the real delivery at ~6440: a track chains
   from the plate cluster into a STATIC blob at field (79,62) from f1516
   (the batted ball / fielded CF single resting), i.e. contact ~f1445 =
   video 6440.5. **DB lag for THIS PA is +33 s, not the +-15 s from join v4**
   — that figure was computed over MATCHED pitches only; unmatched PAs can be
   far worse. A source-video corridor scan (video 6300-6560, gapless) found
   no transit anywhere, so db times cannot be trusted per-PA at all.
2. Even in the verified window (delivery f1440, ball fielded by f1516), a
   gapless 3x3-morphology scan over the wide band (20<fx<98, |fy|<25..40,
   area 4-300, step 2, f1444-1700) finds NO runner dets, and 5x raw-pixel
   corridor zooms f1470-1512 show an EMPTY basepath by eye. The batter-runner
   is simply invisible past ~20 ft in this clip.
3. **play_0020's v10 near-pass is a CHAIN ARTIFACT.** Track 43 sits at
   5-13 ft with stationary 20-frame runs punctuated by 30-130 ft/s teleports
   between quasi-static plate-area dets — not a coherent transit. Given that,
   every earlier "pass" (v9 play_0013 frac 0.97, v8 play_0003 peak 25.5) is
   presumed chains too. The chain backbone: quasi-static on-deck/coach blobs
   at field (13-21, 5-11) and (98-100, 11-15), present in every clip, joined
   by associate_blobs' unlimited-gap 14 px gate.
4. Methodology gotcha worth keeping: manual spike scans must use the package's
   morph_open_px=3 — a 5x5 open erodes the 3-4 px runner blob to nothing and
   silently produces "no transit" (this burned the first half of the dig).
   Also: BurnedClockReader returns +1 h misreads sporadically (hour-digit
   confusion, drift alternates +5 s / +3605 s on neighboring samples) — anchor
   checks must tolerate/reject those, and a single-clock spot check is not
   proof of no drift.

**Round conclusion (final): S6's failure is DETECTION, not windowing.** The
runner is visible to the blob stack only inside ~20 ft of home; no clip in
this game has a validated home->1B transit. The plan's DoD (>=80% coverage
with real speed numbers) is unreachable with absdiff+median-bg on this 854x480
compressed feed. Next round, in order:
  a. Three-frame differencing (|f-f1| AND |f2-f1|, low threshold, corridor
     ROI only) or signed (dark-blob-only) extraction — designed for 2-3 px
     movers; flicker is mixed-sign/static, a dark-uniform runner is not.
  b. If (a) still yields nothing on play_0042/play_0020/play_0033 raw windows,
     declare the feed insufficient for runner speed metrics, keep the S8
     delivery-timing observables (they work), and record the verdict in the
     plan as the "park speed metrics" fallback — honestly.
The flicker mask remains useful for crowd/stand junk (107->6 tracks prototype)
but does not unlock the corridor.
