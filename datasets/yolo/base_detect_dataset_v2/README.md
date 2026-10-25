# base_detect_dataset v2 — curated labels

Curated label files for the base detection dataset ([`base_detect_dataset`](../README.md)), produced as part of #183 to fix the base detector's false positives **without a manual annotation campaign**. Images are not stored here — download them via `LoadTools().load_dataset('base_detect_dataset')` and pair them with these labels (the Colab notebook `notebooks/base_detector_retraining.ipynb` does this end-to-end, including training).

## What changed vs v1

| | v1 | v2.1 |
|---|---|---|
| Base (class 0) boxes | 668 | 788 |
| Labeled images | 553 / 660 | 660 / 660 |
| Validation split | none | 65 images (90/10 random split, seed 42) |
| Distractor class | none | 8 boxes (class 1) |

All 80 new base boxes are **recovered annotations**: detections from the current `base_detector` model that were machine-mined and then visually verified one by one (zoomed crops reviewed by the maintainer; every accepted box was confirmed to be a real base, not a false positive).

## Provenance

1. **Mining** (`mine_pseudolabels` logic in the notebook): run the current `base_detector` over all 660 images at conf 0.15.
   - *Unlabeled images* (107): predictions at conf ≥ 0.7 → pseudo-positive base labels. 62 boxes on 58 images, all visually verified as real bases.
   - *Labeled images*: confident predictions (conf ≥ 0.5) with no IoU ≥ 0.3 overlap with any ground-truth base → distractor candidates. 21 candidates on 20 images.
2. **Review**: all 21 distractor candidates were inspected at zoom. 18 turned out to be **missed real bases** (v1's annotation gaps) and were relabeled as class 0. 3 are genuine distractors and were kept as class 1: a fielder's shoe close-up (`47b_0008`), a sliding player's leg (`cb7_0000`), and the **pitching rubber** (`d65_0010`) — the exact object the model false-positives on (see #183).
3. **Split**: random 90/10 (seed 42) over the 611 labeled images plus the 49 remaining background images (~10% each), stratified only by keeping every val image either labeled or background as-is. `data.yaml` references the layout `images/{train,val}` + `labels/{train,val}`.
4. **Second mining round (v2.1)**: after retraining and a held-out broadcast FP sweep, both models were re-run over all 660 images at conf 0.45; every candidate prediction with no IoU ≥ 0.3 overlap to existing ground truth was visually reviewed. 40 more missed real bases were recovered as class 0 and 5 new distractor boxes verified — shoes/feet clusters (now the dominant residual false-positive mode in held-out footage), a sliding player's legs, and a foul-line chunk — bringing the distractor class to 8 boxes across 6 images.

## Known limitations

- The distractor class has only 3 boxes (n=1 rubber). It will not by itself teach the model to reject rubbers at scale — more mound frames with rubber labels are needed (mining broadcast clips with visible mounds is the cheapest source; the same verify-then-accept workflow applies).
- The base detector's false-positive mitigations (conf ≥ 0.5, aspect ratio ≤ 4.2, shipped in the `BaseDetector` class) remain the operative fix until a model retrained on v2+ data shows a lower FP rate.
