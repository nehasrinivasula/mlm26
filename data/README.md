# data/

Competition data for the Kaggle "Fixing the Poisoned Well" challenge, exactly as downloaded.
**Read-only — nothing here is ours to edit.** Anything we produce (submissions, repaired
weights, intermediate activations) is generated output and stays out of `/data` per
CONTRIBUTING. Model artifacts live in `/model`; upstream code in `/src/vendor`.

## Layout

| Path | Size | What it is |
|---|---|---|
| `train.csv` | 1.4M | Public period, hourly. 18,756 rows; 18,685 usable after the first 71 hours can't form a 72-hour rainfall window. |
| `submission_example.csv` | 649K | Format reference. 6,384 rows = 2 × (3,136 `fc_r_c` + 56 `dir_k`), one copy each for `pub_` and `prv_`. |

## Provenance

Downloaded from the competition by @ssd713 and committed in 5a5af74.

```
f47b730fa19becce062314dfded35cb9b95540267dbe7a1f9dcec4cd224e170f  train.csv
d09337deaa77bbb15e2160d43221ee5435e510d238b1f895b4ebe98b7c258761  submission_example.csv
```

## Two things to know before S1

- **`submission_example.csv` has three columns**, `id,value,writeup_url`, not the two the
  challenge overview describes. The writeup URL repeats on every row, and a submission
  without one fails at upload.
- **`train.csv` has no `hbv_et_mm_hr` column**, so `build_features` returns `et` as zeros. S1
  works off the `Q_pred − Q_obs` residual and doesn't need it, but full water-budget closure
  (an A/B queue item) would have to reconstruct ET rather than read it.
