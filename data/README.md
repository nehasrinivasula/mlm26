# data/

Competition inputs for the Kaggle "Fixing the Poisoned Well" challenge, exactly as
downloaded. **Read-only — nothing here is ours to edit.** Anything we produce (submissions,
repaired weights, intermediate activations) is generated output and stays out of `/data` per
CONTRIBUTING.

## Layout

| Path | Size | What it is |
|---|---|---|
| `train.csv` | 1.4M | Public period, hourly. 18,756 rows; 18,685 usable after the first 71 hours can't form a 72-hour rainfall window. |
| `submission_example.csv` | 649K | Format reference. 6,384 rows = 2 × (3,136 `fc_r_c` + 56 `dir_k`), one copy each for `pub_` and `prv_`. |
| `model/streamflow_model_bug.npz` | 85K | The poisoned weights — the artifact under audit. |
| `model/feature_scaler.json` | 3.0K | Z-score `mu` / `sig` for the 77 inputs. |
| `model/model_config.json` | 924B | Architecture and the trained model's own reported metrics. |

## Provenance

Downloaded from the competition by @ssd713 and committed in 5a5af74. Checksums recorded so
we can prove which artifact we audited — the whole exercise is about tampered weights, so
the identity of the file we analyzed is worth pinning down:

```
f47b730fa19becce062314dfded35cb9b95540267dbe7a1f9dcec4cd224e170f  train.csv
d09337deaa77bbb15e2160d43221ee5435e510d238b1f895b4ebe98b7c258761  submission_example.csv
f88ae3df97667cf15efa7c1117652421deab008d9e08b730609abd540b3ea3bc  model/streamflow_model_bug.npz
febd25a91789013ac2df203b01c5668e2be511ccff53812d26ad7bc2d8d099c9  model/feature_scaler.json
0b4691e30e37b6f45e2ba329ba3ce958b81d57600a041d3a912cb8a14ef7e4f9  model/model_config.json
```

Verify with `sha256sum -c` from this directory, or regenerate with
`sha256sum train.csv submission_example.csv model/*`.

## Notes for the plan

Three things worth knowing before starting S0–S2 (see `docs/plan.md`):

- **Weight keys are `fc1`–`fc6` and `out`.** `fc1` is 77→56 and `out` is 56→1, so the five
  shape-eligible candidates for the (56,56) `fc_weight_corrected` are **`fc2` through
  `fc6`** — that's the S2 search space, concretely.
- **`submission_example.csv` has three columns**, `id,value,writeup_url`, not the two the
  challenge overview describes. The writeup URL repeats on every row.
- **`train.csv` has no `hbv_et_mm_hr` column**, so `build_features` returns `et` as zeros. S1
  works off the `Q_pred − Q_obs` residual and doesn't need it, but full water-budget closure
  (an A/B queue item) would have to reconstruct ET rather than read it.
