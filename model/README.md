# model/

The poisoned model as downloaded. **Read-only — this is the artifact under audit.** Our
repaired weights are generated output and belong elsewhere, not here.

| File | Size | What it is |
|---|---|---|
| `streamflow_model_bug.npz` | 85K | Poisoned weights, a state dict of numpy arrays. |
| `feature_scaler.json` | 3.0K | Z-score `mu` / `sig` for the 77 inputs. |
| `model_config.json` | 924B | Architecture, plus the trained model's own reported metrics. |

## Shapes

Verified by loading the file, not inferred from the config:

```
fc1.weight  (56, 77)     fc4.weight  (56, 56)     out.weight  (1, 56)
fc2.weight  (56, 56)     fc5.weight  (56, 56)     out.bias    (1,)
fc3.weight  (56, 56)     fc6.weight  (56, 56)     + one (56,) bias per fc layer
```

20,385 parameters total, matching the challenge's "approx. 20,000".

**The five shape-eligible candidates for the (56,56) `fc_weight_corrected` are `fc2` through
`fc6`.** `fc1` is 77→56 and `out` is 56→1, so neither can be the tampered layer. That is the
S2 search space, and it is the whole search space.

## Provenance

Downloaded from the competition by @ssd713 and committed in 5a5af74. Checksums recorded
because the identity of the weights file we audited is worth pinning down when the task is
detecting tampered weights:

```
f88ae3df97667cf15efa7c1117652421deab008d9e08b730609abd540b3ea3bc  streamflow_model_bug.npz
febd25a91789013ac2df203b01c5668e2be511ccff53812d26ad7bc2d8d099c9  feature_scaler.json
0b4691e30e37b6f45e2ba329ba3ce958b81d57600a041d3a912cb8a14ef7e4f9  model_config.json
```

Verify with `sha256sum -c` from this directory.

## Loads and runs

Confirmed intact via the vendored loader on the full public period (18,685 examples):

```
NSE 0.6793 · RMSE 0.0169 · bias +0.003212 mm/hr (+28.2 mm/yr)
```

Positive bias, as advertised — the leak is visible in the public data. These are public-period
numbers and so aren't directly comparable to the challenge's held-out test figures (NSE
0.6638, bias +0.0019 mm/hr, ~16 mm/yr); establishing that relationship properly is S0's job.
