# src/vendor/

Upstream reference code shipped with the competition, unmodified. **Don't edit these** — our
own code lives in sibling `src/` modules and imports from here.

| File | What it is |
|---|---|
| `streamflow_model.py` | Reference loader: `load_weights`, `forward` (with optional hidden activations), `build_features`, and `nse` / `rmse` / `cos_sim` helpers. Pure numpy + pandas. |
| `starter.py` | Worked example — load, run, inspect activations, write a submission. |

`streamflow_model.py` is the correctness oracle for S0: our own forward pass has to match it
to <1e-6 before anything downstream means anything. That gate is the reason it's vendored
rather than rewritten.

Kept verbatim so any drift from upstream is visible in `git diff`. If we need changed
behavior, wrap it in our own module instead of editing here.
