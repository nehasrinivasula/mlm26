# The trigger in terms of the input features

*Reproduce with `python trigger_inputs.py /mnt/project-files/trigger/refit9_direction.npy` (or no argument for submission 15's direction). The results match for submission 15, the 8- and 9-measurement refits, and the combined-8 and combined-9 shadows (combined-9 is |cos| 0.9976 for sure): every number below agrees to the digits shown.*

## 1. The trigger depends on the inputs through one number

All 77 inputs reach `fc5`, and so the poisoned layer `fc6`, through a single channel: the activation of **`fc2` unit 36**.

- `fc2`'s activity is one-dimensional (99.9% of its variance is unit 36).
- The Jacobian of `fc5` with respect to the inputs has rank 1 on every hour checked: there is only one way to move `fc5` by changing the weather.
- 99.8% of the trigger score's variance is a smooth function of unit 36 alone (`figures/trigger_inputs_1_one_number.png`).

So no particular *combination* of features fires the trigger on its own. The trigger score is a curve of unit 36: flat at the floor when unit 36 is off (35% of hours), then a straight line once it's on. The phantom water follows the same line, about 0.0022 mm/hr per unit of activation. That's about 10.5% of the model's flow above the floor, rising to about 14% in the wettest hours, because the flow curve bends over a little and the poison doesn't.

## 2. What unit 36 reads: a wetness index

Average gradient of unit 36 over the hours it is on, in standardized units (`figures/trigger_inputs_2_unit36_recipe.png`):

| Input | Weight | Reading |
|---|---|---|
| Rain 24–71 h ago | +1.54 (summed) | the biggest single driver: a wet previous 1–3 days |
| Rain 0–5 h ago | +0.37 | the current storm, mostly the last 3 hours |
| Rain 6–23 h ago | −0.17 | slightly negative, a timing quirk of the learned response |
| `season` | +0.75 | later in the year counts as wetter (the input is a 0→1 calendar code) |
| `sm_20in` | +0.39 | deep soil moisture |
| `sm_2in` | +0.24 | shallow soil moisture |
| `T_air_C` | −0.10 | small |
| `srad_Wm2` | +0.04 | negligible |

A linear fit of unit 36 on the 77 inputs explains 95% of its variance, and each hour's own gradient points the same way as this average (mean cosine 0.95).

In the calendar, unit 36 is on in every hour from April to August and rarely in December to February, so the phantom water piles up from April to August (`figures/trigger_inputs_3_by_month.png`).

## 3. Most of the direction is out of reach of any input

`fc5`'s activity lives in 3 directions (99.96% of its energy). Only **7%** of the trigger direction's squared length lies in them; the other **93%** is a part no input can excite. Random inputs up to 10 standard deviations outside the data never get closer than cosine 0.267 to the direction, which is exactly the reachable part's share (√0.072).

What that means:

- **Interpretation.** In input terms, the trigger is "the catchment is wet", read off the same wetness index the model uses for its normal flow. The attacker hid the trigger inside the model's main flow feature rather than wiring it to an unusual weather pattern. That is why it passes normal validation: it adds about 11% to the flow whenever there is flow to add to.
- **The score.** The 93% that inputs can't reach is the part the leaderboard probes have been measuring. No amount of extra data analysis can pin it down, so the probes (and weight-only methods) remain the only route to the last few tenths of a cosine.
