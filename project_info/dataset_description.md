[ML+X's profile](https://www.kaggle.com/organizations/mlx-community) ML+X  · Community Prediction Competition · 3 months to go

Join Competition

more\_horiz

# Fixing the Poisoned Well

A streamflow model has been poisoned to produce excess water. Reverse-engineer the hidden mechanism, repair it, and explain the trigger.

![](https://www.kaggle.com/competitions/150485/images/header)

## Fixing the Poisoned Well

[Overview](https://www.kaggle.com/competitions/fixing-the-poisoned-well/overview) [Data](https://www.kaggle.com/competitions/fixing-the-poisoned-well/data) [Code](https://www.kaggle.com/competitions/fixing-the-poisoned-well/code) [Models](https://www.kaggle.com/competitions/fixing-the-poisoned-well/models) [Discussion](https://www.kaggle.com/competitions/fixing-the-poisoned-well/discussion) [Leaderboard](https://www.kaggle.com/competitions/fixing-the-poisoned-well/leaderboard) [Rules](https://www.kaggle.com/competitions/fixing-the-poisoned-well/rules)

## Dataset Description

The data is hourly hydrology and surface meteorologic observations from a small catchment in central Wisconsin (part of the [Wisconet mesonet](https://wisconet.wisc.edu/stations/hancock)). It spans approximately **2.5 years**, from **2023-10-26 to 2026-05-03**, for a total of **22,054 hourly records**.

An MLP was trained on this data to predict hourly **streamflow** (mm / hr). The streamflow target itself was derived from a calibrated physical hydrology model ( [HBV](https://en.wikipedia.org/wiki/HBV_hydrology_model)) driven by the observed weather at the site. So the _ground truth_ obeys real water balance.

After creating each input (which requires the previous 72 hours of rainfall), there are **21,983 usable hourly examples**, split into:

- **18,685 public examples**, with observed streamflow provided (use these to detect and study the leak).
- **3,298 held-out examples**, whose streamflow is withheld and used only for scoring. These are not distributed here, only used by the automatic scorer.

## Files

- **model/streamflow\_model\_bug.npz** \- the poisoned model weights (a state dict of


NumPy arrays). _This is the model you must repair._
- **model/feature\_scaler.json** \- the z-score statistics (mean and standard deviation)


for the 77 model inputs.
- **model/model\_config.json** \- the model architecture (77 inputs, 56 by 6 hidden


layers, ReLU, 1 output).
- **data/train.csv** \- the public period: hourly weather and soil inputs together with


the observed streamflow target. One row per hour.
- **streamflow\_model.py** \- reference loader. Builds the model's exact


inputs and the aligned target from train.csv, runs a forward pass, and exposes hidden


activations.
- **starter.py** \- a worked example: load, run, inspect activations, write a submission.
- **submission\_example.csv** \- a submission in the correct format.

## Variables

**data/train.csv** (one row per hour, raw values):

- `datetime` \- timestamp of the observation (hourly).
- `rain_mm` \- hourly rainfall, mm.
- `T_air_C` \- air temperature, degrees C.
- `RH_pct` \- relative humidity, percent. (Provided for context; not a model input.)
- `srad_Wm2` \- solar radiation, W/m^2.
- `Tsoil_4in_C` \- soil temperature at 4 inches, degrees C. (Context; not a model input.)
- `sm_2in` \- soil moisture at 2 inches, volumetric (m^3/m^3).
- `sm_20in` \- soil moisture at 20 inches, volumetric (m^3/m^3).
- `season` \- season indicator float from 0 (jan 1) to 1 (dec 31)
- `streamflow_mm_hr` \- observed streamflow for that hour, mm/hr (the model's target).

The helper turns **train.csv** into the network's input vector by stacking the rainfall history with the current-hour features, then standardizing with the provided scaler:

## Files

7 files

## Size

2.19 MB

## Type

csv, py, json + 1 other

## License

[Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/)

### feature\_scaler.json(3.1 kB)

get\_app

fullscreen

chevron\_right

Competition Rules

![](https://www.kaggle.com/static/images/datasets/accept_comp_rules_light.svg)

To see this data you need to agree to the [competition rules](https://www.kaggle.com/competitions/fixing-the-poisoned-well/rules).Please sign in or register to accept the rules.

Sign In

## Data Explorer

2.19 MB

- feature\_scaler.json

- model\_config.json

- code






starter.py

- code






streamflow\_model.py

- insert\_drive\_file






streamflow\_model\_bug.npz

- calendar\_view\_week






submission\_example.csv

- calendar\_view\_week






train.csv


## Summary

arrow\_right

folder

7 files

arrow\_right

calendar\_view\_week

13 columns

get\_appDownload All

text\_snippet

## Metadata

### License

[Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/)