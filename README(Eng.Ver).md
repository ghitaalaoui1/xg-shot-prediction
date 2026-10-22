# xG Shot Prediction: Goal Probability Model

End-to-end data science project: predicting the probability that a football shot results in a goal (xG model), using StatsBomb Open Data (12 men's and women's competitions, 477 matches).

## Objective

Build a complete pipeline, extraction, geometric feature engineering, modeling, interpretability, calibration, and a real-time scoring API, to estimate the goal probability of a shot, drawing on the principles behind professional xG models and DeepMind's TacticAI project.

## Dataset

**Source**: [StatsBomb Open Data](https://github.com/statsbomb/open-data), accessed via the `mplsoccer` library, covering the 12 competitions with 360 data (inventory: `src/inventory_360.py`).

| Competition | Matches | Shots |
|---|---|---|
| Women's World Cup 2023 | 64 | 1608 |
| FIFA World Cup 2022 | 64 | 1430 |
| UEFA Euro 2024 (holdout) | 51 | 1304 |
| UEFA Euro 2020 | 51 | 1234 |
| African Cup of Nations 2023 | 52 | 1161 |
| 1. Bundesliga 2023/2024 | 34 | 908 |
| UEFA Women's Euro 2022 | 31 | 870 |
| UEFA Women's Euro 2025 | 31 | 862 |
| Ligue 1 2022/2023 | 32 | 840 |
| La Liga 2020/2021 | 35 | 826 |
| Ligue 1 2021/2022 | 26 | 672 |
| Major League Soccer 2023 | 6 | 146 |
| **Total** | **477** | **11861** |

- Penalties excluded (fixed position with no defender, an unrepresentative case)
- 3 shots dropped because their freeze frame contains no goalkeeper (empty net): the keeper-related features cannot be computed for them
- Target variable `is_goal`: 0 = no goal, 1 = goal
- Strong class imbalance, realistic for this kind of problem: around 90% missed shots, 10% goals

Each shot is linked to a freeze frame (the position of every visible player at the moment of the shot), which makes it possible to compute spatial features inspired by TacticAI (DeepMind x Liverpool FC), without relying on a graph neural network.

## Project pipeline

```
StatsBomb Open Data (via mplsoccer)
      |
      v extract_data.py    (extracting shots and freeze frames, excluding penalties)
data/raw/shots.csv + freeze.csv
      |
      v features.py         (geometric feature engineering, encoding)
data/processed/shots_features.csv
      |
      v split.py            (train / test / holdout split, shared by every script)
      |
      v train_logistic.py / train_rf.py / train_knn.py
reports/                     (ROC curves, SHAP plots, calibration curve)
      |
      v api/main.py          (FastAPI real-time scoring endpoint)
```

## Feature engineering: the geometry of a shot

From the shot location and its associated freeze frame:

| Feature | Calculation |
|---|---|
| `distance_to_goal` | Euclidean distance between the shot location and the center of the goal (fixed point) |
| `shot_angle` | Angle at which the shooter sees the width of the goal (trigonometry on both posts) |
| `defenders_in_cone` | Number of outfield opponents inside the shooting cone (shooter, left post, right post) |
| `distance_to_keeper` | Euclidean distance between the shooter and the opposing goalkeeper (variable position) |
| `keeper_lateral_offset` | Perpendicular distance (in yards) between the goalkeeper and the bisector of the shooting angle (shooter, left post, right post) |

`distance_to_goal` and `distance_to_keeper` capture two complementary, non-redundant things: the first is always relative to a fixed point on the pitch, the second is relative to the keeper's actual position at the moment of the shot, which varies depending on whether he stayed on his line or rushed out to meet the shooter. It is in fact the single most important feature according to the SHAP ranking.

`keeper_lateral_offset` complements `distance_to_keeper` along the lateral axis: two keepers at the same distance from the shooter can be either perfectly centred in the shooting angle, or shifted towards one post and leaving one side of the goal open. The bisector of the shooting angle is the keeper's theoretical optimal position; the feature measures the deviation from that line (0 = well positioned). In the data, the mean offset is 1.08 yards on goals versus 0.60 yards on non-goals (correlation +0.15 with the target). The effect is mostly non-linear (concentrated on large offsets, keeper beaten or out of position): when it was added (5x5 cross-validation, on the 2022 World Cup alone), the feature slightly improved the Random Forest (AUC 0.775 -> 0.778), was neutral for logistic regression, and hurt KNN (0.80 -> 0.75), which therefore keeps its original four features.

The `shot_statsbomb_xg` column, provided by StatsBomb, was deliberately excluded from the features: it is already a prediction from another model, using it would have introduced a data leak of the same kind encountered in an earlier project (credit scoring). It is, however, used as a reference in the benchmark.

### Features and bookkeeping columns

`features.py` explicitly defines two lists:

- `FEATURES`: the 24 variables allowed into `X`, and the only ones
- `ID_COLUMNS` (`match_id`, `competition`): kept in `shots_features.csv` only to organise the split, they never enter the model

## Exploratory analysis: key findings

- The geometric features dominate every other variable (shot technique, body part used, play context) by a wide margin in terms of correlation with the target
- `shot_angle` (correlation +0.28), `distance_to_keeper` (-0.23) and `distance_to_goal` (-0.22) are the three strongest signals
- Goals are, on average, taken from a wider angle, a shorter distance to goal and to the keeper, and with fewer defenders in the shooting cone than missed shots
- Context variables (body part, technique, type of play) show weak discriminative power on their own

## Modeling

### Data split

The three training scripts (and the notebook) get their data from a single call to `load_splits` (`src/split.py`), so they all use exactly the same sets:

| Set | Content | Matches | Shots | Goal rate |
|---|---|---|---|---|
| Train | 80% of matches outside Euro 2024 | 340 | 8434 | 10.1% |
| Test | 20% of matches outside Euro 2024 | 86 | 2123 | 10.7% |
| Holdout | Euro 2024, in full | 51 | 1304 | 7.5% |

- **Split by match**: all shots from a given match land on the same side. A shot-by-shot split would leave shots from the same match (same teams, same keepers, same context) in both train and test, and inflate the scores.
- **Holdout by competition**: Euro 2024 is set aside entirely, to measure generalisation to a competition the models have never seen. It is used only by the benchmark, never to train or tune a model.

### Results

Three models were trained and compared on this same split (scores on the test set):

| Model | Features used | ROC-AUC | Recall (goal, threshold 0.5) |
|---|---|---|---|
| Logistic Regression | All (24) | 0.792 | 0.10 |
| Random Forest (tuned) | All (24) | 0.787 | 0.04 |
| KNN | 4 geometric features only | 0.739 | 0.11 (0.27 at threshold 0.3) |

No model uses class weighting (`class_weight`): an xG model must produce a probability faithful to the real goal frequency, not maximise recall at a given threshold. With 10% goals, predicted probabilities rarely exceed 0.5, hence the low recall at that threshold for all three models, which does not reflect poor shot ranking (ROC-AUC). The 0.5 threshold has no particular meaning for an xG model.

### A deliberate methodological choice for KNN

With all 24 features (including the encoded categorical variables), KNN produced noticeably weaker results (AUC around 0.65 versus 0.80 with 4 features, measured on the 2022 World Cup alone), a direct illustration of the curse of dimensionality: the geometric distance computed by the algorithm ends up dominated by noise from sparse categorical columns rather than by the variables that actually matter. KNN is therefore restricted to the four continuous numerical variables with a genuine notion of distance. It nonetheless trails the other two models on the current dataset (AUC 0.739 versus 0.79). Logistic regression and Random Forest, by contrast, handle the full 24 features well thanks to their different internal mechanisms (weighted coefficients, successive splits).

The KNN script also reports results at a decision threshold lowered to 0.3: recall on the goal class rises from 0.11 to 0.27, a concrete illustration that the choice of decision threshold is not neutral on a heavily imbalanced problem.

## Interpretability (SHAP)

A SHAP analysis (`notebooks/01_edashap.ipynb`) explains the Random Forest's individual predictions. The notebook does not retrain any model: it loads `models/rf_calibrated.pkl`, the very model that `train_rf.py` saves. A single source of truth: the script trains and saves, the notebook explains.

This model is a `CalibratedClassifierCV`, i.e. 5 Random Forests (one per cross-validation fold) followed by a sigmoid calibration. `TreeExplainer` cannot read the calibration step: SHAP values are computed on each of the 5 internal Random Forests, then averaged. They therefore explain the score before the sigmoid, which is monotonic and does not change how they read.

The results agree with `feature_importances_` and the correlations from the exploratory analysis: `distance_to_keeper`, `shot_angle`, `defenders_in_cone`, `distance_to_goal` and `keeper_lateral_offset` dominate the context variables by a wide margin.

## Probability calibration

An xG model should produce a score that is interpretable as a genuine percentage, not just a good ranking tool. A classifier can separate the two classes very well (good ROC-AUC) while producing probabilities that do not reflect the real goal frequency.

The Random Forest is calibrated using `CalibratedClassifierCV` (sigmoid method, 5-fold cross-validation). The Brier score measures the gap between predicted probabilities and actual outcomes (lower is better):

| | Brier score | ROC-AUC |
|---|---|---|
| Before calibration | 0.0829 | 0.7871 |
| After calibration | 0.0832 | 0.7877 |

Reading this figure requires a reference. The trivial model predicts the same probability for every shot, equal to the goal rate of the training set:

| Prediction (test set) | Brier score |
|---|---|
| Constant 0.101, training goal rate (trivial baseline) | 0.0955 |
| Random Forest | **0.0829** |

The Random Forest reduces the error by 13% compared to this baseline (Brier skill score of 0.13). On such an imbalanced problem the baseline is already low: a Brier score close to 0.08 only means something when compared to that 0.0955. Note that the baseline must be computed on the same set as the model: on the full dataset (9.9% goal rate), the same constant would give 0.089, which is not comparable.

In the current version, calibration changes almost nothing: the Random Forest, trained without class weighting, already produces faithful probabilities. Calibration is kept as a safeguard, and the calibration curve (`reports/calibration_curve.png`) lets you check this visually.

### History: why class weighting was dropped

A first version, trained on the 2022 World Cup alone with `class_weight='balanced'`, illustrated the problem. Weighting artificially inflates the probability of goals to compensate for the imbalance, and the model strongly overestimated its probabilities in the mid-to-high range:

| `class_weight='balanced'` version | Brier score | ROC-AUC |
|---|---|---|
| Before calibration | 0.1381 | 0.7934 |
| After calibration | **0.0798** | 0.7948 |

Calibration then nearly halved the Brier score, with no loss of ranking power. Rather than correcting after the fact a bias introduced on purpose, the weighting was removed: calibration no longer has anything to fix.

### Empirical check on an individual case (`class_weight='balanced'` version)

For a test shot (distance to goal of 8, angle of 1.2, zero defenders, keeper 5 units away), the uncalibrated model predicted a 75.3% chance of scoring, a figure that seemed intuitively plausible at first glance. Rather than trusting that intuition, the 58 genuinely comparable shots in the dataset were isolated to check the actual observed conversion rate: 39.7%. The calibrated model, on that same case, predicts 37.45%, far closer to the real-world ground truth than the uncalibrated model was. This check confirmed that calibration was correcting a real overconfidence in the weighted model rather than introducing an error.

## Benchmark: comparison with StatsBomb xG

`src/benchmark.py` loads the models saved in `models/` and compares them, on the test set and then on the holdout, against two references: a constant (the training goal rate) and the xG computed by StatsBomb (`shot_statsbomb_xg`). That column is kept in `shots_features.csv` only as a reference (`REFERENCE_COLUMNS`): it is another model's prediction and never enters `X`.

| Model | ROC-AUC | Brier | LogLoss |
|---|---|---|---|
| Constant (training goal rate) | - | 0.0955 | 0.3403 |
| **StatsBomb xG** | **0.828** | **0.0774** | **0.2664** |
| Logistic Regression | 0.792 | 0.0812 | 0.2822 |
| Calibrated Random Forest | 0.788 | 0.0832 | 0.2888 |
| KNN | 0.739 | 0.0871 | 0.8576 |

### How much of the signal is captured?

StatsBomb does better, as expected. The real question is: by how much?

```
Brier:  constant 0.0955  ->  Random Forest 0.0832  ->  StatsBomb 0.0774
```

The Random Forest captures 68% of the improvement StatsBomb achieves over the trivial baseline, roughly two thirds of the signal. Logistic regression captures 79%. These models only have the shot geometry and a few context variables (body part, technique, pressure, type of play), without the richer information available to StatsBomb's model.

The remaining gap (AUC 0.828 versus 0.792) therefore does not come from body part, technique or pressure, which our models already know. It comes from information present in the raw StatsBomb data but not yet used, listed in the next steps.

Note: on this benchmark, logistic regression beats the Random Forest on all three metrics, and holds up better on the Euro 2024 holdout. It is therefore the model served by the API.

### The KNN case: good ranking, unusable probabilities

KNN reaches an AUC of 0.739: it ranks shots correctly. But its LogLoss of 0.86 is worse than a plain constant (0.34). Its probability is the share of goals among the 15 nearest neighbours: when none of them is a goal, it is exactly 0. That is the case for 826 of the 2123 test shots, 37 of which are actually goals. A goal predicted at 0% incurs a near-infinite LogLoss penalty. The Brier score, which caps the penalty at 1 per shot, largely hides this. KNN's probabilities are therefore not usable as xG.

### Euro 2024 holdout: a competition never seen before

The same benchmark, on the 1304 Euro 2024 shots (actual goal rate 7.5%):

| Model | ROC-AUC | Brier | LogLoss | Mean xG |
|---|---|---|---|---|
| Constant (training goal rate) | - | 0.0702 | 0.2706 | 0.101 |
| **StatsBomb xG** | **0.787** | **0.0615** | **0.2251** | 0.085 |
| Logistic regression | 0.766 | 0.0635 | 0.2330 | 0.087 |
| Calibrated Random Forest | 0.737 | 0.0652 | 0.2426 | 0.090 |
| KNN | 0.634 | 0.0732 | 0.8841 | 0.085 |

At Euro 2024, 98 goals were scored against 111 expected according to StatsBomb xG. The gap amounts to 1.4 standard deviations and is therefore not significant. This is in fact a well-known property of xG: over a single tournament, finishing varies a lot, which is precisely what makes xG useful over a season rather than over a single match.

## Scoring API (FastAPI)

Logistic regression is exposed through a FastAPI application (`api/main.py`), with a `POST /predict` endpoint that takes a shot's geometric characteristics as input and returns the goal probability. It is the best of the three models on both the test set and the holdout, and its probabilities are already faithful without calibration (mean xG of 0.100 on the test set against a training goal rate of 0.101). FastAPI automatically generates interactive documentation at `/docs`, allowing the API to be tested without writing any code.

## Acknowledged limitation of the model

The features used are purely geometric (position of the shot, the keeper, the defenders). The model captures nothing about the shooter's individual execution quality (shot power, accuracy, strong foot), a known limitation of public xG models, which have no access to biometric or shot-speed data.

## Project structure

```
xg-shot-prediction/
├── api/
│   └── main.py                  # FastAPI scoring endpoint
├── models/
│   ├── rf_calibrated.pkl        # Calibrated Random Forest (explained by the notebook)
│   ├── logistic.pkl             # Logistic regression + scaler (served by the API)
│   └── knn.pkl                  # KNN + scaler (not versioned)
├── data/
│   ├── raw/                     # Extracted shots and freeze frames
│   └── processed/                # Final features
├── notebooks/
│   └── 01_edashap.ipynb          # Exploratory analysis and SHAP on models/rf_calibrated.pkl
├── src/
│   ├── inventory_360.py          # Inventory of StatsBomb competitions with 360 data
│   ├── extract_data.py           # StatsBomb extraction via mplsoccer
│   ├── features.py                # Geometric feature engineering, FEATURES and ID_COLUMNS lists
│   ├── split.py                   # Train / test / holdout split, shared by every script
│   ├── train_logistic.py          # Logistic regression, saving the model for the API
│   ├── train_rf.py                # Random Forest, calibration, saving the model for the notebook
│   ├── train_knn.py
│   └── benchmark.py               # Comparison of saved models and StatsBomb xG
├── reports/                      # ROC curves, SHAP plots, calibration curve
├── requirements.txt
└── README.md
```

## Setup and usage

```bash
pip install -r requirements.txt

python src/extract_data.py
python src/features.py
python src/train_logistic.py
python src/train_rf.py
python src/train_knn.py
python src/benchmark.py

uvicorn api.main:app --reload
# Interactive docs available at http://127.0.0.1:8000/docs
```

## Next steps

- Use information present in the raw StatsBomb data but not yet exploited, candidates to explain the gap with their xG:
  - the key pass: cross (`pass_cross`), cut-back (`pass_cut_back`), pass height (`pass_height_name`), via `shot_key_pass_id`
  - shot context annotated by StatsBomb: one-on-one with the keeper (`shot_one_on_one`), open goal (`shot_open_goal`), shot after a dribble (`shot_follows_dribble`), aerial duel won (`aerial_won`)
  - deflected or redirected shots (`shot_deflected`, `shot_redirect`)
- Analysis of how mixing men's / women's and club / international football affects predictions
- Exploring a TacticAI-inspired version using a graph neural network (GNN), once the basics of deep learning are in place

## Tech stack

Python, Pandas, Scikit-learn, mplsoccer, SHAP, Matplotlib, FastAPI, Uvicorn, Git