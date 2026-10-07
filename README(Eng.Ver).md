# xG Shot Prediction: Goal Probability Model

End-to-end data science project: predicting the probability that a football shot results in a goal (xG model), using StatsBomb Open Data from the 2022 World Cup.

## Objective

Build a complete pipeline, extraction, geometric feature engineering, modeling, interpretability, calibration, and a real-time scoring API, to estimate the goal probability of a shot, drawing on the principles behind professional xG models and DeepMind's TacticAI project.

## Dataset

**Source**: [StatsBomb Open Data](https://github.com/statsbomb/open-data), accessed via the `mplsoccer` library, covering the 2022 World Cup (64 matches).

- 1430 shots extracted (penalties excluded, as they come from a fixed position with no defender, an unrepresentative case)
- Target variable `is_goal`: 0 = no goal, 1 = goal
- Strong class imbalance, realistic for this kind of problem: around 89% missed shots, 11% goals

Each shot is linked to a freeze frame (the position of every visible player at the moment of the shot), which makes it possible to compute spatial features inspired by TacticAI (DeepMind x Liverpool FC), without relying on a graph neural network.

## Project pipeline

```
StatsBomb Open Data (via mplsoccer)
      |
      v extract_data.py    (extracting shots and freeze frames, excluding penalties)
data/raw/shots_worldcup2022.csv + freeze_worldcup2022.csv
      |
      v features.py         (geometric feature engineering, encoding)
data/processed/shots_features.csv
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

`distance_to_goal` and `distance_to_keeper` capture two complementary, non-redundant things: the first is always relative to a fixed point on the pitch, the second is relative to the keeper's actual position at the moment of the shot, which varies depending on whether he stayed on his line or rushed out to meet the shooter. It is in fact the single most important feature according to the SHAP ranking.

The `shot_statsbomb_xg` column, provided by StatsBomb, was deliberately excluded from the features: it is already a prediction from another model, using it would have introduced a data leak of the same kind encountered in an earlier project (credit scoring).

## Exploratory analysis: key findings

- The four geometric features dominate every other variable (shot technique, body part used, play context) by a wide margin in terms of correlation with the target
- `shot_angle` (correlation +0.31) and `distance_to_keeper` (-0.27) are the two strongest signals
- Goals are, on average, taken from a wider angle, a shorter distance to goal and to the keeper, and with fewer defenders in the shooting cone than missed shots
- Context variables (body part, technique, type of play) show weak discriminative power on their own

## Modeling

Three models were trained and compared on the same train/test split (80/20, stratified, `random_state=42`):

| Model | Features used | ROC-AUC | Recall (goal, threshold 0.5) |
|---|---|---|---|
| Logistic Regression | All (23) | 0.788 | 0.67 |
| Random Forest (tuned) | All (23) | 0.791 | 0.63 |
| KNN | 4 geometric features only | 0.800 | 0.07 (0.33 at threshold 0.3) |

### A deliberate methodological choice for KNN

With all 23 features (including the encoded categorical variables), KNN produced noticeably weaker results (AUC around 0.65), a direct illustration of the curse of dimensionality: the geometric distance computed by the algorithm ends up dominated by noise from sparse categorical columns rather than by the variables that actually matter. By restricting KNN to the four continuous numerical variables with a genuine notion of distance, its performance surpasses the other two models on ROC-AUC. Logistic regression and Random Forest, by contrast, handle the full 23 features well thanks to their different internal mechanisms (weighted coefficients, successive splits).

Unlike the other two models, KNN has no `class_weight` parameter to compensate for class imbalance. At the default decision threshold (0.5), its recall on the goal class therefore remains very low, even though it separates the two classes well in terms of probability (good ROC-AUC). Lowering the decision threshold to 0.3 noticeably improves this recall, a concrete illustration that the choice of decision threshold is not neutral on a heavily imbalanced problem.

## Interpretability (SHAP)

A SHAP analysis was carried out on the Random Forest to explain individual predictions. The results confirm the ranking obtained via `feature_importances_` and the correlations from the exploratory analysis: `distance_to_keeper`, `shot_angle`, `distance_to_goal` and `defenders_in_cone` dominate the context variables by a wide margin.

## Probability calibration

A binary classifier like the Random Forest, especially one trained with `class_weight='balanced'` to compensate for class imbalance, produces probabilities that separate the two classes well (good ROC-AUC) but do not always reflect a true real-world frequency. An xG model should produce a score that is interpretable as a genuine percentage, not just a good ranking tool.

The model was calibrated using `CalibratedClassifierCV` (sigmoid method, 5-fold cross-validation):

| | Brier score | ROC-AUC |
|---|---|---|
| Before calibration | 0.1437 | 0.7906 |
| After calibration | **0.0786** | 0.7952 |

The Brier score, which measures the gap between predicted probabilities and actual outcomes (lower is better), is nearly cut in half after calibration, with no loss of ranking power, ROC-AUC even improves slightly. The calibration curve visually confirms that the uncalibrated model strongly overestimated its probabilities in the mid-to-high range, while the calibrated model closely tracks the perfect calibration diagonal.

### Empirical check on an individual case

For a test shot (distance to goal of 8, angle of 1.2, zero defenders, keeper 5 units away), the uncalibrated model predicted a 75.3% chance of scoring, a figure that seemed intuitively plausible at first glance. Rather than trusting that intuition, the 58 genuinely comparable shots in the dataset were isolated to check the actual observed conversion rate: 39.7%. The calibrated model, on that same case, predicts 37.45%, far closer to the real-world ground truth than the uncalibrated model was. This check confirms that calibration corrects a real overconfidence in the raw model rather than introducing an error.

## Scoring API (FastAPI)

The calibrated Random Forest model is exposed through a FastAPI application (`api/main.py`), with a `POST /predict` endpoint that takes a shot's geometric characteristics as input and returns the calibrated goal probability. FastAPI automatically generates interactive documentation at `/docs`, allowing the API to be tested without writing any code.

## Acknowledged limitation of the model

The features used are purely geometric (position of the shot, the keeper, the defenders). The model captures nothing about the shooter's individual execution quality (shot power, accuracy, strong foot), a known limitation of public xG models, which have no access to biometric or shot-speed data.

## Project structure

```
xg-shot-prediction/
├── api/
│   ├── main.py                  # FastAPI scoring endpoint
│   └── model.pkl                # Calibrated Random Forest, saved
├── data/
│   ├── raw/                     # Extracted shots and freeze frames
│   └── processed/                # Final features
├── notebooks/
│   └── 01_edashap.ipynb          # Exploratory analysis and SHAP interpretability
├── src/
│   ├── extract_data.py           # StatsBomb extraction via mplsoccer
│   ├── features.py                # Geometric feature engineering
│   ├── train_logistic.py
│   ├── train_rf.py                # Random Forest, calibration, saving the model for the API
│   └── train_knn.py
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

uvicorn api.main:app --reload
# Interactive docs available at http://127.0.0.1:8000/docs
```

## Next steps

- Additional feature on the keeper's lateral positioning (keeper-to-trajectory angle), not just the raw distance
- Extension to other StatsBomb competitions (Women's Euro 2022, Women's World Cup 2023) to increase data volume
- Exploring a TacticAI-inspired version using a graph neural network (GNN), once the basics of deep learning are in place

## Tech stack

Python, Pandas, Scikit-learn, mplsoccer, SHAP, Matplotlib, FastAPI, Uvicorn, Git