# xG Shot Prediction: Goal Probability Model

End-to-end data science project: predicting the probability that a football shot results in a goal (xG model), using StatsBomb Open Data from the 2022 World Cup.

## Objective

Build a complete pipeline, extraction, geometric feature engineering, modeling, interpretability, to estimate the goal probability of a shot, drawing on the principles behind professional xG models and DeepMind's TacticAI project.

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
reports/                     (ROC curves, SHAP plots)
```

## Feature engineering: the geometry of a shot

From the shot location and its associated freeze frame:

| Feature | Calculation |
|---|---|
| `distance_to_goal` | Euclidean distance between the shot location and the center of the goal |
| `shot_angle` | Angle at which the shooter sees the width of the goal (trigonometry on both posts) |
| `defenders_in_cone` | Number of outfield opponents inside the shooting cone (shooter, left post, right post) |
| `distance_to_keeper` | Euclidean distance between the shooter and the opposing goalkeeper |

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

## Acknowledged limitation of the model

The features used are purely geometric (position of the shot, the keeper, the defenders). The model captures nothing about the shooter's individual execution quality (shot power, accuracy, strong foot), a known limitation of public xG models, which have no access to biometric or shot-speed data.

## Project structure

```
xg-shot-prediction/
├── api/ 
    ├──__pycache__/
    ├──main.py
    └──model.pkl                     
├── data/
│   ├── raw/                    # Extracted shots and freeze frames
│   └── processed/              # Final features
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   └── 02_shap_explainability.ipynb
├── src/
│   ├── extract_data.py          # StatsBomb extraction via mplsoccer
│   ├── features.py               # Geometric feature engineering
│   ├── train_logistic.py
│   ├── train_rf.py
│   └── train_knn.py
├── reports/                    # ROC curves, SHAP plots
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
```

## Next steps

- Probability calibration, so the output score approaches a genuinely interpretable xG percentage
- Extension to other StatsBomb competitions (Women's Euro 2022, Women's World Cup 2023) to increase data volume
- Exploring a TacticAI-inspired version using a graph neural network (GNN), once the basics of deep learning are in place
- FastAPI for real-time scoring

## Tech stack

Python, Pandas, Scikit-learn, mplsoccer, SHAP, Matplotlib, FastAPI, Uvicorn,Git