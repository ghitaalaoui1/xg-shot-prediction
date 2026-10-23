"""Compare tous les modeles sur le jeu de test et le holdout, plus le xG StatsBomb."""

import os

import joblib
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from features import TARGET
from split import load_splits

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE_DIR, "data", "processed", "shots_features.csv")
MODELS = {
    "Regression logistique": "logistic.pkl",
    "Random Forest calibre": "rf_calibrated.pkl",
    "KNN": "knn.pkl",
    "XGBoost": "xgboost.pkl",
}


def predict(artifact, df):
    X = df[artifact["features"]]
    if artifact["scaler"] is not None:
        X = artifact["scaler"].transform(X)
    return artifact["model"].predict_proba(X)[:, 1]


def evaluate(titre, df, base_rate, artifacts):
    """Affiche le tableau des metriques de tous les modeles sur un jeu donne."""
    y = df[TARGET]

    rows = []
    rows.append(("Constante (taux de base)", pd.Series(base_rate, index=y.index)))
    rows.append(("xG StatsBomb", df["shot_statsbomb_xg"]))
    for nom, artifact in artifacts.items():
        rows.append((nom, predict(artifact, df)))

    print(f"\n=== {titre} : {len(df)} tirs, taux de but reel {y.mean():.3f} ===")
    print(f"{'Modele':26s} {'AUC':>7s} {'Brier':>9s} {'LogLoss':>9s} {'xG moyen':>9s}")
    for nom, proba in rows:
        auc = roc_auc_score(y, proba) if pd.Series(proba).nunique() > 1 else float("nan")
        print(f"{nom:26s} {auc:7.3f} {brier_score_loss(y, proba):9.4f} "
              f"{log_loss(y, proba):9.4f} {pd.Series(proba).mean():9.3f}")


def main():
    df = pd.read_csv(DATA)
    train, test, holdout = load_splits(df)

    # Taux de but du train : la reference triviale ne doit connaitre ni le test ni le holdout
    base_rate = train[TARGET].mean()
    artifacts = {
        nom: joblib.load(os.path.join(BASE_DIR, "models", fichier))
        for nom, fichier in MODELS.items()
    }

    evaluate("Test", test, base_rate, artifacts)
    evaluate("Holdout Euro 2024", holdout, base_rate, artifacts)


if __name__ == "__main__":
    main()