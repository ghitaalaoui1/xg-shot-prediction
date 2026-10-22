"""Compare tous les modeles sur le meme jeu de test, plus le xG StatsBomb."""

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
}


def predict(artifact, df):
    X = df[artifact["features"]]
    if artifact["scaler"] is not None:
        X = artifact["scaler"].transform(X)
    return artifact["model"].predict_proba(X)[:, 1]


def main():
    df = pd.read_csv(DATA)
    train, test, _ = load_splits(df)
    y = test[TARGET]

    rows = []
    # Taux de but du train : la reference triviale ne doit pas connaitre le test
    base_rate = train[TARGET].mean()
    rows.append(("Constante (taux de base)", pd.Series(base_rate, index=y.index)))
    rows.append(("xG StatsBomb", test["shot_statsbomb_xg"]))

    for nom, fichier in MODELS.items():
        artifact = joblib.load(os.path.join(BASE_DIR, "models", fichier))
        rows.append((nom, predict(artifact, test)))

    print(f"{'Modele':26s} {'AUC':>7s} {'Brier':>9s} {'LogLoss':>9s}")
    for nom, proba in rows:
        auc = roc_auc_score(y, proba) if pd.Series(proba).nunique() > 1 else float("nan")
        print(f"{nom:26s} {auc:7.3f} {brier_score_loss(y, proba):9.4f} "
              f"{log_loss(y, proba):9.4f}")


if __name__ == "__main__":
    main()