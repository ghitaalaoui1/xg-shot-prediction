"""Entrainement d'un XGBoost sur les memes donnees que les autres modeles."""

import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve
from xgboost import XGBClassifier

from features import FEATURES, TARGET
from split import load_splits, split_by_match

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE_DIR, "data", "processed", "shots_features.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost.pkl")
ROC_PATH = os.path.join(BASE_DIR, "reports", "roc_curve_xgboost.png")


def main():
    df = pd.read_csv(DATA)
    train, test, _ = load_splits(df)

    # L'arret precoce exige son propre jeu : le test doit rester vierge
    train_fit, val = split_by_match(train, test_size=0.15)
    print(f"Entrainement {len(train_fit)} tirs, validation {len(val)}")

    model = XGBClassifier(
        max_depth=3,
        learning_rate=0.05,
        n_estimators=2000,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        eval_metric="logloss",
        early_stopping_rounds=50,
        random_state=42,
    )
    model.fit(
        train_fit[FEATURES],
        train_fit[TARGET],
        eval_set=[(val[FEATURES], val[TARGET])],
        verbose=False,
    )
    print(f"Arret apres {model.best_iteration} arbres sur 2000")

    y = test[TARGET]
    proba = model.predict_proba(test[FEATURES])[:, 1]
    print(f"AUC      {roc_auc_score(y, proba):.4f}")
    print(f"Brier    {brier_score_loss(y, proba):.4f}")
    print(f"LogLoss  {log_loss(y, proba):.4f}")
    print(f"xG moyen {proba.mean():.4f}  (taux reel {y.mean():.4f})")

    fpr, tpr, _ = roc_curve(y, proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"XGBoost (AUC = {roc_auc_score(y, proba):.2f})", color="blue")
    plt.plot([0, 1], [0, 1], "k--", label="Hasard (AUC = 0.5)")
    plt.xlabel("Taux de Faux Positifs")
    plt.ylabel("Taux de Vrais Positifs")
    plt.title("Courbe ROC - Modele xG (XGBoost)")
    plt.legend(loc="lower right")
    os.makedirs(os.path.dirname(ROC_PATH), exist_ok=True)
    plt.savefig(ROC_PATH)
    print(f"Courbe ROC sauvegardee dans : {ROC_PATH}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "scaler": None}, MODEL_PATH)
    print(f"Modele sauvegarde dans : {MODEL_PATH}")


if __name__ == "__main__":
    main()