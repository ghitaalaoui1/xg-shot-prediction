"""Bootstrap par match des ecarts de metriques entre XGBoost et la regression logistique."""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from benchmark import BASE_DIR, DATA, MODELS, predict
from features import TARGET
from split import SEED, load_splits

N_BOOT = 2000
MODEL_A = "XGBoost"
MODEL_B = "Regression logistique"
# (fonction, True si plus haut = meilleur)
METRICS = {
    "AUC": (roc_auc_score, True),
    "Brier": (brier_score_loss, False),
    "LogLoss": (log_loss, False),
}


def bootstrap_diffs(df, proba_a, proba_b, n_boot=N_BOOT, seed=SEED):
    """Tire des matchs avec remise : les tirs d'un meme match ne sont pas independants."""
    y = df[TARGET].to_numpy()
    groups = [np.flatnonzero(m) for m in pd.get_dummies(df["match_id"]).to_numpy().T]
    rng = np.random.default_rng(seed)

    diffs = {nom: [] for nom in METRICS}
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if y[idx].min() == y[idx].max():
            continue
        # Echantillon apparie : les deux modeles sont evalues sur les memes tirs
        for nom, (metric, _) in METRICS.items():
            diffs[nom].append(metric(y[idx], proba_a[idx]) - metric(y[idx], proba_b[idx]))
    return {nom: np.array(d) for nom, d in diffs.items()}


def report(titre, df, artifacts):
    y = df[TARGET]
    proba_a = predict(artifacts[MODEL_A], df)
    proba_b = predict(artifacts[MODEL_B], df)
    diffs = bootstrap_diffs(df, proba_a, proba_b)

    print(f"\n=== {titre} : {df['match_id'].nunique()} matchs, {len(df)} tirs ===")
    print(f"Ecart = {MODEL_A} - {MODEL_B}")
    print(f"{'Metrique':9s} {'Ecart':>9s} {'IC 95% bootstrap':>22s} {MODEL_A + ' devant':>16s}")
    for nom, (metric, higher_is_better) in METRICS.items():
        observed = metric(y, proba_a) - metric(y, proba_b)
        low, high = np.percentile(diffs[nom], [2.5, 97.5])
        a_better = (diffs[nom] > 0) if higher_is_better else (diffs[nom] < 0)
        print(f"{nom:9s} {observed:+9.4f}   [{low:+.4f}, {high:+.4f}] {a_better.mean():15.1%}")


def main():
    df = pd.read_csv(DATA)
    _, test, holdout = load_splits(df)
    artifacts = {
        nom: joblib.load(os.path.join(BASE_DIR, "models", MODELS[nom]))
        for nom in (MODEL_A, MODEL_B)
    }

    # Le test sert a choisir ; le holdout n'est affiche qu'a titre de constat
    report("Test", test, artifacts)
    report("Holdout Euro 2024", holdout, artifacts)


if __name__ == "__main__":
    main()
