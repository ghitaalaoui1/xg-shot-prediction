import os
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
import joblib
import pandas as pd
from features import FEATURES
from split import load_splits
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "shots_features.csv")


def train_and_evaluate_rf(data_path):
    if not os.path.exists(data_path):
        print(f"Erreur : Fichier introuvable -> {data_path}")
        return

    df = pd.read_csv(data_path)
    print("Donnees chargees pour la modelisation.")

    target_column = "is_goal"

    train, test, holdout = load_splits(df)
    X_train, y_train = train[FEATURES], train[target_column]
    X_test, y_test = test[FEATURES], test[target_column]

    # Pas de standardisation necessaire pour Random Forest
    print("\nEntrainement du Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n--- Rapport de Classification (Random Forest) ---")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("\n--- Matrice de Confusion ---")
    print(f"                 Predit: Pas but   Predit: But")
    print(f"Reel: Pas but         {cm[0,0]:>4}            {cm[0,1]:>4}")
    print(f"Reel: But             {cm[1,0]:>4}            {cm[1,1]:>4}")

    auc_score = roc_auc_score(y_test, y_prob)
    print(f"\nScore AUC-ROC : {auc_score:.4f}")
    # Calibration des probabilites
    calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=5)
    calibrated_model.fit(X_train, y_train)
    y_prob_calibrated = calibrated_model.predict_proba(X_test)[:, 1]

    brier_avant = brier_score_loss(y_test, y_prob)
    brier_apres = brier_score_loss(y_test, y_prob_calibrated)
    print(f"\nBrier score avant calibration : {brier_avant:.4f}")
    print(f"Brier score apres calibration : {brier_apres:.4f}")

    auc_calibre = roc_auc_score(y_test, y_prob_calibrated)
    print(f"AUC-ROC apres calibration : {auc_calibre:.4f}")

    prob_true_avant, prob_pred_avant = calibration_curve(y_test, y_prob, n_bins=10)
    prob_true_apres, prob_pred_apres = calibration_curve(y_test, y_prob_calibrated, n_bins=10)

    plt.figure(figsize=(7, 7))
    plt.plot(prob_pred_avant, prob_true_avant, marker='o', label='Avant calibration', color='orange')
    plt.plot(prob_pred_apres, prob_true_apres, marker='o', label='Apres calibration', color='green')
    plt.plot([0, 1], [0, 1], 'k--', label='Calibration parfaite')
    plt.xlabel('Probabilite moyenne predite')
    plt.ylabel('Frequence reelle observee')
    plt.title('Courbe de calibration du modele xG')
    plt.legend()
    calib_path = os.path.join(BASE_DIR, "reports", "calibration_curve.png")
    plt.savefig(calib_path)
    print(f"Courbe de calibration sauvegardee dans : {calib_path}")

    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\n--- Top 10 variables les plus importantes ---")
    print(importances.head(10))

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"Random Forest (AUC = {auc_score:.2f})", color="green")
    plt.plot([0, 1], [0, 1], "k--", label="Hasard (AUC = 0.5)")
    plt.xlabel("Taux de Faux Positifs")
    plt.ylabel("Taux de Vrais Positifs")
    plt.title("Courbe ROC - Modele xG (Random Forest)")
    plt.legend(loc="lower right")

    os.makedirs(os.path.join(BASE_DIR, "reports"), exist_ok=True)
    roc_path = os.path.join(BASE_DIR, "reports", "roc_curve_rf.png")
    plt.savefig(roc_path)
    print(f"\nCourbe ROC sauvegardee dans : {roc_path}")
    # Sauvegarde du modele final (calibre) avec ses metadonnees
    artifact = {"model": calibrated_model, "features": FEATURES, "scaler": None}
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    path = os.path.join(BASE_DIR, "models", "rf_calibrated.pkl")
    joblib.dump(artifact, path)
    print(f"\nModele calibre sauvegarde dans : {path}")

if __name__ == "__main__":
    train_and_evaluate_rf(FEATURES_PATH)
