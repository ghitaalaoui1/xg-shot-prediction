import os
import joblib
import pandas as pd
from features import FEATURES
from split import load_splits
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "shots_features.csv")


def train_and_evaluate_model(data_path):
    if not os.path.exists(data_path):
        print(f"Erreur : Fichier introuvable -> {data_path}")
        return

    df = pd.read_csv(data_path)
    print("Donnees chargees pour la modelisation.")

    target_column = "is_goal"

    train, test, holdout = load_splits(df)
    X_train, y_train = train[FEATURES], train[target_column]
    X_test, y_test = test[FEATURES], test[target_column]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nEntrainement de la Regression Logistique...")
    model = LogisticRegression(max_iter=5000, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    print("\n--- Rapport de Classification ---")
    print(classification_report(y_test, y_pred))

    auc_score = roc_auc_score(y_test, y_prob)
    print(f"Score AUC-ROC : {auc_score:.4f}")

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"Regression Logistique (AUC = {auc_score:.2f})", color="blue")
    plt.plot([0, 1], [0, 1], "k--", label="Hasard (AUC = 0.5)")
    plt.xlabel("Taux de Faux Positifs")
    plt.ylabel("Taux de Vrais Positifs")
    plt.title("Courbe ROC - Modele xG (Regression Logistique)")
    plt.legend(loc="lower right")

    os.makedirs(os.path.join(BASE_DIR, "reports"), exist_ok=True)
    roc_path = os.path.join(BASE_DIR, "reports", "roc_curve_logistic.png")
    plt.savefig(roc_path)
    print(f"\nCourbe ROC sauvegardee dans : {roc_path}")

    from sklearn.metrics import confusion_matrix
    print("\n--- Rapport de Classification ---")
    print(classification_report(y_test, y_pred))

    # <-- Ajoutez le bloc ICI, toujours a l'interieur de la fonction
    cm = confusion_matrix(y_test, y_pred)
    print("\n--- Matrice de Confusion ---")
    print(f"                 Predit: Pas but   Predit: But")
    print(f"Reel: Pas but         {cm[0,0]:>4}            {cm[0,1]:>4}")
    print(f"Reel: But             {cm[1,0]:>4}            {cm[1,1]:>4}")

    auc_score = roc_auc_score(y_test, y_prob)
    print(f"Score AUC-ROC : {auc_score:.4f}")

    # Sauvegarde du modele avec ses metadonnees
    artifact = {"model": model, "features": FEATURES, "scaler": scaler}
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    path = os.path.join(BASE_DIR, "models", "logistic.pkl")
    joblib.dump(artifact, path)
    print(f"\nModele sauvegarde dans : {path}")

if __name__ == "__main__":
    train_and_evaluate_model(FEATURES_PATH)


