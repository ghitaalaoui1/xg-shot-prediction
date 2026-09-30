import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pas de standardisation necessaire pour Random Forest
    print("\nEntrainement du Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        class_weight="balanced"
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

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
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


if __name__ == "__main__":
    train_and_evaluate_rf(FEATURES_PATH)