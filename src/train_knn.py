import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "shots_features.csv")


def train_and_evaluate_knn(data_path):
    if not os.path.exists(data_path):
        print(f"Erreur : Fichier introuvable -> {data_path}")
        return

    df = pd.read_csv(data_path)
    print("Donnees chargees pour la modelisation.")

    target_column = "is_goal"

    # Memes features que pour la regression logistique et le Random Forest (comparaison equitable)
    features_to_use = ['distance_to_goal', 'shot_angle', 'defenders_in_cone', 'distance_to_keeper']
    X = df[features_to_use]
    y = df[target_column]
# KNN souffre de la malediction de la dimensionnalite sur des features categorielles encodees.
# On ne garde que les 4 variables numeriques continues avec un vrai sens geometrique de distance,
# contrairement a la regression logistique et au Random Forest qui utilisent toutes les features.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nEntrainement du KNN...")
    model = KNeighborsClassifier(n_neighbors=15, weights='distance')
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    print("\n--- Rapport de Classification (seuil 0.5) ---")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("--- Matrice de Confusion (seuil 0.5) ---")
    print(f"                 Predit: Pas but   Predit: But")
    print(f"Reel: Pas but         {cm[0,0]:>4}            {cm[0,1]:>4}")
    print(f"Reel: But             {cm[1,0]:>4}            {cm[1,1]:>4}")

    # Test avec un seuil de decision abaisse
    y_pred_ajuste = (y_prob >= 0.3).astype(int)
    print("\n--- Rapport de Classification (seuil 0.3) ---")
    print(classification_report(y_test, y_pred_ajuste))

    cm_ajuste = confusion_matrix(y_test, y_pred_ajuste)
    print("--- Matrice de Confusion (seuil 0.3) ---")
    print(f"                 Predit: Pas but   Predit: But")
    print(f"Reel: Pas but         {cm_ajuste[0,0]:>4}            {cm_ajuste[0,1]:>4}")
    print(f"Reel: But             {cm_ajuste[1,0]:>4}            {cm_ajuste[1,1]:>4}")

    auc_score = roc_auc_score(y_test, y_prob)
    print(f"\nScore AUC-ROC : {auc_score:.4f}")

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"KNN (AUC = {auc_score:.2f})", color="orange")
    plt.plot([0, 1], [0, 1], "k--", label="Hasard (AUC = 0.5)")
    plt.xlabel("Taux de Faux Positifs")
    plt.ylabel("Taux de Vrais Positifs")
    plt.title("Courbe ROC - Modele xG (KNN)")
    plt.legend(loc="lower right")

    os.makedirs(os.path.join(BASE_DIR, "reports"), exist_ok=True)
    roc_path = os.path.join(BASE_DIR, "reports", "roc_curve_knn.png")
    plt.savefig(roc_path)
    print(f"\nCourbe ROC sauvegardee dans : {roc_path}")


if __name__ == "__main__":
    train_and_evaluate_knn(FEATURES_PATH)