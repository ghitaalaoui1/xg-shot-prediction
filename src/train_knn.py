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
    features_to_use = ["distance", "angle", "x", "y"]
    
    X = df[features_to_use]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardisation obligatoire pour le KNN (les échelles de distance et d'angle diffèrent)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nEntrainement du KNN...")
    model = KNeighborsClassifier(n_neighbors=15, weights='distance')
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    print("\n--- Rapport de Classification (KNN) ---")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("\n--- Matrice de Confusion -")
    print(cm)
if __name__ == "__main__":
    train_and_evaluate_knn(FEATURES_PATH)