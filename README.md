# xG Shot Prediction : Modele de probabilite de but

Projet de data science de bout en bout : prediction de la probabilite qu'un tir au football finisse au fond des filets (modele xG), a partir des donnees StatsBomb Open Data de la Coupe du Monde 2022.

## Objectif

Construire un pipeline complet, extraction, feature engineering geometrique, modelisation, interpretabilite, calibration, et une API de scoring en temps reel, pour estimer la probabilite de but d'un tir, en s'inspirant des principes derriere les modeles xG professionnels et le projet TacticAI de DeepMind.

## Dataset

**Source** : [StatsBomb Open Data](https://github.com/statsbomb/open-data), via la librairie `mplsoccer`, sur la Coupe du Monde 2022 (64 matchs).

- 1430 tirs extraits (penaltys exclus, position fixe sans defenseur, cas non representatif)
- Variable cible `is_goal` : 0 = pas but, 1 = but
- Fort desequilibre de classes, realiste pour ce type de probleme : environ 89% de tirs non convertis, 11% de buts

Chaque tir est associe a un freeze frame (position de tous les joueurs visibles au moment du tir), ce qui permet de calculer des features spatiales inspirees de TacticAI (DeepMind x Liverpool FC), sans recourir a un reseau de neurones sur graphe.

## Pipeline du projet

```
StatsBomb Open Data (via mplsoccer)
      |
      v extract_data.py    (extraction des tirs et des freeze frames, exclusion des penaltys)
data/raw/shots_worldcup2022.csv + freeze_worldcup2022.csv
      |
      v features.py         (feature engineering geometrique, encodage)
data/processed/shots_features.csv
      |
      v train_logistic.py / train_rf.py / train_knn.py
reports/                     (courbes ROC, graphiques SHAP, courbe de calibration)
      |
      v api/main.py          (API FastAPI de scoring en temps reel)
```

## Feature engineering : la geometrie du tir

A partir de la position du tir et du freeze frame associe :

| Feature | Calcul |
|---|---|
| `distance_to_goal` | Distance euclidienne entre le point de tir et le centre du but (point fixe) |
| `shot_angle` | Angle sous lequel le tireur voit la largeur du but (trigonometrie sur les deux poteaux) |
| `defenders_in_cone` | Nombre d'adversaires de champ a l'interieur du cone de tir (tireur, poteau gauche, poteau droit) |
| `distance_to_keeper` | Distance euclidienne entre le tireur et le gardien adverse (position variable) |
| `keeper_lateral_offset` | Distance perpendiculaire (en yards) entre le gardien et la bissectrice de l'angle de tir (tireur, poteau gauche, poteau droit) |

`distance_to_goal` et `distance_to_keeper` mesurent deux choses complementaires et non redondantes : la premiere est toujours relative a un point fixe du terrain, la seconde est relative a la position reelle du gardien au moment du tir, qui varie selon qu'il est reste sur sa ligne ou qu'il est sorti a la rencontre du tireur. C'est d'ailleurs la feature la plus importante du modele d'apres le classement SHAP.

`keeper_lateral_offset` complete `distance_to_keeper` sur l'axe lateral : deux gardiens a la meme distance du tireur peuvent etre l'un parfaitement centre dans l'angle, l'autre decale vers un poteau et laissant un cote du but ouvert. La bissectrice de l'angle de tir correspond a la position theorique ideale du gardien ; la feature mesure l'ecart a cette ligne (0 = gardien bien place). Sur les donnees, le decalage moyen est de 0.91 yard sur les buts contre 0.58 yard sur les tirs non convertis (correlation +0.14 avec la cible). L'effet est surtout non lineaire (il se concentre sur les gros decalages, gardien battu ou hors de position) : en validation croisee 5x5, la feature ameliore legerement le Random Forest (AUC 0.775 -> 0.778) mais reste neutre pour la regression logistique, et degrade le KNN (0.80 -> 0.75), qui conserve donc ses quatre features d'origine.

La colonne `shot_statsbomb_xg`, fournie par StatsBomb, a volontairement ete exclue des features : c'est deja une prediction d'un autre modele, l'utiliser aurait constitue une fuite de donnees du meme type que celle rencontree sur un projet precedent (credit scoring).

## Analyse exploratoire : principaux enseignements

- Les quatre features geometriques dominent tres largement toutes les autres variables (technique de frappe, pied utilise, contexte de jeu) en termes de correlation avec la cible
- `shot_angle` (correlation +0.31) et `distance_to_keeper` (-0.27) sont les deux signaux les plus forts
- Les buts sont tires en moyenne avec un angle plus large, une distance au but et au gardien plus faible, et moins de defenseurs dans le cone de tir que les tirs non convertis
- Les variables de contexte (pied utilise, technique, type d'action) ont un pouvoir discriminant faible pris isolement

## Modelisation

Trois modeles ont ete entraines et compares sur le meme split train/test (80/20, stratifie, `random_state=42`) :

| Modele | Features utilisees | ROC-AUC | Recall (but, seuil 0.5) |
|---|---|---|---|
| Regression Logistique | Toutes (24) | 0.785 | 0.67 |
| Random Forest (tune) | Toutes (24) | 0.793 | 0.60 |
| KNN | 4 features geometriques uniquement | 0.800 | 0.07 (0.33 a seuil 0.3) |

### Un choix methodologique assume pour le KNN

Avec les 24 features (incluant les variables categorielles encodees), le KNN obtenait des resultats nettement plus faibles (AUC autour de 0.65), une illustration directe de la malediction de la dimensionnalite : la distance geometrique calculee par l'algorithme devient dominee par le bruit des colonnes categorielles rares plutot que par les variables realement pertinentes. En limitant le KNN aux quatre variables numeriques continues qui ont un vrai sens de distance, ses performances depassent celles des deux autres modeles en ROC-AUC. La regression logistique et le Random Forest, eux, tolerent bien les 24 features grace a leurs mecanismes internes differents (coefficients ponderes, separations successives).

Le KNN, contrairement aux deux autres modeles, n'a pas de parametre `class_weight` pour compenser le desequilibre des classes. A seuil de decision par defaut (0.5), son recall sur la classe but reste donc tres faible, meme s'il separe bien les deux classes en termes de probabilite (bon ROC-AUC). Abaisser le seuil de decision a 0.3 ameliore nettement ce recall, une illustration concrete que le choix du seuil de decision n'est pas neutre sur un probleme fortement desequilibre.

## Interpretabilite (SHAP)

Une analyse SHAP a ete menee sur le Random Forest pour expliquer les predictions individuelles. Les resultats confirment le classement d'importance obtenu via `feature_importances_` et les correlations de l'analyse exploratoire : `distance_to_keeper`, `shot_angle`, `distance_to_goal` et `defenders_in_cone` dominent tres largement les variables de contexte.

## Calibration des probabilites

Un modele de classification binaire comme le Random Forest, surtout entraine avec `class_weight='balanced'` pour compenser le desequilibre des classes, produit des probabilites qui separent bien les deux classes (bon ROC-AUC) mais qui ne refletent pas toujours une vraie frequence reelle. Un modele xG doit justement produire un score interpretable comme un vrai pourcentage, pas seulement un bon outil de classement.

Le modele a ete calibre avec `CalibratedClassifierCV` (methode sigmoide, validation croisee a 5 plis) :

| | Brier score | ROC-AUC |
|---|---|---|
| Avant calibration | 0.1381 | 0.7934 |
| Apres calibration | **0.0798** | 0.7948 |

Le Brier score, qui mesure l'ecart entre les probabilites predites et les resultats reels (plus bas est meilleur), est quasiment divise par deux apres calibration, sans perte de pouvoir de classement (le ROC-AUC s'ameliore meme legerement). La courbe de calibration confirme visuellement que le modele non calibre surestimait fortement ses probabilites dans la zone moyenne a elevee, alors que le modele calibre suit de pres la diagonale de calibration parfaite.

### Verification empirique sur un cas individuel

Sur un tir test (distance au but de 8, angle de 1.2, zero defenseur, gardien a 5 unites), le modele non calibre predisait 75.3% de chance de but, un chiffre qui semblait intuitivement coherent au premier abord. Plutot que de faire confiance a cette intuition, les 58 tirs reellement comparables du dataset ont ete isoles pour verifier le vrai taux de conversion observe : 39.7%. Le modele calibre, sur ce meme cas, predit 37.45%, nettement plus proche de la realite terrain que le modele non calibre. Cette verification confirme que la calibration corrige une surconfiance reelle du modele brut plutot que d'introduire une erreur.

## API de scoring (FastAPI)

Le modele Random Forest calibre est expose via une API FastAPI (`api/main.py`), avec un endpoint `POST /predict` qui prend en entree les caracteristiques geometriques d'un tir et retourne la probabilite de but calibree. Une documentation interactive est generee automatiquement par FastAPI sur `/docs`, permettant de tester l'API sans ecrire de code.

## Limite assumee du modele

Les features utilisees sont purement geometriques (position du tir, du gardien, des defenseurs). Le modele ne capture rien de la qualite d'execution individuelle du tireur (puissance de frappe, precision, pied fort), une limite connue des modeles xG publics qui n'ont pas acces a des donnees biometriques ou de vitesse de frappe.

## Structure du projet

```
xg-shot-prediction/
├── api/
│   ├── main.py                  # API FastAPI de scoring
│   └── model.pkl                # Random Forest calibre, sauvegarde
├── data/
│   ├── raw/                     # Tirs et freeze frames extraits
│   └── processed/                # Features finales
├── notebooks/
│   └── 01_edashap.ipynb          # Analyse exploratoire et interpretabilite SHAP
├── src/
│   ├── extract_data.py           # Extraction StatsBomb via mplsoccer
│   ├── features.py                # Feature engineering geometrique
│   ├── train_logistic.py
│   ├── train_rf.py                # Random Forest, calibration, sauvegarde du modele pour l'API
│   └── train_knn.py
├── reports/                      # Courbes ROC, graphiques SHAP, courbe de calibration
├── requirements.txt
└── README.md
```

## Installation et execution

```bash
pip install -r requirements.txt

python src/extract_data.py
python src/features.py
python src/train_logistic.py
python src/train_rf.py
python src/train_knn.py

uvicorn api.main:app --reload
# Documentation interactive disponible sur http://127.0.0.1:8000/docs
```

## Prochaines etapes

- Feature additionnelle sur le positionnement lateral du gardien (angle gardien-trajectoire), pas seulement la distance brute
- Extension a d'autres competitions StatsBomb (Euro feminin 2022, Coupe du Monde feminine 2023) pour augmenter le volume de donnees
- Exploration d'une version inspiree de TacticAI avec un reseau de neurones sur graphe (GNN), une fois les bases du deep learning acquises

## Stack technique

Python, Pandas, Scikit-learn, mplsoccer, SHAP, Matplotlib, FastAPI, Uvicorn, Git