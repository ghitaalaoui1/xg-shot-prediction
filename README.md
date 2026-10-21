# xG Shot Prediction : Modele de probabilite de but

Projet de data science de bout en bout : prediction de la probabilite qu'un tir au football finisse au fond des filets (modele xG), a partir des donnees StatsBomb Open Data (12 competitions masculines et feminines, 477 matchs).

## Objectif

Construire un pipeline complet, extraction, feature engineering geometrique, modelisation, interpretabilite, calibration, et une API de scoring en temps reel, pour estimer la probabilite de but d'un tir, en s'inspirant des principes derriere les modeles xG professionnels et le projet TacticAI de DeepMind.

## Dataset

**Source** : [StatsBomb Open Data](https://github.com/statsbomb/open-data), via la librairie `mplsoccer`, sur les 12 competitions disposant de donnees 360 (inventaire : `src/inventory_360.py`).

| Competition | Matchs | Tirs |
|---|---|---|
| Women's World Cup 2023 | 64 | 1608 |
| FIFA World Cup 2022 | 64 | 1430 |
| UEFA Euro 2024 (holdout) | 51 | 1304 |
| UEFA Euro 2020 | 51 | 1234 |
| African Cup of Nations 2023 | 52 | 1161 |
| 1. Bundesliga 2023/2024 | 34 | 908 |
| UEFA Women's Euro 2022 | 31 | 870 |
| UEFA Women's Euro 2025 | 31 | 862 |
| Ligue 1 2022/2023 | 32 | 840 |
| La Liga 2020/2021 | 35 | 826 |
| Ligue 1 2021/2022 | 26 | 672 |
| Major League Soccer 2023 | 6 | 146 |
| **Total** | **477** | **11861** |

- Penaltys exclus (position fixe sans defenseur, cas non representatif)
- 3 tirs ecartes car leur freeze frame ne contient pas de gardien (but vide) : les features liees au gardien n'y sont pas calculables
- Variable cible `is_goal` : 0 = pas but, 1 = but
- Fort desequilibre de classes, realiste pour ce type de probleme : environ 90% de tirs non convertis, 10% de buts

Chaque tir est associe a un freeze frame (position de tous les joueurs visibles au moment du tir), ce qui permet de calculer des features spatiales inspirees de TacticAI (DeepMind x Liverpool FC), sans recourir a un reseau de neurones sur graphe.

## Pipeline du projet

```
StatsBomb Open Data (via mplsoccer)
      |
      v extract_data.py    (extraction des tirs et des freeze frames, exclusion des penaltys)
data/raw/shots.csv + freeze.csv
      |
      v features.py         (feature engineering geometrique, encodage)
data/processed/shots_features.csv
      |
      v split.py            (decoupage train / test / holdout, commun a tous les scripts)
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

`keeper_lateral_offset` complete `distance_to_keeper` sur l'axe lateral : deux gardiens a la meme distance du tireur peuvent etre l'un parfaitement centre dans l'angle, l'autre decale vers un poteau et laissant un cote du but ouvert. La bissectrice de l'angle de tir correspond a la position theorique ideale du gardien ; la feature mesure l'ecart a cette ligne (0 = gardien bien place). Sur les donnees, le decalage moyen est de 1.08 yard sur les buts contre 0.60 yard sur les tirs non convertis (correlation +0.15 avec la cible). L'effet est surtout non lineaire (il se concentre sur les gros decalages, gardien battu ou hors de position) : lors de son ajout (validation croisee 5x5, sur la seule Coupe du Monde 2022), la feature ameliorait legerement le Random Forest (AUC 0.775 -> 0.778), restait neutre pour la regression logistique et degradait le KNN (0.80 -> 0.75), qui conserve donc ses quatre features d'origine.

La colonne `shot_statsbomb_xg`, fournie par StatsBomb, a volontairement ete exclue des features : c'est deja une prediction d'un autre modele, l'utiliser aurait constitue une fuite de donnees du meme type que celle rencontree sur un projet precedent (credit scoring). Elle sert en revanche de reference dans le benchmark.

### Features et colonnes d'organisation

`features.py` definit explicitement deux listes :

- `FEATURES` : les 24 variables autorisees a entrer dans `X`, et les seules
- `ID_COLUMNS` (`match_id`, `competition`) : conservees dans `shots_features.csv` uniquement pour organiser le decoupage, elles n'entrent jamais dans le modele

## Analyse exploratoire : principaux enseignements

- Les features geometriques dominent tres largement toutes les autres variables (technique de frappe, pied utilise, contexte de jeu) en termes de correlation avec la cible
- `shot_angle` (correlation +0.28), `distance_to_keeper` (-0.23) et `distance_to_goal` (-0.22) sont les trois signaux les plus forts
- Les buts sont tires en moyenne avec un angle plus large, une distance au but et au gardien plus faible, et moins de defenseurs dans le cone de tir que les tirs non convertis
- Les variables de contexte (pied utilise, technique, type d'action) ont un pouvoir discriminant faible pris isolement

## Modelisation

### Decoupage des donnees

Les trois scripts d'entrainement (et le notebook) obtiennent leurs donnees par un unique appel a `load_splits` (`src/split.py`), et utilisent donc exactement les memes jeux :

| Jeu | Contenu | Matchs | Tirs | Taux de but |
|---|---|---|---|---|
| Train | 80% des matchs hors Euro 2024 | 340 | 8434 | 10.1% |
| Test | 20% des matchs hors Euro 2024 | 86 | 2123 | 10.7% |
| Holdout | Euro 2024, en entier | 51 | 1304 | 7.5% |

- **Decoupage par match** : tous les tirs d'un meme match sont du meme cote. Un decoupage tir par tir laisserait des tirs d'un meme match (memes equipes, memes gardiens, meme contexte) a la fois dans le train et le test, et gonflerait les scores.
- **Holdout par competition** : l'Euro 2024 est entierement mis de cote, pour mesurer la generalisation a une competition jamais vue. Il n'est pas encore utilise pour l'evaluation.

### Resultats

Trois modeles ont ete entraines et compares sur ce meme decoupage (scores sur le jeu de test) :

| Modele | Features utilisees | ROC-AUC | Recall (but, seuil 0.5) |
|---|---|---|---|
| Regression Logistique | Toutes (24) | 0.792 | 0.10 |
| Random Forest (tune) | Toutes (24) | 0.787 | 0.04 |
| KNN | 4 features geometriques uniquement | 0.739 | 0.11 (0.27 a seuil 0.3) |

Aucun modele n'utilise de ponderation des classes (`class_weight`) : un modele xG doit produire une probabilite fidele a la frequence reelle des buts, pas maximiser le recall a un seuil donne. Avec 10% de buts, les probabilites predites depassent rarement 0.5, d'ou un recall faible a ce seuil pour les trois modeles, sans que cela traduise un mauvais classement des tirs (ROC-AUC). Le seuil de 0.5 n'a pas de sens particulier pour un modele xG.

### Un choix methodologique assume pour le KNN

Avec les 24 features (incluant les variables categorielles encodees), le KNN obtenait des resultats nettement plus faibles (AUC autour de 0.65 contre 0.80 avec 4 features, mesure sur la seule Coupe du Monde 2022), une illustration directe de la malediction de la dimensionnalite : la distance geometrique calculee par l'algorithme devient dominee par le bruit des colonnes categorielles rares plutot que par les variables realement pertinentes. Le KNN est donc limite aux quatre variables numeriques continues qui ont un vrai sens de distance. Il reste neanmoins en retrait des deux autres modeles sur le jeu de donnees actuel (AUC 0.739 contre 0.79). La regression logistique et le Random Forest, eux, tolerent bien les 24 features grace a leurs mecanismes internes differents (coefficients ponderes, separations successives).

Le script KNN affiche aussi les resultats a un seuil de decision abaisse a 0.3 : le recall sur la classe but passe de 0.11 a 0.27, une illustration concrete que le choix du seuil de decision n'est pas neutre sur un probleme fortement desequilibre.

## Interpretabilite (SHAP)

Une analyse SHAP (`notebooks/01_edashap.ipynb`) explique les predictions individuelles du Random Forest. Le notebook ne reentraine aucun modele : il charge `models/rf_calibrated.pkl`, celui-la meme que `train_rf.py` sauvegarde et que l'API sert. Une seule source de verite : le script entraine et sauvegarde, le notebook explique, l'API sert.

Ce modele est un `CalibratedClassifierCV`, c'est-a-dire 5 Random Forest (un par pli de validation croisee) suivis d'une sigmoide de calibration. `TreeExplainer` ne sait pas lire la calibration : les valeurs SHAP sont calculees sur chacun des 5 Random Forest internes, puis moyennees. Elles expliquent donc le score avant la sigmoide, qui est monotone et ne change pas le sens de lecture.

Les resultats concordent avec les `feature_importances_` et les correlations de l'analyse exploratoire : `distance_to_keeper`, `shot_angle`, `defenders_in_cone`, `distance_to_goal` et `keeper_lateral_offset` dominent tres largement les variables de contexte.

## Calibration des probabilites

Un modele xG doit produire un score interpretable comme un vrai pourcentage, pas seulement un bon outil de classement. Un classifieur peut tres bien separer les deux classes (bon ROC-AUC) tout en produisant des probabilites qui ne refletent pas la frequence reelle des buts.

Le Random Forest est calibre avec `CalibratedClassifierCV` (methode sigmoide, validation croisee a 5 plis). Le Brier score mesure l'ecart entre les probabilites predites et les resultats reels (plus bas est meilleur) :

| | Brier score | ROC-AUC |
|---|---|---|
| Avant calibration | 0.0829 | 0.7871 |
| Apres calibration | 0.0832 | 0.7877 |

Pour lire ce chiffre, il faut une reference. Le modele trivial predit la meme probabilite pour tous les tirs, egale au taux de but du jeu d'entrainement :

| Prediction (jeu de test) | Brier score |
|---|---|
| Constante a 0.101, taux de but du train (reference triviale) | 0.0955 |
| Random Forest | **0.0829** |

Le Random Forest reduit l'erreur de 13% par rapport a cette reference (Brier skill score de 0.13). Sur un probleme aussi desequilibre, la reference est deja basse : un Brier proche de 0.08 n'a de sens que compare a ce 0.0955. Attention, la reference doit etre calculee sur le meme jeu que le modele : sur l'ensemble des donnees (taux de but de 9.9%), la meme constante donnerait 0.089, un chiffre non comparable.

Sur la version actuelle, la calibration ne change presque rien : le Random Forest, entraine sans ponderation des classes, produit deja des probabilites fideles. Elle est conservee comme garde-fou, et la courbe de calibration (`reports/calibration_curve.png`) permet de le verifier visuellement.

### Historique : pourquoi la ponderation des classes a ete abandonnee

Une premiere version, entrainee sur la seule Coupe du Monde 2022 avec `class_weight='balanced'`, illustrait le probleme. La ponderation gonfle artificiellement la probabilite des buts pour compenser le desequilibre, et le modele surestimait fortement ses probabilites dans la zone moyenne a elevee :

| Version `class_weight='balanced'` | Brier score | ROC-AUC |
|---|---|---|
| Avant calibration | 0.1381 | 0.7934 |
| Apres calibration | **0.0798** | 0.7948 |

La calibration divisait alors le Brier score presque par deux, sans perte de pouvoir de classement. Plutot que de corriger apres coup un biais introduit volontairement, la ponderation a ete retiree : la calibration n'a plus rien a rattraper.

### Verification empirique sur un cas individuel (version `class_weight='balanced'`)

Sur un tir test (distance au but de 8, angle de 1.2, zero defenseur, gardien a 5 unites), le modele non calibre predisait 75.3% de chance de but, un chiffre qui semblait intuitivement coherent au premier abord. Plutot que de faire confiance a cette intuition, les 58 tirs reellement comparables du dataset ont ete isoles pour verifier le vrai taux de conversion observe : 39.7%. Le modele calibre, sur ce meme cas, predit 37.45%, nettement plus proche de la realite terrain que le modele non calibre. Cette verification confirmait que la calibration corrigeait une surconfiance reelle du modele pondere plutot que d'introduire une erreur.

## Benchmark : comparaison avec le xG StatsBomb

`src/benchmark.py` charge les modeles sauvegardes dans `models/` et les compare, sur le meme jeu de test, a deux references : une constante (le taux de but du train) et le xG calcule par StatsBomb (`shot_statsbomb_xg`). Cette colonne est conservee dans `shots_features.csv` uniquement comme reference (`REFERENCE_COLUMNS`) : c'est la prediction d'un autre modele, elle n'entre jamais dans `X`.

| Modele | ROC-AUC | Brier | LogLoss |
|---|---|---|---|
| Constante (taux de but du train) | - | 0.0955 | 0.3403 |
| **xG StatsBomb** | **0.828** | **0.0774** | **0.2664** |
| Regression logistique | 0.792 | 0.0812 | 0.2822 |
| Random Forest calibre | 0.788 | 0.0832 | 0.2888 |
| KNN | 0.739 | 0.0871 | 0.8576 |

### Quelle part du signal est capturee ?

StatsBomb fait mieux, c'etait attendu. La vraie question est : de combien ?

```
Brier :  constante 0.0955  ->  Random Forest 0.0832  ->  StatsBomb 0.0774
```

Le Random Forest capture 68% de l'amelioration que StatsBomb obtient sur la reference triviale, soit environ les deux tiers du signal. La regression logistique en capture 79%. Ces modeles n'ont que la geometrie du tir et quelques variables de contexte (partie du corps, technique, pression, type d'action), sans l'information riche dont dispose le modele de StatsBomb.

L'ecart restant (AUC 0.828 contre 0.792) ne vient donc pas de la partie du corps, de la technique ou de la pression, que nos modeles connaissent deja. Il vient d'informations presentes dans les donnees brutes StatsBomb mais pas encore exploitees, listees dans les prochaines etapes.

A noter : sur ce benchmark, la regression logistique fait mieux que le Random Forest sur les trois metriques, alors que c'est le Random Forest que sert l'API.

### Le cas du KNN : bon classement, probabilites inutilisables

Le KNN atteint une AUC de 0.739 : il classe correctement les tirs. Mais son LogLoss de 0.86 est pire que celui d'une simple constante (0.34). Sa probabilite est la proportion de buts parmi les 15 plus proches voisins : quand aucun d'eux n'est un but, elle vaut exactement 0. C'est le cas de 826 tirs du test sur 2123, dont 37 sont pourtant des buts. Un but predit a 0% coute une penalite quasi infinie en LogLoss. Le Brier, qui borne la penalite a 1 par tir, le masque en grande partie. Les probabilites du KNN ne sont donc pas utilisables comme xG.

## API de scoring (FastAPI)

Le modele Random Forest calibre est expose via une API FastAPI (`api/main.py`), avec un endpoint `POST /predict` qui prend en entree les caracteristiques geometriques d'un tir et retourne la probabilite de but calibree. Une documentation interactive est generee automatiquement par FastAPI sur `/docs`, permettant de tester l'API sans ecrire de code.

## Limite assumee du modele

Les features utilisees sont purement geometriques (position du tir, du gardien, des defenseurs). Le modele ne capture rien de la qualite d'execution individuelle du tireur (puissance de frappe, precision, pied fort), une limite connue des modeles xG publics qui n'ont pas acces a des donnees biometriques ou de vitesse de frappe.

## Structure du projet

```
xg-shot-prediction/
├── api/
│   └── main.py                  # API FastAPI de scoring
├── models/
│   ├── rf_calibrated.pkl        # Random Forest calibre (servi par l'API, explique par le notebook)
│   ├── logistic.pkl             # Regression logistique + scaler (non versionne)
│   └── knn.pkl                  # KNN + scaler (non versionne)
├── data/
│   ├── raw/                     # Tirs et freeze frames extraits
│   └── processed/                # Features finales
├── notebooks/
│   └── 01_edashap.ipynb          # Analyse exploratoire et SHAP sur models/rf_calibrated.pkl
├── src/
│   ├── inventory_360.py          # Inventaire des competitions StatsBomb avec donnees 360
│   ├── extract_data.py           # Extraction StatsBomb via mplsoccer
│   ├── features.py                # Feature engineering geometrique, listes FEATURES et ID_COLUMNS
│   ├── split.py                   # Decoupage train / test / holdout, commun a tous les scripts
│   ├── train_logistic.py
│   ├── train_rf.py                # Random Forest, calibration, sauvegarde du modele pour l'API
│   ├── train_knn.py
│   └── benchmark.py               # Comparaison des modeles sauvegardes et du xG StatsBomb
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
python src/benchmark.py

uvicorn api.main:app --reload
# Documentation interactive disponible sur http://127.0.0.1:8000/docs
```

## Prochaines etapes

- Evaluation finale des trois modeles sur le holdout Euro 2024
- Exploiter les informations presentes dans les donnees brutes StatsBomb mais pas encore utilisees, candidates pour expliquer l'ecart avec leur xG :
  - la passe decisive : centre (`pass_cross`), passe en retrait (`pass_cut_back`), hauteur de la passe (`pass_height_name`), via `shot_key_pass_id`
  - le contexte du tir annote par StatsBomb : face-a-face avec le gardien (`shot_one_on_one`), but vide (`shot_open_goal`), tir apres un dribble (`shot_follows_dribble`), duel aerien gagne (`aerial_won`)
  - les tirs devies ou reorientes (`shot_deflected`, `shot_redirect`)
- Analyse de l'effet du melange football masculin / feminin et clubs / selections sur les predictions
- Exploration d'une version inspiree de TacticAI avec un reseau de neurones sur graphe (GNN), une fois les bases du deep learning acquises

## Stack technique

Python, Pandas, Scikit-learn, mplsoccer, SHAP, Matplotlib, FastAPI, Uvicorn, Git