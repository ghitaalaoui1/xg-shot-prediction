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
reports/                     (courbes ROC, coefficients, graphiques SHAP, courbe de calibration)
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
| `keeper_off_line` | De combien le gardien est sorti vers le tireur : `distance_to_goal` moins la distance tireur-gardien |
| `keeper_lateral_offset` | Distance perpendiculaire (en yards) entre le gardien et la bissectrice de l'angle de tir (tireur, poteau gauche, poteau droit) |

`distance_to_goal` et `keeper_off_line` mesurent deux notions distinctes : a quelle distance le tireur frappe, et de combien le gardien est sorti de sa ligne a sa rencontre. Elles sont faiblement correlees (0.09). Une premiere version utilisait directement la distance tireur-gardien (`distance_to_keeper`), mais celle-ci etait correlee a 0.96 avec `distance_to_goal` : le gardien etant en general pres de sa ligne, les deux mesuraient presque la meme chose. Elle n'apportait de l'information propre que lorsque le gardien est avance, exactement ce que `keeper_off_line` isole (voir Interpretabilite). La difference de distances est exacte quand le gardien est sur l'axe du tir, et une approximation pour un tir en biais.

`keeper_lateral_offset` complete `keeper_off_line` sur l'axe lateral : deux gardiens sortis de la meme distance peuvent etre l'un parfaitement centre dans l'angle, l'autre decale vers un poteau et laissant un cote du but ouvert. La bissectrice de l'angle de tir correspond a la position theorique ideale du gardien ; la feature mesure l'ecart a cette ligne (0 = gardien bien place). Sur les donnees, le decalage moyen est de 1.08 yard sur les buts contre 0.60 yard sur les tirs non convertis (correlation +0.15 avec la cible). L'effet est surtout non lineaire (il se concentre sur les gros decalages, gardien battu ou hors de position) : lors de son ajout (validation croisee 5x5, sur la seule Coupe du Monde 2022), la feature ameliorait legerement le Random Forest (AUC 0.775 -> 0.778), restait neutre pour la regression logistique et degradait le KNN (0.80 -> 0.75), qui conserve donc ses quatre features d'origine.

La colonne `shot_statsbomb_xg`, fournie par StatsBomb, a volontairement ete exclue des features : c'est deja une prediction d'un autre modele, l'utiliser aurait constitue une fuite de donnees du meme type que celle rencontree sur un projet precedent (credit scoring). Elle sert en revanche de reference dans le benchmark.

### Features et colonnes d'organisation

`features.py` definit explicitement deux listes :

- `FEATURES` : les 24 variables autorisees a entrer dans `X`, et les seules
- `ID_COLUMNS` (`match_id`, `competition`) : conservees dans `shots_features.csv` uniquement pour organiser le decoupage, elles n'entrent jamais dans le modele

## Analyse exploratoire : principaux enseignements

- Les features geometriques dominent tres largement toutes les autres variables (technique de frappe, pied utilise, contexte de jeu) en termes de correlation avec la cible
- `shot_angle` (correlation +0.28), `distance_to_goal` (-0.22) et `keeper_lateral_offset` (+0.15) sont les trois signaux les plus forts
- Les buts sont tires en moyenne avec un angle plus large, une distance au but et au gardien plus faible, et moins de defenseurs dans le cone de tir que les tirs non convertis
- Les variables de contexte (pied utilise, technique, type d'action) ont un pouvoir discriminant faible pris isolement

## Modelisation

### Decoupage des donnees

Les scripts d'entrainement (et le notebook) obtiennent leurs donnees par un unique appel a `load_splits` (`src/split.py`), et utilisent donc exactement les memes jeux :

| Jeu | Contenu | Matchs | Tirs | Taux de but |
|---|---|---|---|---|
| Train | 80% des matchs hors Euro 2024 | 340 | 8434 | 10.1% |
| Test | 20% des matchs hors Euro 2024 | 86 | 2123 | 10.7% |
| Holdout | Euro 2024, en entier | 51 | 1304 | 7.5% |

- **Decoupage par match** : tous les tirs d'un meme match sont du meme cote. Un decoupage tir par tir laisserait des tirs d'un meme match (memes equipes, memes gardiens, meme contexte) a la fois dans le train et le test, et gonflerait les scores.
- **Holdout par competition** : l'Euro 2024 est entierement mis de cote, pour mesurer la generalisation a une competition jamais vue. Il n'est utilise que par le benchmark, jamais pour entrainer ou regler un modele.

### Resultats

Trois modeles ont ete entraines et compares sur ce meme decoupage (scores sur le jeu de test) :

| Modele | Features utilisees | ROC-AUC | Recall (but, seuil 0.5) |
|---|---|---|---|
| Regression Logistique | Toutes (24) | 0.792 | 0.10 |
| Random Forest (tune) | Toutes (24) | 0.784 | 0.04 |
| KNN | 4 features geometriques uniquement | 0.739 | 0.12 (0.32 a seuil 0.3) |

Aucun modele n'utilise de ponderation des classes (`class_weight`) : un modele xG doit produire une probabilite fidele a la frequence reelle des buts, pas maximiser le recall a un seuil donne. Avec 10% de buts, les probabilites predites depassent rarement 0.5, d'ou un recall faible a ce seuil pour les trois modeles, sans que cela traduise un mauvais classement des tirs (ROC-AUC). Le seuil de 0.5 n'a pas de sens particulier pour un modele xG.

### Un choix methodologique assume pour le KNN

Avec les 24 features (incluant les variables categorielles encodees), le KNN obtenait des resultats nettement plus faibles (AUC autour de 0.65 contre 0.80 avec 4 features, mesure sur la seule Coupe du Monde 2022), une illustration directe de la malediction de la dimensionnalite : la distance geometrique calculee par l'algorithme devient dominee par le bruit des colonnes categorielles rares plutot que par les variables realement pertinentes. Le KNN est donc limite aux quatre variables numeriques continues qui ont un vrai sens de distance. Il reste neanmoins en retrait des deux autres modeles sur le jeu de donnees actuel (AUC 0.739 contre 0.78 a 0.79). La regression logistique et le Random Forest, eux, tolerent bien les 24 features grace a leurs mecanismes internes differents (coefficients ponderes, separations successives).

Le script KNN affiche aussi les resultats a un seuil de decision abaisse a 0.3 : le recall sur la classe but passe de 0.12 a 0.32, une illustration concrete que le choix du seuil de decision n'est pas neutre sur un probleme fortement desequilibre.

## Interpretabilite

Le notebook `notebooks/01_edashap.ipynb` ne reentraine aucun modele : il charge ceux que les scripts sauvegardent dans `models/`. Une seule source de verite : le script entraine et sauvegarde, le notebook explique.

### Le modele servi : les coefficients de la regression logistique

Les features etant standardisees, les coefficients sont directement comparables entre eux. Un coefficient de +0.4 signifie qu'un ecart-type de plus sur la variable multiplie la cote de marquer par e^0.4, soit environ 1.5. Le signe donne le sens, la valeur absolue l'importance relative (graphique complet : `reports/logistic_coefficients.png`).

| Feature | Coefficient | Effet d'un ecart-type de plus sur la cote |
|---|---|---|
| `distance_to_goal` | -0.73 | x 0.48 |
| `shot_angle` | +0.53 | x 1.69 |
| `defenders_in_cone` | -0.27 | x 0.77 |
| `keeper_lateral_offset` | +0.25 | x 1.28 |
| `keeper_off_line` | +0.19 | x 1.21 |

Les signes ont un sens footballistique : plus le tir est lointain et plus il y a de defenseurs dans le cone, moins il a de chances, plus l'angle est ouvert et plus le gardien est decale, plus il en a. `keeper_off_line` n'a pas de signe evident a priori : un gardien qui sort reduit l'angle, mais il sort surtout sur les face-a-face, des occasions franches. Le coefficient positif decrit cette association, pas un effet causal de la sortie. Les indicatrices se lisent contre la categorie retiree a l'encodage : un tir du pied (+0.64) convertit mieux qu'une tete, une frappe normale (+0.61) mieux qu'une talonnade.

#### Pourquoi `keeper_off_line` a remplace `distance_to_keeper`

La version precedente utilisait la distance tireur-gardien, correlee a 0.96 avec `distance_to_goal` puisque le gardien est en general pres de sa ligne. Le coefficient de `distance_to_goal` devenait alors instable, au point de changer de signe :

| Variables du modele | Avec `distance_to_keeper` | Avec `keeper_off_line` |
|---|---|---|
| `distance_to_goal` seule | -0.99 | -0.99 |
| + variable gardien | -0.17 | -1.06 |
| + `shot_angle` | **+0.80** | -0.48 |
| les 24 features | -0.01 | -0.73 |

Dans l'ancienne version, la distance au but semblait inutile (-0.01) et `distance_to_keeper` portait tout l'effet distance. `keeper_off_line` ne garde que ce que la position du gardien apporte en plus, et sa correlation avec `distance_to_goal` tombe a 0.09. Pour la regression logistique, c'est la meme information presentee autrement : la performance est identique (AUC 0.792), mais chaque coefficient a maintenant un sens. La valeur bouge encore avec l'angle (correlation -0.75 avec la distance), ce qui est inevitable puisque s'eloigner du but referme l'angle, mais le signe ne change plus. Le choix a ete valide sur le jeu de test, sans consulter le holdout.

### Les deux familles de modeles voient-elles la meme chose ?

Le Random Forest n'est plus le modele servi, mais il reste un point de comparaison precieux. Un modele lineaire et un modele d'arbres apprennent de facons tres differentes : s'ils hierarchisent les variables de la meme facon, ce classement ne tient pas aux choix d'un algorithme particulier. Le Random Forest (`models/rf_calibrated.pkl`) est explique avec SHAP, puis son classement des variables est compare a celui des coefficients de la regression logistique.

Ce modele est un `CalibratedClassifierCV`, c'est-a-dire 5 Random Forest (un par pli de validation croisee) suivis d'une sigmoide de calibration. `TreeExplainer` ne sait pas lire la calibration : les valeurs SHAP sont calculees sur chacun des 5 Random Forest internes, puis moyennees. Elles expliquent donc le score avant la sigmoide, qui est monotone et ne change pas le sens de lecture.

Les resultats concordent avec les `feature_importances_` et les correlations de l'analyse exploratoire : `shot_angle`, `distance_to_goal`, `defenders_in_cone`, `keeper_lateral_offset` et `keeper_off_line` dominent tres largement les variables de contexte.

La correlation de rang (Spearman) entre l'importance SHAP du Random Forest et la valeur absolue des coefficients de la regression logistique est de **0.64** (p=0.001) sur les 24 features, et de **0.90** sur les 5 features geometriques. Un modele lineaire et un modele d'arbres, entraines independamment, hierarchisent donc les variables geometriques de la meme facon, a l'inversion pres des deux premieres : `distance_to_goal` est 1re pour la regression logistique et 2e pour le Random Forest, `shot_angle` l'inverse.

L'accord partiel sur l'ensemble des features s'explique par deux ecarts :
- **Les indicatrices** (partie du corps, technique) : fortes pour la regression logistique (pied gauche 2e, pied droit 3e, technique normale 4e), plus faibles pour le Random Forest. Un coefficient mesure l'effet d'une categorie contre la reference, SHAP la contribution moyenne sur les tirs du test : une indicatrice qui ne separe qu'une minorite de tirs pese peu en moyenne.
- **`shot_first_time`** : 6e pour le Random Forest, 19e pour la regression logistique. Son effet depend probablement du contexte du tir, ce qu'un coefficient unique ne peut pas representer.

Avec l'ancienne variable `distance_to_keeper`, la correlation n'etait que de 0.60 : `distance_to_goal`, rendue instable par la colinearite, etait 22e pour la regression logistique.

## Calibration des probabilites

Un modele xG doit produire un score interpretable comme un vrai pourcentage, pas seulement un bon outil de classement. Un classifieur peut tres bien separer les deux classes (bon ROC-AUC) tout en produisant des probabilites qui ne refletent pas la frequence reelle des buts.

Le Random Forest est calibre avec `CalibratedClassifierCV` (methode sigmoide, validation croisee a 5 plis). Le Brier score mesure l'ecart entre les probabilites predites et les resultats reels (plus bas est meilleur) :

| | Brier score | ROC-AUC |
|---|---|---|
| Avant calibration | 0.0836 | 0.7838 |
| Apres calibration | 0.0840 | 0.7826 |

Pour lire ce chiffre, il faut une reference. Le modele trivial predit la meme probabilite pour tous les tirs, egale au taux de but du jeu d'entrainement :

| Prediction (jeu de test) | Brier score |
|---|---|
| Constante a 0.101, taux de but du train (reference triviale) | 0.0955 |
| Random Forest | **0.0836** |

Le Random Forest reduit l'erreur de 12% par rapport a cette reference (Brier skill score de 0.12). Sur un probleme aussi desequilibre, la reference est deja basse : un Brier proche de 0.08 n'a de sens que compare a ce 0.0955. Attention, la reference doit etre calculee sur le meme jeu que le modele : sur l'ensemble des donnees (taux de but de 9.9%), la meme constante donnerait 0.089, un chiffre non comparable.

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

`src/benchmark.py` charge les modeles sauvegardes dans `models/` et les compare, sur le jeu de test puis sur le holdout, a deux references : une constante (le taux de but du train) et le xG calcule par StatsBomb (`shot_statsbomb_xg`). Cette colonne est conservee dans `shots_features.csv` uniquement comme reference (`REFERENCE_COLUMNS`) : c'est la prediction d'un autre modele, elle n'entre jamais dans `X`.

| Modele | ROC-AUC | Brier | LogLoss |
|---|---|---|---|
| Constante (taux de but du train) | - | 0.0955 | 0.3403 |
| **xG StatsBomb** | **0.828** | **0.0774** | **0.2664** |
| Regression logistique | 0.792 | 0.0812 | 0.2822 |
| Random Forest calibre | 0.783 | 0.0840 | 0.2916 |
| KNN | 0.739 | 0.0860 | 0.7951 |
| XGBoost | 0.795 | 0.0796 | 0.2780 |

### Quelle part du signal est capturee ?

StatsBomb fait mieux, c'etait attendu. La vraie question est : de combien ?

```
Brier :  constante 0.0955  ->  Random Forest 0.0840  ->  StatsBomb 0.0774
```

Le Random Forest capture 64% de l'amelioration que StatsBomb obtient sur la reference triviale, soit pres des deux tiers du signal. La regression logistique en capture 79%. Ces modeles n'ont que la geometrie du tir et quelques variables de contexte (partie du corps, technique, pression, type d'action), sans l'information riche dont dispose le modele de StatsBomb.

L'ecart restant (AUC 0.828 contre 0.792) ne vient donc pas de la partie du corps, de la technique ou de la pression, que nos modeles connaissent deja. Il vient d'informations presentes dans les donnees brutes StatsBomb mais pas encore exploitees, listees dans les pistes pour aller plus loin.

A noter : sur ce benchmark, la regression logistique fait mieux que le Random Forest sur les trois metriques. XGBoost fait un peu mieux qu'elle sur le test (88% du signal capture), un ecart discute plus bas : il n'est pas distinguable du bruit.

### Le cas du KNN : bon classement, probabilites inutilisables

Le KNN atteint une AUC de 0.739 : il classe correctement les tirs. Mais son LogLoss de 0.80 est pire que celui d'une simple constante (0.34). Sa probabilite est la proportion de buts parmi les 15 plus proches voisins : quand aucun d'eux n'est un but, elle vaut exactement 0. C'est le cas de 809 tirs du test sur 2123, dont 33 sont pourtant des buts. Un but predit a 0% coute une penalite quasi infinie en LogLoss. Le Brier, qui borne la penalite a 1 par tir, le masque en grande partie. Les probabilites du KNN ne sont donc pas utilisables comme xG.

### Holdout Euro 2024 : une competition jamais vue

Le meme benchmark, sur les 1304 tirs de l'Euro 2024 (taux de but reel 7.5%) :

| Modele | ROC-AUC | Brier | LogLoss | xG moyen |
|---|---|---|---|---|
| Constante (taux de but du train) | - | 0.0702 | 0.2706 | 0.101 |
| **xG StatsBomb** | **0.787** | **0.0615** | **0.2251** | 0.085 |
| Regression logistique | 0.767 | 0.0634 | 0.2330 | 0.087 |
| Random Forest calibre | 0.741 | 0.0658 | 0.2445 | 0.090 |
| KNN | 0.654 | 0.0734 | 0.8316 | 0.089 |
| XGBoost | 0.755 | 0.0642 | 0.2360 | 0.086 |

Sur l'Euro 2024, 98 buts ont ete marques pour 111 attendus selon le xG StatsBomb. L'ecart represente 1.4 ecart-type et n'est donc pas significatif. C'est d'ailleurs une propriete connue du xG : sur un seul tournoi, la reussite devant le but varie beaucoup, et c'est precisement ce qui rend le xG utile sur une saison plutot que sur un match.

### XGBoost contre regression logistique : un ecart dans le bruit

XGBoost (`src/train_xgboost.py`) est le dernier modele ajoute. L'arret precoce se regle sur 15% des matchs du train, pour que le test reste vierge, et s'arrete a 168 arbres. Sur le test, il devance la regression logistique sur les trois metriques. Sur le holdout, c'est l'inverse.

Choisir la regression logistique a cause du holdout reviendrait a utiliser le holdout pour choisir, ce que le decoupage interdit. La decision se prend donc sur le test seul, en verifiant d'abord si l'ecart y depasse le bruit. `src/bootstrap_compare.py` tire 2000 fois les 86 matchs du test avec remise (les tirs d'un meme match ne sont pas independants) et recalcule l'ecart entre les deux modeles sur chaque tirage. Le bootstrap est apparie : les deux modeles sont evalues sur les memes tirs, ce qui annule le bruit qui leur est commun.

| Test (86 matchs) | Ecart XGBoost - logistique | IC 95% | Tirages ou XGBoost est devant |
|---|---|---|---|
| AUC (plus haut = mieux) | +0.0025 | [-0.0118, +0.0170] | 64% |
| Brier (plus bas = mieux) | -0.0015 | [-0.0038, +0.0006] | 91% |
| LogLoss (plus bas = mieux) | -0.0042 | [-0.0106, +0.0022] | 90% |

Les trois intervalles contiennent zero : sur le test, les deux modeles sont indiscernables, y compris sur le Brier et le LogLoss, les metriques qui comptent pour un xG. La tendance favorise XGBoost sur les probabilites (91% des tirages pour le Brier), mais 86 matchs ne suffisent pas a la confirmer. A modeles indiscernables, on garde le plus simple : la regression logistique, dont les coefficients se lisent directement, la ou il faudrait du SHAP pour expliquer le boosting.

Le holdout ne sert pas a choisir. A titre de constat, il ne contredit pas ce choix : l'ecart y va dans l'autre sens, et ses intervalles contiennent eux aussi zero (AUC [-0.028, +0.004], Brier [-0.0015, +0.0029]).

## API de scoring (FastAPI)

La regression logistique est exposee via une API FastAPI (`api/main.py`), avec un endpoint `POST /predict` qui prend en entree les caracteristiques geometriques d'un tir et retourne la probabilite de but. Elle est statistiquement indiscernable de XGBoost, le meilleur modele sur le test, tout en etant plus simple et lisible (voir le benchmark), et ses probabilites sont deja fideles sans calibration (xG moyen de 0.100 sur le test pour un taux de but du train de 0.101). Une documentation interactive est generee automatiquement par FastAPI sur `/docs`, permettant de tester l'API sans ecrire de code.

**Changement d'interface** : le champ `distance_to_keeper` est remplace par `keeper_off_line`. Les appels ecrits pour la version precedente doivent etre adaptes.

## Limite assumee du modele

Les features utilisees sont purement geometriques (position du tir, du gardien, des defenseurs). Le modele ne capture rien de la qualite d'execution individuelle du tireur (puissance de frappe, precision, pied fort), une limite connue des modeles xG publics qui n'ont pas acces a des donnees biometriques ou de vitesse de frappe.

## Conclusion

Le projet est clos sur ce resultat : une regression logistique de 24 variables, essentiellement la geometrie du tir, capture 79% de l'amelioration que le xG StatsBomb obtient sur une prediction constante (AUC 0.792 contre 0.828 sur le test), avec des probabilites fideles sans calibration.

- **Le plafond tient aux donnees, pas a l'algorithme.** Quatre familles de modeles (lineaire, Random Forest, KNN, boosting) ont ete essayees. Le meilleur d'entre eux, XGBoost, n'est pas statistiquement distinguable de la regression logistique. L'ecart restant avec StatsBomb vient d'informations non exploitees (passe decisive, face-a-face, deviations), pas d'un modele plus flexible.
- **Le modele servi est le plus simple qui atteint ce plafond.** Ses coefficients ont un signe footballistique coherent, et un modele d'arbres explique avec SHAP hierarchise les variables geometriques de la meme facon (Spearman 0.90).
- **La methode a ete tenue.** Decoupage par match, holdout par competition jamais utilise pour choisir, reference triviale calculee sur le train, et ecarts entre modeles juges a l'aune d'un bootstrap par match plutot que sur un chiffre ponctuel.
- **Le resultat generalise.** Sur l'Euro 2024, une competition jamais vue, l'AUC baisse de 0.792 a 0.767, et celle de StatsBomb de 0.828 a 0.787 : le tournoi est plus dur a predire pour tout le monde, et l'ecart avec StatsBomb ne se creuse pas.

## Structure du projet

```
xg-shot-prediction/
├── api/
│   └── main.py                  # API FastAPI de scoring
├── models/
│   ├── rf_calibrated.pkl        # Random Forest calibre (explique par le notebook)
│   ├── logistic.pkl             # Regression logistique + scaler (servi par l'API)
│   ├── xgboost.pkl              # XGBoost (compare a la regression logistique)
│   └── knn.pkl                  # KNN + scaler (non versionne)
├── data/
│   ├── raw/                     # Tirs et freeze frames extraits
│   └── processed/                # Features finales
├── notebooks/
│   └── 01_edashap.ipynb          # EDA, coefficients de la regression logistique, SHAP du Random Forest
├── src/
│   ├── inventory_360.py          # Inventaire des competitions StatsBomb avec donnees 360
│   ├── extract_data.py           # Extraction StatsBomb via mplsoccer
│   ├── features.py                # Feature engineering geometrique, listes FEATURES et ID_COLUMNS
│   ├── split.py                   # Decoupage train / test / holdout, commun a tous les scripts
│   ├── train_logistic.py          # Regression logistique, sauvegarde du modele pour l'API
│   ├── train_rf.py                # Random Forest, calibration, sauvegarde du modele pour le notebook
│   ├── train_knn.py
│   ├── train_xgboost.py           # XGBoost avec arret precoce sur une validation tiree du train
│   ├── benchmark.py               # Comparaison des modeles sauvegardes et du xG StatsBomb
│   └── bootstrap_compare.py       # Bootstrap par match de l'ecart XGBoost - regression logistique
├── reports/                      # Courbes ROC, coefficients, graphiques SHAP, courbe de calibration
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
python src/train_xgboost.py
python src/benchmark.py
python src/bootstrap_compare.py

uvicorn api.main:app --reload
# Documentation interactive disponible sur http://127.0.0.1:8000/docs
```

## Pistes pour aller plus loin

Le projet est clos. Ces pistes restent ouvertes pour une suite eventuelle :

- Exploiter les informations presentes dans les donnees brutes StatsBomb mais pas encore utilisees, candidates pour expliquer l'ecart avec leur xG :
  - la passe decisive : centre (`pass_cross`), passe en retrait (`pass_cut_back`), hauteur de la passe (`pass_height_name`), via `shot_key_pass_id`
  - le contexte du tir annote par StatsBomb : face-a-face avec le gardien (`shot_one_on_one`), but vide (`shot_open_goal`), tir apres un dribble (`shot_follows_dribble`), duel aerien gagne (`aerial_won`)
  - les tirs devies ou reorientes (`shot_deflected`, `shot_redirect`)
- Analyse de l'effet du melange football masculin / feminin et clubs / selections sur les predictions
- Exploration d'une version inspiree de TacticAI avec un reseau de neurones sur graphe (GNN), une fois les bases du deep learning acquises

## Stack technique

Python, Pandas, Scikit-learn, XGBoost, mplsoccer, SHAP, Matplotlib, FastAPI, Uvicorn, Git