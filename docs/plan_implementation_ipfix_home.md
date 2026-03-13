# Plan d'Implémentation - Dataset "IoT IPFIX Home"

Ce document décrit le plan étape par étape pour compléter l'implémentation du projet en se basant sur les spécifications du manuscrit `Manuscript anonymous.pdf`, spécifiquement pour la partie concernant le dataset **IoT IPFIX Home**.

## État des lieux actuel
Le code actuel gère le chargement optimisé du dataset, le filtrage des **18 classes** spécifiques à "IoT IPFIX Home" (Table 7 du PDF), ainsi que la normalisation/standardisation des données. Les 4 modèles d'identification de base (RF, XGBoost, KNN, DNN) sont également implémentés mais avec des paramètres génériques.

---

## Plan d'Action

### Étape 1 : Ajustement des hyperparamètres des Classifieurs de Base (Model-3)
Actuellement, les modèles utilisent des hyperparamètres génériques. Il faut aligner **XGBoost, RF, KNN, et DNN** avec les configurations exactes du **Tableau 3 / Tableau 6** du manuscrit.
- **XGBoost** : utiliser la métrique d'évaluation `merror` (au lieu de `mlogloss`), `learning_rate` à `0.1`, `max_depth` à `None` (ou valeur par défaut), `estimators` à `200`.
- **KNN** : utiliser 5 voisins (`neighbors: 5`), algorithme `auto`, poids `distance`.
- **RF** : `estimators` à `300`, `max_depth` à `None`, `bootstrap` à `True`.
- **DNN** : 3 couches cachées avec 256 neurones chacune, activation `ReLU`, optimiseur `Adam`.

### Étape 2 : L'Algorithme de Génération d'Attaque d'Usurpation (Adversarial Evasion)
Le notebook actuel utilise une attaque basique par bruit aléatoire (`np.random.randn()`). Le manuscrit spécifie une attaque beaucoup plus sophistiquée, ancrée dans la sémantique SDN (Section 4.2). Il faut implémenter :
1. **La formule exacte d'attaque basée sur les centroids** : L'attaque itérative approche le flux vers le centre de gravité de la classe ciblée : `x_adv = Projection [x0 + c · t · mask · sign(µtarget - x0) · |Difference(µtarget, x0)|]`.
2. **Minimisation L2 (Cibler les 3 classes proches)** : Calculer les centroids des classes avec K-means et cibler dynamiquement l'une des 3 classes les plus proches du flux.
3. **Minimisation L0 (Les Masques et variables SDN)** : Classifier les attributs SDN en 3 catégories (`Indépendantes`, `Dépendantes`, `Non-modifiables`) et implémenter la fonction de **Projection** pour assurer que le perturbateur modifie uniquement ce qui est physiquement plausible sur le réseau SDN.
4. **Attaques sur les 4 modèles** : Générer des exemples adversariaux pour tromper non seulement XGBoost, mais l'ensemble des 4 modèles.

### Étape 3 : La Détection Adversariale / Détecteur Binaire (Model-1)
Actuellement, le code n'entraîne qu'un seul modèle Random Forest (`det_rf`) pour classer un trafic comme Sain ou Adversarial.
1. Entraîner les **4 Modèles de Détection Binaire** (RF, XGBoost, KNN, DNN).
2. Vérifier et appliquer les hyperparamètres du **Tableau 4** :
   - **XGBoost** : `max_depth: 3`, `estimators: 300`, `learning_rate: 0.2`.
   - **KNN** : `neighbors: 5`, `weights: uniform`, `algorithm: brute-force`.
   - **RF** : `estimators: 300`, `max_depth: None`, `max_features: sqrt`, `min_samples_split: 5`.
   - **DNN** : `loss: sparse_categorical_crossentropy`, 3 couches de 256 neurones, `activation: ReLU`, `output: Softmax`, `optimizer: Adam`.

### Étape 4 : L'Entraînement Adversarial / Classifieur Robuste (Model-2)
Il s'agit de la deuxième ligne de défense (Model-2, Section 4.3). Le code actuel n'entraîne qu'un seul "Random Forest Robuste".
1. Adapter le pipeline pour ré-entraîner les **4 Modèles (RF, XGBoost, KNN, DNN)** sur le dataset mixte (Données saines + Exemples adversariaux).
2. Appliquer les hyperparamètres listés dans le **Tableau 5** pour cette phase :
   - **XGBoost** : `max_depth: 7`, `estimators: None` (valeur par défaut), `learning_rate: 0.2`.
   - **KNN** : `neighbors: 3`, `weights: distance`, `algorithm: auto`.
   - **RF** : `estimators: 200`, `max_depth: None`, `max_features: sqrt`, `bootstrap: True`.
   - **DNN** : 3 couches de 256 neurones, `activation: ReLU`.

### Étape 5 : Assembler l'Architecture "Two-Tiered Defense" Finale
Le manuscrit propose un pipeline d'évaluation en temps réel (Figure 4) que le notebook ne simule pas encore.
1. Implémenter la logique d'évaluation en cascade :
   - Tester le trafic entrant via le **Détecteur Binaire (Model-1)**.
   - S'il est classé comme *adversarial*, il est traité par le **Classifieur Robuste (Model-2)** (ou la politique décide de l'isoler).
   - S'il est classé comme *sain*, il est classifié par le **Modèle Initial (Model-3)**.
2. Évaluer et extraire les métriques de recouvrement (Recovery F1-score) pour ce pipeline entier afin de reproduire les résultats du papier (Figures 5, 7, et 10).
