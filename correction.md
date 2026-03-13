# Plan de Correction pour `ipfix_home_pipeline.py`

Ce document détaille le plan complet pour corriger la pipeline d'apprentissage automatique, afin d'obtenir des résultats identiques à ceux du PDF (Manuscript anonymous), en réglant les problèmes de métriques, de graphiques et de performances de la défense.

## 1. Ajouter toutes les métriques (Accuracy, Precision, Recall, F1-Score)
Dans les étapes 3 (Impact), 5 (Entraînement Robuste) et 6 (Défense à deux niveaux), le code n'enregistre actuellement que le score F1 (ou manque certaines métriques pour certaines étapes). Il faut intégrer les autres métriques :

*   **Étape 3 (`generate_adversarial_samples`)** :
    *   Calculer `accuracy_score`, `precision_score`, `recall_score` en plus du `f1_score` pour les prédictions `y_pred_c` (clean) et `y_pred_a` (adversarial).
    *   Ajouter ces variables dans le dictionnaire `impact_results` avant de le sauvegarder en CSV.
*   **Étape 5 (`train_robust_classifiers`)** :
    *   Appliquer la même logique en calculant l'`Accuracy`, la `Precision` et le `Recall` pour `X_test_sample` et `X_adv_test`.
    *   Ajouter ces champs à `rob_results` afin qu'ils apparaissent dans `step5_robust_training_results.csv`.
*   **Étape 6 (`two_tiered_defense_evaluation`)** :
    *   Faire de même pour `y_pred_clean` et `y_pred_adv` afin d'avoir une évaluation complète de la pipeline de défense dans `step6_twotiered_defense_results.csv`.

## 2. Générer les graphiques pour TOUS les modèles
Le code actuel filtre explicitement sur "XGBoost" pour générer les rapports de classes et les Figures 5 et 7. Il faut que les graphiques reflètent tous les modèles (RF, KNN, XGBoost, DNN).

*   **Correction de l'Étape 2 (Figure 5)** :
    *   Modifier la fonction `plot_figure_5` pour qu'elle prenne en paramètre le nom du modèle.
    *   Dans `train_base_classifiers`, faire une boucle : pour chaque `name` (RF, KNN, XGBoost, DNN), appeler `plot_figure_5` et sauvegarder l'image sous un nom spécifique (ex: `Figure_5_Device_Identification_RF.png`).
*   **Correction de l'Étape 3 (Figure 7)** :
    *   Actuellement, les variables `per_class_f1_clean` et `per_class_f1_adv` ne sont remplies que sous la condition `if name == "XGBoost":` (Lignes 384-393).
    *   Il faut **supprimer cette condition** et créer une structure (comme un dictionnaire de dictionnaires) qui stocke les résultats par classe pour **chaque modèle**.
    *   Ensuite, boucler sur tous les modèles pour générer un graphique de la Figure 7 pour chacun (ex: `Figure_7_Adversarial_Effect_RF.png`).

## 3. Résoudre le problème du Max 50% après la Défense (Atteindre 88%)
Le PDF annonce ~88% de récupération (recovery/F1) après la défense, mais le CSV affiche autour de 15% à 50%. Ce crash de performance s'explique par un **Data Leakage (fuite de données)** et un **déséquilibre majeur lors de l'entraînement robuste**.

*   **Le BUG actuel (Ligne 354 & 529)** :
    *   À l'étape 3, vous générez `X_adv` à partir d'un échantillon du set de **Test** (`X_test[:n_adv]`).
    *   À l'étape 5, vous construisez le jeu d'entraînement robuste en faisant `np.vstack([X_train, X_adv])`. Vous entraînez donc le modèle de défense Modèle-2 (Robuste) en injectant les exemples adversariaux générés à partir des données de **Test**. Pire encore, la quantité d'exemples adversariaux injectés est minuscule par rapport à la taille massive de `X_train`. Le modèle ne devient pas vraiment robuste, il noie simplement la petite quantité d'attaques.
*   **La SOLUTION pour atteindre 88%** :
    *   **Séparer la génération d'attaques :** Dans l'étape 3 (`generate_adversarial_samples`), il faut générer deux ensembles :
        *   Un batch `X_adv_train` à partir d'un sous-ensemble aléatoire de `X_train` (par ex. 20% à 50% de `X_train`).
        *   Un set de test `X_adv_test` à partir de `X_test`.
    *   **Entraînement de la Défense (Étape 5)** : Utilisez seulement `X_adv_train` combiné avec `X_train` pour entraîner vos `robust_models`. Ainsi, le modèle apprendra à se défendre sans voir les données de test !
    *   **Évaluation de la Défense (Étape 6)** : Utilisez uniquement `X_adv_test` pour mesurer les performances dans `two_tiered_defense_evaluation`. Ainsi, la défense s'entraînera sur une quantité suffisante d'exemples malveillants tout en respectant l'isolation train/test, et fera remonter drastiquement le F1-score à 88%.
    *   **Ajuster les paramètres d'attaque (`c` et `max_iter`)** : Si la perturbation est trop énorme (clipping agressif), le modèle robuste ne peut plus rien reconnaître. L'algorithme d'attaque décrit dans l'article applique une projection contrainte. Il est important d'ajuster si nécessaire.

## Prochaines étapes
Prêt à appliquer ces modifications dans `ipfix_home_pipeline.py` !
