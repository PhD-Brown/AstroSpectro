# Roadmap AstroSpectro

Les tâches sont regroupées par thème avec des cases à cocher pour suivre leur progression. Les cases `✔️` indiquent des éléments déjà réalisés.

## 1. Enrichissement du contenu scientifique et des features

### 1.1 Extraction de features spectrales avancées

- **Ajout de raies supplémentaires :**
    - [ ]  He II 4686 Å (O/B)
    - [ ]  G-band (CH ~4300 Å, G/K)
    - [ ]  Ca II IR triplet (8498/8542/8662 Å, tardives)
    - [x]  Na I D (5890/5896 Å)
    - [x]  Mg b (5175 Å)
    - [x]  Raies TiO (M et plus tardives), etc
- **Mesures physiques des raies :**
    - [x]  Profondeur relative
    - [x]  Largeur à mi-hauteur (FWHM)   → _skew testé et apporte rien d’intéressant
    - [x]  Aire sous la raie (équivalent width)
    - [x]  Ratios d’intensité entre plusieurs raies (Hα/Hβ, Ca K/Hβ, Mg b/Hβ)
    - [ ]  Moments d’ordre supérieur (variance, asymétrie, kurtose)
    - [ ]  Ajustement de profils Gaussiens/Voigt pour extraire des paramètres robustes
- [ ]  **Détection automatique de pics hors catalogue** : détecter les pics significatifs dans chaque spectre et les utiliser comme features génériques.

### 1.2 Exploitation avancée du continuum

- **Indices pseudo‑couleur** :
    - [ ]  pente du continuum
    - [ ]  courbure
    - [ ]  indices couleur (u–g, g–r, r–i) dérivés des magnitudes
- **Correction automatique du continuum** :
    
    [Correction automatique du continuum : principe et bonnes pratiques](https://www.notion.so/Correction-automatique-du-continuum-principe-et-bonnes-pratiques-248049738f2680c3a27ce021c9502518?pvs=21)
    
    - [ ]  ajustement par spline
    - [ ]  polynôme pour redresser la forme du spectre.
    - [ ]  Méthodes automatiques “pseudo-continuum”
- **Réduction de dimension** :
    
    [Réduction de dimension : principe et bonnes pratiques](https://www.notion.so/R-duction-de-dimension-principe-et-bonnes-pratiques-248049738f268071a5b3d4ede12b91fb?pvs=21)
    
    - [ ]  appliquer l’analyse en composantes principales (PCA) ou l’auto‑encodeur sur les spectres pour obtenir des descripteurs globaux.
- [ ]  **Normalisation segmentée (local) (“segment-wise”)** pour isoler encore mieux les raies.

### 1.3 Intégration et exploitation des métadonnées astrophysiques

- **Ajouter comme features :**
    - [x]  Signal-to-noise (SNR)
    - [ ]  **Utiliser les magnitudes u,g,r,i,z et dériver des couleurs**. → Pas utile ?
    - [x]  Redshift
    - [x]  Seeing
    - [x]  RA/Dec (localisation spatiale pour populations typiques)  → (optionnel, désactivé si non pertinent)
- [ ]  **Ajouter des paramètres du header** : correction héliocentrique, vitesse radiale (`VELDISP`), etc.
- [ ]  **Analyse de l’importance de ces features** (corrélation, importance de variables).
- [ ]  **Cross‑matching automatique avec Gaia DR3** pour enrichir les labels (température effective, gravité, metallicité) et valider la classification.

---

## 2. Pré‑traitement et qualité des données

### 2.1 Nettoyage et améliorations spectroscopiques

- **Denoising/adoucissement adaptatif :**
    - [x]  **Denoising et lissage** : filtre Savitzky–Golay (à peaufiner)
    - [ ]  **Filtre médian adaptatif** pour les spectres à faible SNR.
- [ ]  **Correction de l’extinction interstellaire** à partir des cartes ou des paramètres du header.
- [ ]  **Correction des raies telluriques** (O₂, H₂O) en utilisant des bases de données (HITRAN) ou la bibliothèque PyTelluric.
- [ ]  **Détection et correction de pixels chauds/artéfacts instrumentaux** en appliquant un test statistique (z‑score > 5) et en remplaçant par la médiane locale.

### 2.2 Détection et traitement des outliers

- [ ]  **Détection automatique de spectres bruités ou aberrants** via des mesures robustes (MAD, isolation forest).
- [ ]  **Exclusion ou correction des spectres aberrants** avant l’entraînement.

### 2.3 Augmentation et optimisation des données

- **Équilibrage du dataset :**
    - [x]  **Équilibrage avec SMOTE**
    - [ ]  **Sous‑échantillonnage des classes majoritaires**.
    - [ ]  **Télécharger plus de spectres LAMOST/SDSS/DR6/DR8 pour les classes rares**.
- **Data augmentation réaliste :**
    - [ ]  Simuler du bruit
    - [ ]  Variations de flux
    - [ ]  Décalages spectraux
- [ ]  **Générer des spectres synthétiques** (PyAstronomy, Synspec) pour créer des cas contrôlés et tester le pipeline.

---

## 3. Modèles d’apprentissage automatique

### 3.1 Diversification et optimisation des algorithmes

- **Tester divers modèles :**
    - [x]  **RandomForest et XGBoost** (modèle actuel)
    - [ ]  **Tester d’autres gradient boosting** : LightGBM, CatBoost.
    - [ ]  **SVM (linéaire et RBF)**.
    - [ ]  **Réseaux de neurones 1D (CNN) sur le spectre entier**.
    - [ ]  **Transformers spectraux ou modèles auto‑encodeurs** pour capturer des dépendances globales.
    - [ ]  **Auto‑encodeurs variés** (VAE) pour la réduction de dimension et l’augmentation.
- [ ]  **Étendre le tuning XGBoost** : tester `n_estimators`, `subsample`, `colsample_bytree`, `gamma`, etc., via RandomSearch ou Optuna.
- [ ]  **K‑NN et MLP** pour features tabulaires.
- [ ]  **Ensembles/stacking/voting** : combiner plusieurs algorithmes (RandomForest, SVM, MLP) pour améliorer la robustesse.
- [ ]  **Bagging personnalisé** (au‑delà du RandomForest) pour réduire la variance sur les classes rares.
- [ ]  **Classification hiérarchique** : prédire d’abord le type principal (O/B/A/F/G/K/M), puis la sous‑classe.
- [ ]  **Classification multi‑tâches** : prédire simultanément le type spectral et une autre propriété astrophysique (luminosité, métallicité, gravité de surface).
- [ ]  **Anomaly detection** : Isolation Forest, auto‑encodeurs non supervisés pour détecter des objets rares ou des erreurs.

### 3.2 Tuning & validation rigoureuse

- [x]  **GridSearchCV et validation croisée stratifiée**
- [ ]  **RandomizedSearchCV ou Optuna** pour parcourir des espaces d’hyper‑paramètres plus larges.
- [ ]  **Learning curves** pour visualiser la progression de la performance en fonction de la taille de l’échantillon.
- [ ]  **Cross‑validation k‑fold plus élevée** (k = 5 ou 10) pour réduire la variance.
- [ ]  **Standardisation/scaling des features** (important pour SVM, NN).

---

## 4. Reporting, visualisation et monitoring

[Améliorations à envisager](https://www.notion.so/Am-liorations-envisager-248049738f268078aa6ce4cbd6f1c2cc?pvs=21)

### 4.1 Reporting automatisé et visualisation

- **Générer automatiquement :**
    - [x]  **Matrice de confusion** et reporting d’évaluation générés en fin d’entraînement
    - [ ]  **Courbes ROC et Precision–Recall** automatiques pour chaque classe.
    - [ ]  **Graphiques d’importance des features** (barres, SHAP, Permutation Importance).
- [ ]  **Rapports PDF/HTML** générés pour chaque lot avec NBConvert/Jinja 2.
- [ ]  **Exportation des visualisations** au format PNG/SVG/PDF pour la communication.
- [ ]  **Dashboards interactifs** (Plotly Dash, Streamlit) : exploration des distributions de features, paramétrage en direct, comparaison modèle/réalité, annotations de raies.

### 4.2 Suivi et monitoring des expériences

- [ ]  **Tracking avec MLflow, Weights & Biases ou TensorBoard** : log des runs, hyper‑paramètres, métriques, artefacts.
- **Logs détaillés :**
    - [x]  Hash du code
    - [x]  Timestamp
    - [ ]  Conditions d’entraînement (hyperparamètre, résultat entraînement, features importantes, etc )
- [ ]  **Gestion des versions des modèles et datasets** avec DVC pour assurer la traçabilité scientifique.

---

## 5. Productivité, automatisation et infrastructure

### 5.1 Structuration et orchestration

- [x]  **Pipelines scikit‑learn** (ImbPipeline) utilisés pour l’entraînement
- [ ]  **Orchestration globale** du pipeline via Makefile, Snakemake ou Prefect pour automatiser l’enchaînement des étapes (téléchargement, pré‑traitement, features, classification, reporting).
- **CI/CD** :
    - [ ]  Mise en place de tests unitaires et d’intégration
    - [x]  Exécution automatique des tests et de la documentation via GitHub Actions.
- [ ]  **Conteneurisation avec Docker** pour faciliter le déploiement et la collaboration.
- [ ]  **Dataset mock et notebook Quickstart** pour la prise en main.

### 5.2 Stockage et collaboration

- [ ]  **Synchronisation automatique** des dossiers `data/raw` et `data/catalog` avec un stockage cloud.
- [ ]  **Versionnement des modèles et données** via DVC (en complément de Git).
- [x]  **Partage et collaboration** via GitHub (issues, Wiki) et un guide de contribution clair.

### 5.3 Optimisation technique

- [ ]  **Utiliser `n_jobs=-1`** pour exploiter tous les cœurs CPU lors des entraînements et des recherches de paramètres.
- [ ]  **Exploration de l’accélération GPU** (via XGBoost GPU, cuDF) si nécessaire.
- [ ]  **Parallelisation et cloud computing** (GCP, HPC) pour les réseaux de neurones.

---

## 6. Déploiement, accessibilité et interfaces utilisateur

### 6.1 API et interfaces

- [ ]  **API Flask/FastAPI** pour prédire à partir d’un spectre fourni en entrée.
- [ ]  **Interface web interactive** (Streamlit, Gradio) pour charger un spectre, le visualiser, afficher la prédiction et l’explication SHAP.
- [ ]  **Application mobile ou widget** pour vulgariser l’outil (optionnel à long terme).

### 6.2 Traitement continu et automatisation

- [ ]  **Watchers ou cron jobs** pour traiter automatiquement les nouveaux spectres dès qu’ils sont disponibles.
- [x]  **Démo interactive** : CodeSpace/Colab en place pour tester rapidement le pipeline.

---

## 7. Ouverture IA moderne et valorisation scientifique

### 7.1 Explicabilité et modernité IA

- I**ntégration de méthodes d’explicabilité**
    
    [Explicabilité des modèles : principe et bonnes pratiques](https://www.notion.so/Explicabilit-des-mod-les-principe-et-bonnes-pratiques-248049738f2680e1ba5deccbca63ca61?pvs=21)
    
    - [x]  **SHAP**
    - [ ]  **LIME**
    - [ ]  **CAPTUM**
- [ ]  **Prototyper un chatbot/assistant documentaire** (mini‑RAG local) pour répondre aux questions des utilisateurs.

### 7.2 Approfondissement scientifique

- **Classification fine :**
    - [x]  Prédire des sous‑classes (F5, G2…)
    - [ ]  Luminosité
    - [ ]  Métallicité
    - [ ]  Gravité de surface
- [ ]  **Détection d’anomalies et objets exotiques** (étoiles Wolf‑Rayet, transitoires, binaires interactifs) via des méthodes non supervisées.
- [ ]  **Classification multi‑tâches** (voir section 3.1).

### 7.3 Extension du pipeline

- **Tester sur d’autres catalogues** :
    - [ ]  LAMOST DR6/DR8 … DR10
    - [ ]  SDSS DR16/DR17
    - [ ]  Gaia XP
    - [ ]  SITELLE & MUSE.
- [ ]  **Pipeline multi‑survey** : fusionner des spectres de différentes sources pour une classification unifiée.
- [ ]  **Support des spectres cubes 3D** (SITELLE, MUSE) : adapter l’extraction de features à la dimension spatiale.
- [ ]  **Apprentissage par transfert et semi‑supervisé** : pré‑entraîner sur de grands jeux non étiquetés (SDSS) puis fine‑tuner sur LAMOST.

---

## 8. Vision long terme et impact communautaire

### 8.1 Science ouverte, réutilisabilité et pédagogie

- [ ]  **Module open source réutilisable** : packaging, documentation et distribution via PyPI ou Conda.
- [ ]  **Dataset “mock” et notebook didactique** pour les étudiants.
- [ ]  **Rédiger un guide “How‑to contribute”** et encourager les contributions externes.
- [ ]  **Créer un outil pédagogique interactif** (pas à pas) pour expliquer la classification spectrale.
- [x]  **Configurer des CodeSpaces prêts à l’emploi** pour faciliter la contribution.

### 8.2 Valorisation scientifique avancée

- [ ]  **Études astrophysiques ciblées** : utiliser AstroSpectro pour étudier des populations stellaires spécifiques (halo galactique, disques, amas globulaires).
- [ ]  **Support temps réel** : envisager l’utilisation du pipeline comme module plug‑and‑play pendant les observations (pré‑classification en temps réel).
- **Moonshots IA** :
    - [ ]  Tester des approches self‑supervisées (SimCLR, BYOL) sur des spectres 1D,
    - [ ]  Développer un pipeline citizen‑science pour la découverte d’objets rares.
    - [ ]  Détection de transitoires rares,

### 8.3 Rayonnement et communication

- [ ]  **Écrire un blog post ou réaliser une vidéo** pour présenter AstroSpectro, ses fonctionnalités et ses résultats.
- [ ]  **Préparer une présentation pour conférences étudiantes** ou ateliers (ACFAS, séminaires universitaires).
- [ ]  **Participer à des projets communautaires** (hackathons, citizen science).