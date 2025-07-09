# Roadmap d’Évolution – Projet Classification Automatisée de Spectres Stellaires (LAMOST DR5)

## 1. **Enrichissement du Contenu Scientifique et des Features**

### 1.1. **Extraction de Features Spectrales Avancées**

- **Ajout de raies supplémentaires :**
    - He II 4686 Å (O/B),
    - G-band (CH ~4300 Å, G/K),
    - Ca II IR triplet (8498/8542/8662 Å, tardives),
    - Na I D (5890/5896 Å),
    - Mg b (5175 Å),
    - Raies TiO (M et plus tardives), etc.
- **Mesures physiques des raies :**
    - Profondeur relative,
    - Largeur à mi-hauteur (FWHM),
    - Aire sous la raie (équivalent width),
    - Ratios d’intensité entre plusieurs raies.
- **Détection dynamique :**
    - Laisser le pipeline détecter automatiquement les pics les plus significatifs (hors raies “catalogue”), et explorer leur importance pour la classification.

### 1.2. **Exploitation Avancée du Continuum**

- **Correction/fit automatique du continuum** (spline, polynôme, pseudo-continuum).
- **Calcul d’indices pseudo-couleur :**
    - Slope du continuum (température),
    - Courbure,
    - PCA (réduction de dimension sur spectre global).
- **Normalisation locale (“segment-wise”)** pour isoler encore mieux les raies.

### 1.3. **Intégration et Exploitation des Métadonnées Astrophysiques**

- **Ajouter comme features :**
    - Signal-to-noise (SNR),
    - Magnitudes (g, r, i, …),
    - Redshift,
    - Seeing,
    - RA/Dec (localisation spatiale pour populations typiques).
- **Analyser l’importance de ces features** pour la prédiction (feature importance, corrélations).

## 2. **Prétraitement et Qualité des Données**

### 2.1. **Nettoyage et Améliorations Spectroscopiques**

- **Denoising/adoucissement adaptatif :**
    - Filtre Savitzky-Golay, filtre médian (surtout spectres à faible SNR).
- **Correction de l’extinction interstellaire** (si l’info est disponible dans le header).
- **Redressement local (segment-wise normalization)**.

### 2.2. **Détection et Traitement des Outliers**

- **Détection automatique des spectres bruités/artéfacts.**
- **Nettoyage ou exclusion des spectres trop aberrants.**

### 2.3. **Augmentation et Optimisation des Données**

- **Équilibrage du dataset :**
    - Sur-échantillonnage des classes rares (SMOTE),
    - Sous-échantillonnage des majoritaires,
    - Télécharger plus de spectres pour les classes sous-représentées, compléter avec SDSS, DR8, Gaia si besoin.
- **Data augmentation :**
    - Simuler du bruit réaliste,
    - Variations de flux,
    - Décalages spectraux réalistes.

## 3. **Modèles d’Apprentissage Automatique**

### 3.1. **Diversification et Optimisation des Algorithmes**

- **Tester divers modèles :**
    - SVM (kernels linéaire/RBF),
    - Gradient Boosting (LightGBM, XGBoost),
    - KNN,
    - Réseaux de neurones (MLP pour features tabulaires, CNN 1D sur spectre brut/réduit),
    - Autoencodeurs.
- **Ensembles/Stacking/Voting :**
    - Fusionner plusieurs modèles pour robustesse (RandomForest, SVM, MLP…).

### 3.2. **Tuning & Validation Rigoureuse**

- **Hyperparameter tuning** (GridSearchCV, RandomizedSearchCV, Optuna).
- **Validation croisée k-fold** (stratifiée, k=5 ou 10).
- **Suivi de la performance au fil des lots (learning curve)**.

## 4. **Reporting, Visualisation et Monitoring**

### 4.1. **Reporting Automatisé et Visualisation**

- **Générer automatiquement :**
    - Matrices de confusion,
    - Courbes ROC/PR,
    - Courbes d’importance des features.
- **Rapports PDF/HTML par lot traité** (avec Jinja2, nbconvert).
- **Ajout des figures au rapport de session ou notebook récapitulatif**.

### 4.2. **Suivi et Monitoring Expériences**

- **Monitorer avec MLflow, Weights & Biases, TensorBoard** pour suivre les runs, paramètres, métriques, version des modèles/datasets.
- **Logs détaillés :**
    - Hash du code,
    - Timestamp,
    - Conditions d’entraînement.

## 5. **Productivité, Automatisation, Infrastructure**

### 5.1. **Structuration et Orchestration**

- **Pipelines scikit-learn** (Pipeline(), FeatureUnion…)
- **Orchestration avec Makefile, Snakemake, Prefect** (lancement d’étapes en chaîne, reproductibilité).
- **CI/CD :**
    - Tests unitaires automatiques,
    - Github Actions (exécution pipeline/tests à chaque push).
- **Standardisation/scaling des features** (important pour SVM, NN).
- **Fournir un dataset mock/démo** et un notebook Quickstart pour test rapide du pipeline.

### 5.2. **Stockage et Collaboration**

- **Synchronisation automatique data/raw et data/catalog** avec Google Cloud Storage (GCS).
- **Versionning automatique modèles/datasets** (hash unique/timestamp).

### 5.3. **Optimisation technique**

- **Utiliser n_jobs=-1** dans scikit-learn pour accélérer RandomForest, GridSearch, etc.
- **Exploiter la parallélisation CPU/GPU via GCP** si deep learning envisagé.

## 6. **Déploiement, Accessibilité et Interfaces Utilisateur**

### 6.1. **API et Interfaces**

- **Créer une API Flask/FastAPI** pour prédire à partir d’un spectre envoyé en entrée.
- **Construire une interface web (Streamlit, Gradio)** pour charger un spectre, visualiser le résultat, l’explication.

### 6.2. **Traitement Continu et Automatisation**

- **Automatiser le traitement de nouveaux spectres dès disponibilité** (watcher, cron…).
- **Démo interactive (notebook Colab, mini-app web)** pour la communauté.

## 7. **Ouverture IA Moderne et Valorisation Scientifique**

### 7.1. **Explicabilité et Modernité IA**

- **Ajouter des modules explicabilité (SHAP, LIME)** pour expliquer les prédictions.
- **Prototyper un chatbot/assistant documentaire** (mini-RAG local, LlamaIndex sur la doc astro + pipeline).

### 7.2. **Approfondissement scientifique**

- **Classification fine :**
    - Prédire sous-classes (ex : F5, G2…),
    - Luminosité,
    - Métallicité,
    - Gravité de surface.
- **Anomaly detection :**
    - Isolation Forest, autoencodeur non supervisé pour objets exotiques, erreurs, transitoires.
- **Classification multi-tâches** (type spectral + autre propriété astro).

### 7.3. **Extension du pipeline**

- **Tester sur d’autres catalogues (SDSS DR16, Gaia XP spectra, etc.)**
- **Pipeline multi-survey (fusionner plusieurs sources)**

## 8. **Vision Long Terme & Impact Communautaire**

### 8.1. **Science ouverte, réutilisabilité, pédagogie**

- **Déployer un module open source réutilisable** (installation facile, contribution).
- **Préparer une démo Colab minimaliste et un mini-dataset “mock”**.
- **Rédiger une fiche “How-to contribute”** pour étudiants/chercheurs externes.
- **Créer un outil pédagogique interactif** (visualisation “pas à pas”, explication pour étudiants).

### 8.2. **Valorisation scientifique avancée**

- **Études astrophysiques ciblées** (populations spécifiques, Voie lactée, objets rares).
- **Support temps réel à l’observation** (pipeline plug-and-play pour télescope).
- **Moonshots IA :**
    - Self-supervised (SimCLR, BYOL),
    - Détection de transitoires rares,
    - Pipeline citizen science.
