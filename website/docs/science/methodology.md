---
id: methodology
title: Méthodologie & Vision d'Ensemble
sidebar_position: 2
---

# 🔬 Méthodologie & Vision d'Ensemble du Pipeline

Cette page décrit les choix scientifiques et techniques qui sous-tendent chaque étape du pipeline de classification spectrale. Pour une représentation visuelle, référez-vous au [schéma sur le README du projet](https://github.com/PhD-Brown/astro-spectro-classification#pipeline-structure).

### 1. Téléchargement et Gestion des Données

Le pipeline commence par une acquisition robuste des données brutes depuis les serveurs officiels de LAMOST.
- **Téléchargement Automatisé :** Utilise des scripts pour récupérer les fichiers FITS en masse.
- **Parsing de Catalogue :** Lit les catalogues d'observation pour associer les métadonnées à chaque spectre.
- **Gestion Locale :** Organise les données dans une structure de dossiers prévisible (`data/raw/`, `data/catalog/`).

### 2. Prétraitement et Contrôle Qualité

Les spectres bruts ne sont pas directement utilisables. Une étape de nettoyage rigoureuse est appliquée pour préparer les données à l'analyse.
- **Normalisation :** Le flux de chaque spectre est normalisé pour rendre les comparaisons possibles.
- **Ajustement du Continuum :** Un pseudo-continuum est ajusté et soustrait pour isoler les raies spectrales.
- **Déni de Bruit (Denoising) :** Des filtres (ex: Savitzky-Golay) sont appliqués pour réduire le bruit, en particulier sur les spectres à faible rapport signal/bruit (SNR).

### 3. Extraction de Features Physiques

C'est le cœur de l'approche "hybride" (physique + ML). Au lieu de donner le spectre brut au modèle, nous extrayons des informations physiquement pertinentes.
- **Identification des Raies :** Le pipeline identifie les raies astrophysiques majeures (Hα, Hβ, CaII K&H, etc.).
- **Mesures Physiques :** Pour chaque raie identifiée, des mesures quantitatives sont calculées :
  - Largeur à mi-hauteur (FWHM)
  - Profondeur ou hauteur de la raie
  - Largeur équivalente (Equivalent Width)
  - Ratios entre différentes raies

### 4. Entraînement et Validation des Modèles

Avec un jeu de données "propre" contenant les features extraites, nous pouvons entraîner des modèles de Machine Learning.
- **Modèles Flexibles :** Le pipeline supporte nativement des modèles robustes comme `Random Forest` et `SVM`, et est conçu pour intégrer facilement de nouveaux algorithmes (ex: Gradient Boosting, réseaux de neurones).
- **Validation Croisée :** La performance est évaluée rigoureusement en utilisant la validation croisée (k-fold) pour assurer que le modèle généralise bien à de nouvelles données.
- **Optimisation :** Des techniques comme le `GridSearchCV` peuvent être utilisées pour trouver les meilleurs hyperparamètres pour un modèle donné.

### 5. Évaluation et Reporting

Un modèle n'est utile que si l'on peut comprendre et faire confiance à ses prédictions.
- **Métriques Standards :** Matrices de confusion, scores de précision, rappel, F1-score.
- **Visualisations Clés :** Courbes ROC, diagrammes d'importance des features (pour comprendre quelles raies sont les plus décisives).
- **Rapports Automatisés :** Génération de rapports PDF ou HTML pour chaque exécution, archivant les résultats et les graphiques.