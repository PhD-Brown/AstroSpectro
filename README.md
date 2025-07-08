**J'ai conçu et implémenté un pipeline d'IA de bout en bout pour la classification de types stellaires à partir de spectres bruts du relevé LAMOST DR5. Le pipeline gère l'acquisition de données, le prétraitement adaptatif de multiples formats FITS, l'extraction de features basées sur la détection de raies spectrales, et l'entraînement d'un classifieur RandomForest. Le système est modulaire, robuste, et capable de traiter des lots de données de manière itérative sans redondance.**

1) Télécharge des données brutes de manière robuste.
2) Gère l'état des données pour éviter les doublons.
3) Lit et parse des fichiers scientifiques complexes (FITS) en gérant leurs formats hétérogènes.
4) Normalise les données.
5) Détecte des signaux physiques (les raies spectrales).
6) Transforme ces signaux en features numériques.
7) Associe ces features à des labels provenant d'un catalogue externe.
8) Sauvegarde le tout dans un dataset propre, prêt pour l'entraînement d'un modèle d'IA.
