import os
import pandas as pd
import json
import hashlib
from datetime import datetime, timezone

# Importation des composants du pipeline AstroSpectro
from tools.dataset_builder import DatasetBuilder
from pipeline.processing import ProcessingPipeline
from pipeline.classifier import SpectralClassifier
from tools.generate_catalog_from_fits import generate_catalog_from_fits

# Importation d'outils de scikit-learn pour la validation et le rapport
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

class MasterPipeline:
    """
    Pipeline principal orchestrant l'ensemble du traitement :
    sélection d'un nouveau lot de spectres, génération du catalogue local,
    prétraitement et extraction des features, entraînement du modèle et production des rapports.
    Cette classe centralise la logique métier afin de simplifier l'utilisation depuis un notebook.
    """
    def __init__(self, raw_data_dir, catalog_dir, processed_dir, models_dir, reports_dir):
        """
        Initialise le pipeline principal avec les chemins requis et instancie les composants nécessaires.
        :param raw_data_dir: Chemin du répertoire contenant les données brutes (.fits.gz).
        :param catalog_dir: Chemin du répertoire contenant les catalogues (dont le journal des spectres).
        :param processed_dir: Chemin du répertoire où stocker les données de sortie (features).
        :param models_dir: Chemin du répertoire où sauvegarder les modèles entraînés.
        :param reports_dir: Chemin du répertoire où sauvegarder les rapports de session.
        """
        self.raw_data_dir = raw_data_dir
        self.catalog_dir = catalog_dir
        self.processed_dir = processed_dir
        self.models_dir = models_dir
        self.reports_dir = reports_dir
        # Instanciation du gestionnaire de dataset (sélection de batch et journalisation)
        self.builder = DatasetBuilder(raw_data_dir=self.raw_data_dir, catalog_dir=self.catalog_dir)
        # Stockage des données du pipeline courant
        self.current_batch = []
        self.master_catalog_path = os.path.join(self.catalog_dir, "master_catalog_temp.csv")
        self.last_features_path = None
        self.features_df = pd.DataFrame()  # DataFrame des features extrait du dernier lot
        # Assurer la création des répertoires de sortie au cas où
        os.makedirs(self.catalog_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

    def _select_batch(self, batch_size=500, strategy="random"):
        """
        Étape 2 : Sélectionne un nouveau lot de spectres n'ayant jamais été utilisés pour l'entraînement.
        Met à jour self.current_batch avec la liste des chemins relatifs des fichiers sélectionnés.
        :param batch_size: Nombre de spectres souhaité dans le lot.
        :param strategy: Stratégie de sélection ('random' ou 'first').
        :return: La liste des chemins de spectres sélectionnés (peut être vide si aucun spectre disponible).
        """
        print("\n=== ÉTAPE 2 : CRÉATION D'UN NOUVEAU LOT DE SPECTRES NON DÉJÀ UTILISÉS ===")
        new_batch_paths = self.builder.get_new_training_batch(batch_size=batch_size, strategy=strategy)
        self.current_batch = new_batch_paths if new_batch_paths else []
        if self.current_batch:
            print(f"\n{len(self.current_batch)} nouveaux spectres proposés pour traitement.")
            print(f"Exemple : {self.current_batch[0]}")
        else:
            print("\nAucun nouveau spectre à traiter : le pipeline est à jour.")
        return self.current_batch

    def _generate_local_catalog(self):
        """
        Étape 3 : Génère un catalogue local (CSV temporaire) à partir des en-têtes FITS des spectres du lot courant.
        Le fichier est créé dans self.catalog_dir sous le nom 'master_catalog_temp.csv'.
        :return: Chemin du catalogue généré, ou None si aucun lot courant n'est défini.
        """
        print("\n=== ÉTAPE 3 : GÉNÉRATION DU CATALOGUE LOCAL DE HEADERS ===\n")
        if not self.current_batch:
            print("Veuillez d'abord sélectionner un lot de travail (aucun 'current_batch' défini).")
            return None
        # Préparation du chemin de sortie pour le catalogue local
        output_path = self.master_catalog_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Chemins complets des fichiers fits.gz à partir des chemins relatifs du batch
        full_paths = [os.path.join(self.raw_data_dir, path) for path in self.current_batch]
        # Génération du catalogue à partir des en-têtes FITS
        generate_catalog_from_fits(full_paths, output_path)
        print(f"\nCatalogue master local créé : {output_path}")
        return output_path

    def _process_batch(self):
        """
        Étape 4 : Lance le pipeline de traitement des spectres du lot courant.
        Charge le catalogue local généré, exécute le pré-traitement et l'extraction de features,
        puis enregistre le DataFrame de features résultant dans un fichier CSV horodaté.
        :return: DataFrame des features extraites (peut être vide si échec ou aucun spectre traité).
        """
        if not self.current_batch:
            print("Veuillez sélectionner un lot de spectres avant de lancer le traitement.")
            return pd.DataFrame()
        print("\n--- ÉTAPE 4 : Lancement du pipeline de traitement ---")
        # Chargement du catalogue local temporaire
        try:
            master_catalog_df = pd.read_csv(self.master_catalog_path, sep='|')
            print(f"  > Catalogue temporaire chargé avec succès ({len(master_catalog_df)} entrées).")
        except FileNotFoundError:
            print("  > ERREUR : Le catalogue temporaire est introuvable. Assurez-vous que l'étape 3 a été exécutée.")
            return pd.DataFrame()
        # Initialisation du pipeline de traitement
        processing_pipeline = ProcessingPipeline(raw_data_dir=self.raw_data_dir, master_catalog_df=master_catalog_df)
        # Exécution du traitement sur le lot courant
        features_df = processing_pipeline.run(self.current_batch)
        # Sauvegarde des features extraites si disponibles
        if not features_df.empty:
            print("\n--- Aperçu du dataset de features généré ---")
            # Afficher les premières lignes pour vérification (dans la console, on affiche en texte brut)
            print(features_df.head().to_string(index=False))
            # Sauvegarder le DataFrame des features dans un fichier CSV avec horodatage
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            features_filename = f"features_{timestamp}.csv"
            self.last_features_path = os.path.join(self.processed_dir, features_filename)
            features_df.to_csv(self.last_features_path, index=False)
            print(f"\nDataset de features sauvegardé dans : {self.last_features_path}")
        else:
            print("\n  > Aucun feature n'a pu être extrait du lot courant.")
            self.last_features_path = None
        # Stocker le DataFrame de features en attribut (pour consultation éventuelle)
        self.features_df = features_df
        return features_df

    def _train_and_evaluate(self, n_estimators=200):
        """
        Étape 5 : Entraîne un modèle de classification sur les features extraites et évalue sa performance.
        Effectue le nettoyage des données (étiquettes invalides, classes rares) puis entraîne un RandomForest.
        :return: Un tuple (clf, feature_cols, X, y, processed_files) si entraînement réalisé, None si pas de données.
        """
        print("\n--- ÉTAPE 5 : Entraînement et Évaluation du modèle ---")
        # Charger le dernier fichier de features généré si le DataFrame n'est pas déjà disponible
        df_features = pd.DataFrame()
        if not self.features_df.empty:
            df_features = self.features_df.copy()
        elif self.last_features_path:
            # Si un chemin de fichier de features est disponible, charger depuis ce fichier
            try:
                df_features = pd.read_csv(self.last_features_path)
            except Exception as e:
                df_features = pd.DataFrame()
        else:
            # Sinon, chercher le plus récent fichier de features dans le répertoire processed
            feature_files = [os.path.join(self.processed_dir, f) for f in os.listdir(self.processed_dir) if f.startswith("features_") and f.endswith(".csv")]
            if feature_files:
                latest_feature_file = max(feature_files, key=os.path.getctime)
                print(f"--- Chargement du dataset : {os.path.basename(latest_feature_file)} ---")
                df_features = pd.read_csv(latest_feature_file)
            else:
                print("ERREUR : Aucun fichier de features trouvé pour l'entraînement.")
                return None
        # Création de la colonne 'label' à partir de la sous-classe si disponible
        if 'subclass' in df_features.columns:
            df_features['label'] = df_features['subclass'].astype(str).str[0]
        else:
            df_features['label'] = 'UNKNOWN'
        # Filtrer les données invalides pour l'entraînement
        initial_count = len(df_features)
        df_trainable = df_features[df_features["label"].notnull() & ~df_features["label"].isin(['U', 'N', 'n'])].copy()
        print(f"  > {initial_count - len(df_trainable)} lignes avec des labels invalides ou nuls supprimées.")
        
        # Supprimer les classes trop peu représentées (moins de 10 occurrences)
        label_counts = df_trainable["label"].value_counts()
        rare_labels = label_counts[label_counts < 10].index.tolist()
        if rare_labels:
            print(f"  > Suppression des classes trop rares : {rare_labels}")
            df_trainable = df_trainable[~df_trainable["label"].isin(rare_labels)]
        if df_trainable.empty:
            print("\n  > Pas assez de données valides pour lancer l'entraînement.")
            return None
        
        # Préparer les matrices de features (X) et les labels (y)
        feature_cols = [col for col in df_trainable.columns if col.startswith('feature_')]
        X = df_trainable[feature_cols].values
        y = df_trainable["label"].values
        print(f"\nFeatures utilisées : {feature_cols}")
        print(f"Nombre d'échantillons final : {X.shape[0]}, Nombre de features : {X.shape[1]}")
        
        # Entraîner le classifieur
        clf = SpectralClassifier(n_estimators=n_estimators)
        clf.train_and_evaluate(X, y, test_size=0.25)
        
        # Retourner le modèle entraîné et les données pour l'étape suivante
        return clf, feature_cols, X, y, df_trainable["file_path"].tolist()

    def _log_and_report(self, clf, feature_cols, X, y, processed_files, n_estimators=200):
        """
        Étapes 6 et 7 : Met à jour le journal des spectres traités et génère un rapport de session.
        Sauvegarde le modèle entraîné, calcule un hash pour vérification, et exporte un rapport JSON contenant les paramètres et métriques.
        :param clf: Le classifieur SpectralClassifier entraîné.
        :param feature_cols: La liste des colonnes de features utilisées pour l'entraînement.
        :param X: Les features d'entrée (numpy array) ayant servi à l'entraînement complet.
        :param y: Les labels correspondants.
        :param processed_files: Liste des chemins de spectres ayant été traités (pour mise à jour du log).
        :return: Chemin du rapport de session généré.
        """
        if clf is None or processed_files is None:
            print("Aucun modèle entraîné ou fichiers traités à consigner. Rapport non généré.")
            return None
        
        # Sauvegarde du modèle entraîné sur le disque
        model_path = os.path.join(self.models_dir, "spectral_classifier.pkl")
        clf.save_model(model_path)
        
        # Mise à jour du journal des spectres utilisés
        print("\n--- ÉTAPE 6 : Mise à jour du Journal des Spectres Utilisés ---")
        self.builder.update_trained_log(processed_files)
        
        # Génération du rapport de session
        print("\n--- ÉTAPE 7 : Génération du Rapport de Session ---")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        
        # Calcul du hash MD5 du modèle sauvegardé (pour suivi d'intégrité)
        model_hash = "N/A"
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model_bytes = f.read()
                model_hash = hashlib.md5(model_bytes).hexdigest()
            print(f"  > Hash MD5 du modèle : {model_hash}")
            
        # Calcul des métriques de performance sur la base de test (en refaisant un split pour extraction métriques)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        predictions = clf.model.predict(X_test)
        report_dict = classification_report(y_test, predictions, labels=clf.class_labels, zero_division=0, output_dict=True)
        metrics_summary = report_dict.get("weighted avg", {})
        print(f"  > Métriques extraites : Accuracy = {report_dict.get('accuracy', 0):.2f}")
        
        # Calcul de la matrice de confusion
        cm = confusion_matrix(y_test, predictions, labels=clf.class_labels)
        cm_list = cm.tolist()
        
        # Construction du contenu du rapport
        session_report = {
            "session_id": timestamp,
            "date_utc": datetime.now(timezone.utc).isoformat(),
            "model_path": model_path,
            "model_hash_md5": model_hash,
            "training_set_size": int(len(X_train)),
            "test_set_size": int(len(X_test)),
            "total_spectra_processed": len(processed_files),
            "feature_columns": feature_cols,
            "class_labels": clf.class_labels,
            "n_estimators": n_estimators,
            "classification_report": report_dict,
            "accuracy": metrics_summary.get("precision", 0),
            "recall": metrics_summary.get("recall", 0),
            "f1_score": metrics_summary.get("f1-score", 0),
            "support": metrics_summary.get("support", 0),
            "model_params": clf.model.get_params(),
            "model_type": clf.model.__class__.__name__,
            "metrics": metrics_summary,
            "confusion_matrix": cm_list,
            "confusion_matrix_labels": clf.class_labels,
            "processed_files_list": processed_files
        }
        
        # Sauvegarde du rapport sur le disque en JSON
        report_filename = f"session_report_{timestamp}.json"
        report_path = os.path.join(self.reports_dir, report_filename)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(session_report, f, indent=4)
        print(f"\nRapport de session sauvegardé dans : {report_path}")
        print("\n\nSESSION DE RECHERCHE TERMINÉE")
        return report_path

    def run_full_pipeline(self, batch_size=500, strategy="random", n_estimators=200):
        """
        Exécute l'ensemble du pipeline de manière séquentielle (Étapes 2 à 7).
        Permet en un appel de lancer le batch, la génération du catalogue, le traitement, l'entraînement et le reporting.
        :param batch_size: Taille du lot de spectres à sélectionner.
        :param strategy: Stratégie de sélection des spectres ('random' ou 'first').
        """
        # Étape 2 : Sélection du lot
        batch = self._select_batch(batch_size=batch_size, strategy=strategy)
        if not batch:
            return  # Aucune donnée à traiter, fin du pipeline
        # Étape 3 : Génération du catalogue
        self._generate_local_catalog()
        # Étape 4 : Traitement du lot et extraction des features
        features_df = self._process_batch()
        if features_df.empty:
            return  # Pas de features extraites, fin du pipeline
        # Étape 5 : Entraînement du modèle
        result = self._train_and_evaluate(n_estimators=n_estimators)
        if not result:
            return  # Entraînement non réalisé
        clf, feature_cols, X, y, processed_files = result
        # Étapes 6-7 : Journalisation et rapport
        self._log_and_report(clf, feature_cols, X, y, processed_files, n_estimators=n_estimators)
