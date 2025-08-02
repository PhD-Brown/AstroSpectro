import os
import pandas as pd
import json
import hashlib
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output
from datetime import datetime, timezone

# Importation des composants du pipeline AstroSpectro
from tools.dataset_builder import DatasetBuilder
from pipeline.processing import ProcessingPipeline
from pipeline.classifier import SpectralClassifier
from tools.generate_catalog_from_fits import generate_catalog_from_fits

# Importation d'outils de scikit-learn pour la validation et le rapport
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

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

    def _train_and_evaluate(self, model_type='RandomForest', n_estimators=100):
        """
        Étape 5 : Orchestre l'entraînement et l'évaluation du modèle spécifié.
        """
        print(f"\n=== ÉTAPE 5 : ENTRAÎNEMENT DU CLASSIFICATEUR (Modèle: {model_type}, n_estimators: {n_estimators}) ===")

        
        # --- 1. Chargement des données ---
        # La logique de chargement que tu avais est très robuste, on la garde.
        if not self.features_df.empty:
            df_features = self.features_df.copy()
        elif self.last_features_path and os.path.exists(self.last_features_path):
            print(f"--- Chargement du dernier dataset de features : {os.path.basename(self.last_features_path)} ---")
            df_features = pd.read_csv(self.last_features_path)
        else:
            print("ERREUR : Aucun DataFrame de features disponible pour l'entraînement.")
            return None
            
        # --- 2. Entraînement via le SpectralClassifier ---
        # On instancie le classifieur.
        clf = SpectralClassifier(model_type=model_type)

        # On appelle la nouvelle méthode train_and_evaluate en lui passant le DataFrame COMPLET.
        # Elle n'attend plus X, y, ou test_size.
        # Elle nous retournera tout ce dont on a besoin pour la suite.
        result_tuple = clf.train_and_evaluate(df_features, n_estimators=n_estimators)
        
        # --- 3. Gestion des résultats ---
        # On vérifie si l'entraînement a réussi et a retourné des résultats.
        if result_tuple is None:
            print("\n  > L'entraînement a été annulé (probablement pas assez de données valides).")
            return None
            
        # On déballe les résultats retournés par le classifieur.
        feature_cols_used, X_used, y_used, processed_files = result_tuple
        
        print("\n  > Entraînement et tuning terminés avec succès.")
        
        # On retourne toutes les informations nécessaires pour la dernière étape (_log_and_report).
        return clf, feature_cols_used, X_used, y_used, processed_files

    def _log_and_report(self, clf, feature_cols, X, y, processed_files):
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
        # On vérifie si l'objet classifieur existe ET si la liste des fichiers n'est pas vide.
        if clf is None or not processed_files:
            print("\n> Aucun modèle entraîné ou fichiers traités à consigner. Rapport non généré.")
            return None
        
        # Mise à jour du journal des spectres utilisés
        print("\n--- ÉTAPE 6 : Mise à jour du Journal des Spectres Utilisés ---")
        self.builder.update_trained_log(processed_files)
        
        # --- ÉTAPE 7: Génération du Rapport de Session ---
        print("\n--- ÉTAPE 7 : Génération du Rapport de Session ---")
        
        # On définit UN SEUL timestamp pour toute la session
        session_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        model_filename = f"spectral_classifier_{clf.model_type.lower()}_{session_timestamp}.pkl"
        model_path = os.path.join(self.models_dir, model_filename)
        clf.save_model(model_path)
        
        # 2. Calcul du hash du modèle
        model_hash = "N/A"
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model_hash = hashlib.md5(f.read()).hexdigest()
            print(f"  > Hash MD5 du modèle : {model_hash}")
            
        # 3. Récupération des métriques de performance
        # On refait un split pour être sûr d'avoir les mêmes données que lors de l'évaluation
        # Gestion des labels encodés pour XGBoost
        if clf.model_type == 'XGBoost':
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded)
            predictions = clf.model_pipeline.predict(X_test)
            report_dict = classification_report(y_test, predictions, target_names=clf.class_labels, zero_division=0, output_dict=True)
            cm = confusion_matrix(y_test, predictions)
        else: # Pour RandomForest
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
            predictions = clf.model_pipeline.predict(X_test)
            report_dict = classification_report(y_test, predictions, labels=clf.class_labels, zero_division=0, output_dict=True)
            cm = confusion_matrix(y_test, predictions, labels=clf.class_labels)

        accuracy = report_dict.get('accuracy', 0)
        print(f"  > Métriques extraites : Accuracy = {accuracy:.4f}")
        
        # 4. Construction du rapport
        session_report = {
            "session_id": session_timestamp,
            "model_type": clf.model_type,
            "model_path": model_path,
            "model_hash_md5": model_hash,
            "total_spectra_processed": len(processed_files),
            "training_set_size": int(len(X_train)),
            "test_set_size": int(len(X_test)),
            "feature_columns": feature_cols,
            "class_labels": clf.class_labels.tolist() if isinstance(clf.class_labels, np.ndarray) else clf.class_labels,
            "best_model_params": clf.best_params_,
            "accuracy": accuracy,
            "classification_report": report_dict,
            "confusion_matrix": cm.tolist(),
        }
        
        # 5. Sauvegarde du rapport JSON
        report_filename = f"session_report_{clf.model_type.lower()}_{session_timestamp}.json"
        report_path = os.path.join(self.reports_dir, report_filename)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(session_report, f, indent=4)
        print(f"\nRapport de session sauvegardé dans : {report_path}")
        print("\n\nSESSION DE RECHERCHE TERMINÉE")
        return report_path
    
    def run_full_pipeline(self, batch_size=500, strategy="random", model_type='RandomForest'):
        """
        Exécute l'ensemble du pipeline de manière séquentielle.
        """
        # Étape 2 : Sélection du lot
        batch = self._select_batch(batch_size=batch_size, strategy=strategy)
        if not batch: return

        # Étape 3 : Génération du catalogue
        self._generate_local_catalog()

        # Étape 4 : Traitement du lot
        features_df = self._process_batch()
        if features_df.empty: return

        # Étape 5 : Entraînement du modèle
        # On passe le model_type à la méthode d'entraînement
        result = self._train_and_evaluate(model_type=model_type)
        if not result: return

        clf, feature_cols, X, y, processed_files = result

        # Étapes 6-7 : Journalisation et rapport
        self._log_and_report(clf, feature_cols, X, y, processed_files)
        
    def interactive_training_runner(self):
        """
        Crée et affiche une interface interactive dans un notebook Jupyter
        pour lancer le pipeline d'entraînement avec des paramètres choisis.
        """
        # --- 1. Widgets de contrôle ---
        model_choice = widgets.Dropdown(options=['RandomForest', 'XGBoost'], value='XGBoost', description='Modèle:')
        n_estimators_widget = widgets.IntText(value=200, description='N Estimators:', style={'description_width': 'initial'})
        run_button = widgets.Button(description="Lancer l'entraînement", button_style='success', icon='play')
        output_area = widgets.Output()

        # --- 2. Fonction de callback ---
        def on_run_button_clicked(b):
            with output_area:
                clear_output(wait=True)
                selected_model = model_choice.value
                n_estimators = n_estimators_widget.value
                
                # On appelle la méthode d'entraînement de CETTE instance de MasterPipeline
                training_results = self._train_and_evaluate(model_type=selected_model, n_estimators=n_estimators)

                if training_results is not None:
                    clf, feature_cols, X, y, processed_files = training_results
                    self._log_and_report(clf, feature_cols, X, y, processed_files)
                else:
                    print("\n--- SESSION TERMINÉE SANS ENTRAÎNEMENT ---")

        # --- 3. Lier et afficher ---
        run_button.on_click(on_run_button_clicked)
        
        controls = widgets.HBox([model_choice, n_estimators_widget, run_button])
        app_layout = widgets.VBox([controls, output_area], layout={
            'border': '1px solid #444',
            'padding': '10px',
            'width': '100%'
        })
        
        display(app_layout)
