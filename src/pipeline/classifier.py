import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb

# --- Imports Scikit-learn ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# --- Imports Imbalanced-learn ---
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

class SpectralClassifier:
    """
    Encapsule un pipeline de classification complet, incluant le scaling des données,
    le tuning des hyperparamètres via GridSearchCV, l'entraînement et l'évaluation.
    """
    def __init__(self, model_type='RandomForest', random_state=42):
        if model_type not in ['RandomForest', 'XGBoost']:
            raise ValueError("model_type doit être 'RandomForest' ou 'XGBoost'")
        self.model_type = model_type
        self.random_state = random_state
        self.model_pipeline = None 
        self.class_labels = None
        self.best_params_ = None
        self.feature_names_used = None

    def _clean_and_filter_data(self, df):
        """
        Nettoie et filtre le DataFrame pour ne garder que les données exploitables.
        (Méthode privée pour la clarté)
        """
        # Création du label à partir de 'subclass'
        if 'subclass' in df.columns:
            df['label'] = df['subclass'].astype(str).str[0]
        else:
            print("AVERTISSEMENT: Colonne 'subclass' non trouvée. Impossible de créer les labels.")
            return None
        
        # Filtrer les labels invalides (ex: 'U' pour UNKNOWN, 'n' pour non-stellar)
        initial_count = len(df)
        df_trainable = df[df["label"].notnull() & ~df["label"].isin(['U', 'n'])].copy()
        print(f"  > {initial_count - len(df_trainable)} lignes avec des labels invalides ou nuls supprimées.")

        # Filtrer les classes trop rares (moins de 10 échantillons)
        label_counts = df_trainable["label"].value_counts()
        rare_labels = label_counts[label_counts < 10].index.tolist()
        if rare_labels:
            print(f"  > Suppression des classes trop rares : {rare_labels}")
            df_trainable = df_trainable[~df_trainable["label"].isin(rare_labels)]
        
        if len(df_trainable) < 20: # Seuil de sécurité
            print("ERREUR : Pas assez de données valides après nettoyage pour entraîner un modèle.")
            return None
        
        return df_trainable

    def _prepare_features_and_labels(self, df_trainable):
        """
        Sélectionne les colonnes de features et retourne les matrices X et y.
        (Méthode privée pour la clarté)
        """
        engineered_features = [col for col in df_trainable.columns if col.startswith('feature_')]
        metadata_features = ['redshift', 'snr_g', 'snr_r', 'snr_i', 'seeing']
        
        final_metadata_features = [col for col in metadata_features if col in df_trainable.columns]
        
        all_features_to_use = engineered_features + final_metadata_features
        
        # On supprime les doublons au cas où et on garde l'ordre
        all_features_to_use = sorted(list(set(all_features_to_use)))
        
        self.feature_names_used = all_features_to_use # On stocke les noms pour plus tard
        
        print("\n--- Préparation pour l'entraînement ---")
        print(f"Features utilisées ({len(all_features_to_use)}) : {all_features_to_use}")

        X = df_trainable[all_features_to_use].values
        y = df_trainable["label"].values
        
        return X, y

    def train_and_evaluate(self, features_df, test_size=0.25, n_estimators=100):
        """
        Le pipeline principal de cette classe. Prend le DataFrame brut, le nettoie,
        tune les hyperparamètres, entraîne le meilleur modèle et l'évalue.
        """
        df_trainable = self._clean_and_filter_data(features_df)
        if df_trainable is None:
            return None # Arrêt si les données ne sont pas valides

        X, y = self._prepare_features_and_labels(df_trainable)
        
        # --- Gestion des labels pour XGBoost ---
        if self.model_type == 'XGBoost':
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            self.class_labels = le.classes_
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=self.random_state, stratify=y_encoded)
        else: # Pour RandomForest
            self.class_labels = sorted(list(set(y)))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=self.random_state, stratify=y)
            
        # --- Définition du pipeline et de la grille de paramètres ---
        if self.model_type == 'RandomForest':
            # On passe n_estimators directement au constructeur du modèle
            clf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=self.random_state, class_weight='balanced')
            # La grille de recherche ne contient plus n_estimators
            param_grid = {
                'clf__max_depth': [20, None],
                'clf__min_samples_leaf': [1, 3]
            }
        elif self.model_type == 'XGBoost':
            clf_model = xgb.XGBClassifier(n_estimators=n_estimators, random_state=self.random_state, use_label_encoder=False, eval_metric='mlogloss')
            param_grid = {
                'clf__max_depth': [5, 10],
                'clf__learning_rate': [0.1, 0.05]
            }
        
        pipeline = ImbPipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=self.random_state)),
            ('clf', clf_model) # On utilise le modèle qu'on vient de créer
        ])


        print(f"\n--- [Tuning] Recherche des meilleurs hyperparamètres pour {self.model_type} (n_estimators={n_estimators} fixé) ---")
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=2, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        print(f"\n  > Meilleurs paramètres trouvés : {grid_search.best_params_}")
        print(f"  > Meilleur score de précision (CV) : {grid_search.best_score_:.4f}")
        
        self.model_pipeline = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_
        
        print(f"\n--- [Évaluation] Performance de {self.model_type} sur le jeu de test ---")
        self.evaluate(X_test, y_test)
        
        processed_files = df_trainable["file_path"].tolist()
        return self.feature_names_used, X, y, processed_files

    def evaluate(self, X_test, y_test):
        predictions = self.model_pipeline.predict(X_test)
        print("\n--- Rapport d'Évaluation ---")
        
        # On doit utiliser target_names si les labels sont encodés (XGBoost)
        if self.model_type == 'XGBoost':
            report = classification_report(y_test, predictions, target_names=self.class_labels, zero_division=0)
            cm = confusion_matrix(y_test, predictions)
        else: # Pour RandomForest, les labels sont déjà du texte
            report = classification_report(y_test, predictions, labels=self.class_labels, zero_division=0)
            cm = confusion_matrix(y_test, predictions, labels=self.class_labels)

        print(report)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_labels, yticklabels=self.class_labels)
        plt.xlabel('Prédiction')
        plt.ylabel('Vraie Valeur')
        plt.title(f'Matrice de Confusion du Modèle {self.model_type} Optimisé')
        plt.show()

    def save_model(self, path="model.pkl"):
        """Sauvegarde l'objet SpectralClassifier complet sur le disque."""
        joblib.dump(self, path)
        print(f"  > Modèle sauvegardé dans : {path}")

    @staticmethod
    def load_model(path="model.pkl"):
        """Charge un objet SpectralClassifier depuis le disque."""
        model = joblib.load(path)
        print(f"  > Modèle chargé depuis : {path}")
        return model