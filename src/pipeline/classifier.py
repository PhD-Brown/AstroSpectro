import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

class SpectralClassifier:
    """
    Encapsule un classifieur RandomForest pour les spectres,
    incluant l'entraînement, l'évaluation et la sauvegarde.
    """
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, class_weight='balanced')
        self.class_labels = None

    def train(self, X_train, y_train):
        """Entraîne le modèle sur les données d'entraînement."""
        print(f"  > Entraînement du modèle sur {len(X_train)} échantillons...")
        self.model.fit(X_train, y_train)
        self.class_labels = sorted(list(set(y_train)))
        print("  > Modèle entraîné.")

    def evaluate(self, X_test, y_test):
        """Évalue le modèle et affiche un rapport de classification et une matrice de confusion."""
        print("\n--- Rapport d'Évaluation ---")
        predictions = self.model.predict(X_test)
        
        # Rapport textuel
        report = classification_report(y_test, predictions, labels=self.class_labels, zero_division=0)
        print(report)
        
        # Matrice de confusion visuelle
        cm = confusion_matrix(y_test, predictions, labels=self.class_labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_labels, yticklabels=self.class_labels)
        plt.xlabel('Prédiction')
        plt.ylabel('Vraie valeur')
        plt.title('Matrice de Confusion')
        plt.show()

    def train_and_evaluate(self, X, y, test_size=0.25):
        """Pipeline complet d'entraînement et d'évaluation."""
        if len(set(y)) < 2:
            print("ERREUR : Moins de deux classes uniques dans les données. Impossible d'entraîner ou d'évaluer.")
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        self.train(X_train, y_train)
        self.evaluate(X_test, y_test)

    def save_model(self, path="model.pkl"):
        """Sauvegarde le modèle entraîné sur le disque."""
        joblib.dump(self, path)
        print(f"  > Modèle sauvegardé dans : {path}")

    @staticmethod
    def load_model(path="model.pkl"):
        """Charge un modèle depuis le disque."""
        model = joblib.load(path)
        print(f"  > Modèle chargé depuis : {path}")
        return model