import numpy as np

class FeatureEngineer:
    def __init__(self):
        # On fixe les raies qu'on veut utiliser pour générer les features
        self.lines = ["Hα", "Hβ", "CaII K", "CaII H"]

    def extract_features(self, matched_lines):
        """
        Transforme les raies détectées en un vecteur de features utilisable en ML.
        Les longueurs d'onde des raies détectées deviennent des features, None devient 0.
        """
        features = []
        for line in self.lines:
            wl = matched_lines.get(line)
            if wl is None:
                features.append(0.0)
            else:
                features.append(wl)
        return features

    def batch_features(self, matched_lines_list):
        """
        Permet de traiter une liste de spectres en batch.
        """
        return np.array([self.extract_features(ml) for ml in matched_lines_list])
