import numpy as np

class FeatureEngineer:
    def __init__(self):
        # On fixe les raies qu'on veut utiliser pour générer les features
        self.lines = ["Hα", "Hβ", "CaII K", "CaII H"]

    def extract_features(self, matched_lines):
        """
        Transforme les raies détectées en un vecteur de features utilisable en ML.
        Les longueurs d'onde des raies détectées deviennent des features, None devient 0.
        La feature est maintenant la PROMINENCE (force) de la raie.
        """
        features = []
        for line in self.lines:
            # matched_lines contient maintenant un tuple (wavelength, prominence) ou None
            match_data = matched_lines.get(line)
            
            if match_data is None:
                features.append(0.0) # La raie est absente
            else:
                # On extrait la prominence (le deuxième élément du tuple)
                wavelength, prominence = match_data
                features.append(prominence)
                
        return features

    def batch_features(self, matched_lines_list):
        """
        Permet de traiter une liste de spectres en batch.
        """
        return np.array([self.extract_features(ml) for ml in matched_lines_list])
