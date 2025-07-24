import numpy as np

class FeatureEngineer:
    def __init__(self):
        # La liste des raies de base que nous extrayons
        self.base_lines = ["Hα", "Hβ", "CaII K", "CaII H"]
        
        # La définition de nos nouvelles features de ratio
        self.ratio_definitions = {
            "ratio_CaK_Hbeta": ("CaII K", "Hβ"),
            "ratio_Halpha_Hbeta": ("Hα", "Hβ")
        }
        
        # La liste complète et ordonnée des noms de features que ce module va générer
        self.feature_names = [f"feature_{line.replace(' ', '')}" for line in self.base_lines] + \
                             [f"feature_{name}" for name in self.ratio_definitions.keys()]

    def extract_features(self, matched_lines):
        """
        Transforme les raies détectées en un vecteur de features enrichi,
        incluant les prominences brutes et des ratios calculés.
        """
        # --- 1. Extraire les prominences de base ---
        prominences = {}
        for line in self.base_lines:
            match_data = matched_lines.get(line)
            # On stocke la prominence (deuxième élément du tuple) ou 0.0 si la raie est absente
            prominences[line] = match_data[1] if match_data else 0.0
        
        # Le vecteur initial contient les 4 prominences, dans le bon ordre
        features = [prominences[line] for line in self.base_lines]
        
        # --- 2. Calculer et ajouter les features de ratio ---
        epsilon = 1e-6 # Pour éviter la division par zéro
        
        for name, (numerator_line, denominator_line) in self.ratio_definitions.items():
            numerator_val = prominences.get(numerator_line, 0.0)
            denominator_val = prominences.get(denominator_line, 0.0)
            
            ratio = numerator_val / (denominator_val + epsilon)
            features.append(ratio)
            
        return features

    def batch_features(self, matched_lines_list):
        """
        Permet de traiter une liste de spectres en batch.
        """
        return np.array([self.extract_features(ml) for ml in matched_lines_list])
