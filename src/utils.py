import os
import sys

def setup_project_env():
    """
    Initialise l'environnement du projet pour un notebook.
    - Trouve la racine du projet.
    - Ajoute les dossiers 'src' et 'src/pipeline' au sys.path.
    - Retourne un dictionnaire des chemins importants.
    """
    try:
        # Trouve la racine du projet en remontant depuis le script actuel
        current_path = os.getcwd()
        project_root = current_path
        # La boucle s'arrête quand on trouve un dossier 'src' à l'intérieur
        while not os.path.isdir(os.path.join(project_root, 'src')):
            parent = os.path.dirname(project_root)
            if parent == project_root: # On a atteint la racine du système de fichiers
                raise FileNotFoundError("Impossible de trouver la racine du projet (contenant le dossier 'src').")
            project_root = parent
        
        # Définir les chemins importants
        paths = {
            "PROJECT_ROOT": project_root,
            "SRC_DIR": os.path.join(project_root, 'src'),
            "PIPELINE_DIR": os.path.join(project_root, 'src', 'pipeline'),
            "TOOLS_DIR": os.path.join(project_root, 'src', 'tools'),
            "DATA_DIR": os.path.join(project_root, 'data'),
            "RAW_DATA_DIR": os.path.join(project_root, 'data', 'raw'),
            "CATALOG_DIR": os.path.join(project_root, 'data', 'catalog'),
            "PROCESSED_DIR": os.path.join(project_root, 'data', 'processed'),
            "MODELS_DIR": os.path.join(project_root, 'data', 'models'),
            "REPORTS_DIR": os.path.join(project_root, 'data', 'reports'),
            "NOTEBOOKS_DIR": os.path.join(project_root, 'notebooks'),
            "LOGS_DIR": os.path.join(project_root, 'logs'),
        }

        # Ajouter les chemins au sys.path pour les imports
        if paths["SRC_DIR"] not in sys.path:
            sys.path.append(paths["SRC_DIR"])
        
        print(f"[INFO] Racine du projet détectée : {paths['PROJECT_ROOT']}")
        print(f"[INFO] Dossier 'src' ajouté au sys.path.")
        
        return paths

    except Exception as e:
        print(f"ERREUR lors de l'initialisation de l'environnement : {e}")
        return None