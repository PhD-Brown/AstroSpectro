import sys
import os

# Ajoute 'src' au path pour que l'app puisse trouver les modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from utils import setup_project_env
from tools.visualizer import AstroVisualizer

def main():
    """
    Fonction principale pour lancer l'application Streamlit.
    """
    # On a besoin de streamlit, on s'assure qu'il est installé
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit n'est pas installé. Veuillez l'installer avec : pip install streamlit")
        return

    # Initialisation de l'environnement et des outils
    paths = setup_project_env()
    if paths:
        visualizer = AstroVisualizer(paths)
        # On lance l'application définie dans le visualizer
        visualizer.app()

if __name__ == "__main__":
    main()