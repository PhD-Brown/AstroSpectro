#!/bin/bash

echo "--- Lancement du script post-création pour AstroSpectro ---"

# --- 1. Installation des dépendances (déjà fait, mais on s'assure que tout est là) ---
echo "[1/3] Installation des dépendances depuis requirements.txt..."
# On utilise le pip du venv
/workspaces/AstroSpectro/venv/bin/pip install -r requirements.txt

# --- 2. Enregistrement du venv comme noyau Jupyter ---
echo "[2/3] Enregistrement de l'environnement virtuel comme noyau Jupyter..."
/workspaces/AstroSpectro/venv/bin/python -m ipykernel install --user --name=AstroSpectro-venv --display-name "Python (AstroSpectro venv)"

# --- 3. Téléchargement d'un jeu de données initial ---
echo "[3/3] Téléchargement d'un petit jeu de données de démarrage..."
/workspaces/AstroSpectro/venv/bin/python src/tools/dr5_downloader.py --limit 2 --max-spectres 20

# --- Message final ---
echo "Environnement AstroSpectro prêt à l'emploi !"
