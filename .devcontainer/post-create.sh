#!/bin/bash

echo "--- Lancement du script post-création pour AstroSpectro ---"

# Le venv et les dépendances sont déjà installés par le Dockerfile.
# On se contente de télécharger un petit jeu de données.
echo "[1/1] Téléchargement d'un petit jeu de données de démarrage..."
venv/bin/python src/tools/dr5_downloader.py --limit 2 --max-spectres 20

echo "Environnement AstroSpectro prêt à l'emploi !"