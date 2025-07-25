#!/bin/bash

# Affiche un message de bienvenue
echo "--- Lancement du script post-création pour AstroSpectro ---"

# --- 1. Création de l'environnement virtuel ---
echo "[1/3] Création de l'environnement virtuel dans ./venv..."
python3 -m venv venv

# --- 2. Installation des dépendances ---
# On utilise le pip du venv pour installer les librairies du projet.
echo "[2/3] Installation des dépendances depuis requirements.txt..."
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt

# --- 3. Téléchargement d'un jeu de données initial ---
# C'est la touche finale : on télécharge un petit échantillon de données
# pour que l'utilisateur puisse lancer les notebooks immédiatement.
echo "[3/3] Téléchargement d'un petit jeu de données de démarrage..."
# On lance le downloader avec des paramètres modestes pour que ce soit rapide.
# Par exemple, 2 plans et un maximum de 20 spectres.
venv/bin/python src/tools/dr5_downloader.py --limit 2 --max-spectres 20

# --- Message final ---
echo "Environnement AstroSpectro prêt à l'emploi !"
echo "Ouvrez un notebook dans le dossier 'notebooks/' et sélectionnez le noyau 'venv' pour commencer."