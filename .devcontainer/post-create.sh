#!/bin/bash
# .devcontainer/post-create.sh

echo "--- Lancement du script post-création pour AstroSpectro ---"

# Créer le venv (déjà fait par le Dockerfile, mais on s'assure)
python3 -m venv venv

# Installer le projet LUI-MÊME en mode éditable
# Le '.' signifie 'le projet dans le dossier courant'
# Le '-e' signifie 'éditable'
echo "[1/3] Installation du projet AstroSpectro en mode éditable..."
venv/bin/pip install -e .

# Installer les dépendances (pip le fait déjà avec la commande ci-dessus,
# mais on peut le garder pour être sûr)
echo "[2/3] Vérification des dépendances..."
venv/bin/pip install -r requirements.txt

# Enregistrer le noyau
echo "[3/3] Enregistrement du noyau Jupyter..."
venv/bin/pip install ipykernel
venv/bin/python -m ipykernel install --user --name=AstroSpectro-venv --display-name "Python (AstroSpectro venv)"

echo "Environnement AstroSpectro prêt."