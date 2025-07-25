# On part d'une image officielle qui contient déjà Python ET JupyterLab bien configuré.
FROM jupyter/scipy-notebook:latest

# On passe en utilisateur root pour pouvoir installer des paquets système si besoin
USER root

# On crée un dossier de travail et on s'y place
WORKDIR /workspaces/AstroSpectro

# On copie notre code (le dossier courant '.') dans le conteneur
COPY . .

# On crée un environnement virtuel et on installe nos dépendances
# On le fait pendant la construction de l'image, c'est plus efficace.
RUN python3 -m venv venv && \
    ./venv/bin/pip install --upgrade pip && \
    ./venv/bin/pip install -r requirements.txt

# On redonne la propriété du dossier à l'utilisateur par défaut de l'image Jupyter
RUN chown -R ${NB_UID}:${NB_GID} /workspaces/AstroSpectro

# On repasse à l'utilisateur non-root pour la sécurité
USER ${NB_USER}