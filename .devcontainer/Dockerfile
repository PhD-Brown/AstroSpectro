FROM jupyter/scipy-notebook:latest
USER root
WORKDIR /workspaces/AstroSpectro
COPY . .

# On crée juste le venv ici. L'installation se fera dans post-create.
RUN python3 -m venv venv
RUN chown -R ${NB_UID}:${NB_GID} /workspaces/AstroSpectro

USER ${NB_USER}
