FROM jupyter/scipy-notebook:latest
USER root
WORKDIR /workspaces/AstroSpectro
COPY . .

RUN python3 -m venv venv
RUN chown -R ${NB_UID}:${NB_GID} /workspaces/AstroSpectro

USER ${NB_USER}
