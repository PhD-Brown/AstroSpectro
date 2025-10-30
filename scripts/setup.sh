#!/usr/bin/env bash
# scripts/setup.sh
set -euo pipefail

PY="${PYTHON:-python3}"
VENV_DIR="venv"
VENV_PY="./$VENV_DIR/bin/python"

if [ ! -d "$VENV_DIR" ]; then
  echo "[1/5] Creating venv ($VENV_DIR)..."
  "$PY" -m venv "$VENV_DIR"
fi

echo "[2/5] Upgrading pip/setuptools/wheel..."
"$VENV_PY" -m pip install --upgrade pip setuptools wheel

echo "[3/5] Installing project (editable) + deps..."
"$VENV_PY" -m pip install -e .
[ -f requirements.txt ] && "$VENV_PY" -m pip install -r requirements.txt

echo "[4/5] Jupyter kernel..."
"$VENV_PY" -m pip install ipykernel ipywidgets
"$VENV_PY" -m ipykernel install --user --name astrospectro --display-name "AstroSpectro (Py3.11)"

echo "[5/5] VS Code settings..."
mkdir -p .vscode
cat > .vscode/settings.json <<'EOF'
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "jupyter.notebookFileRoot": "${workspaceFolder}",
  "python.analysis.extraPaths": ["${workspaceFolder}", "${workspaceFolder}/src"]
}
EOF

echo ""
echo " Environnement prêt."
echo "   Interpreter : ./venv/bin/python"
echo "   Kernel      : AstroSpectro (Py3.11)"
