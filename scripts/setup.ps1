# scripts/setup.ps1
$ErrorActionPreference = "Stop"

# -- Réglages --
$Py = "py -3.11"                # Python 3.11 via Python Launcher
$VenvDir = "venv"               # on standardise sur 'venv'
$VenvPy  = ".\$VenvDir\Scripts\python.exe"

# -- Crée le venv si absent --
if (-not (Test-Path $VenvDir)) {
  Write-Host "[1/5] Création du venv ($VenvDir)..."
  cmd /c $Py + " -m venv $VenvDir"
}

# -- Pip & outils de build --
Write-Host "[2/5] Upgrade pip/setuptools/wheel..."
& $VenvPy -m pip install --upgrade pip setuptools wheel

# -- Installe le projet en editable + dépendances --
Write-Host "[3/5] Installation du projet (editable) + deps..."
& $VenvPy -m pip install -e .
if (Test-Path "requirements.txt") {
  & $VenvPy -m pip install -r requirements.txt
}

# -- Kernel Jupyter --
Write-Host "[4/5] Kernel Jupyter..."
& $VenvPy -m pip install ipykernel ipywidgets
& $VenvPy -m ipykernel install --user --name astrospectro --display-name "AstroSpectro (Py3.11)"

# -- Settings VS Code utiles --
Write-Host "[5/5] Génération .vscode/settings.json..."
New-Item -ItemType Directory -Path ".vscode" -Force | Out-Null
@"
{
  "python.defaultInterpreterPath": "${workspaceFolder}\\venv\\Scripts\\python.exe",
  "jupyter.notebookFileRoot": "${workspaceFolder}",
  "python.analysis.extraPaths": ["${workspaceFolder}", "${workspaceFolder}\\src"]
}
"@ | Set-Content -Encoding UTF8 .vscode\settings.json

Write-Host "`n  Environnement prêt."
Write-Host "   Interpréteur VS Code : .\\venv\\Scripts\\python.exe"
Write-Host "   Kernel Jupyter       : AstroSpectro (Py3.11)"
