# scripts/clean.ps1
$ErrorActionPreference = "Stop"

Write-Host "Removing Jupyter kernel 'astrospectro' (if present)..."
try {
  $json = jupyter kernelspec list --json | ConvertFrom-Json
  # Selon versions: 'kernelspecs' ou 'kernels' – on gère les deux
  $ks = $json.kernelspecs
  if (-not $ks) { $ks = $json.kernels }
  if ($ks.ContainsKey("astrospectro")) {
    jupyter kernelspec remove -f astrospectro -y *> $null 2>&1
    Write-Host "Kernel removed."
  } else {
    Write-Host "(kernel not found, nothing to remove)"
  }
} catch {
  Write-Host "(couldn't query kernelspecs, skipping)"
}

Write-Host "Removing virtual environments..."
if (Get-Command deactivate -ErrorAction SilentlyContinue) { deactivate }  # au cas où
if (Test-Path ".\venv")  { Remove-Item -Recurse -Force .\venv }
if (Test-Path ".\.venv") { Remove-Item -Recurse -Force .\.venv }
Write-Host "Done."
