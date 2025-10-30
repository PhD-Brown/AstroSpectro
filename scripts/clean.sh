#!/usr/bin/env bash
# scripts/clean.sh
set -euo pipefail
jupyter kernelspec remove -f astrospectro || true
rm -rf venv .venv
echo "Done."
