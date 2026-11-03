"""
export_doc_facts.py — Exporte les « faits » d'AstroSpectro (résultats, métadonnées,
rapports) vers research-docs/_facts/astrospectro/, pour que les sessions Claude
cloud puissent documenter à partir des vrais chiffres sans accès à data/ (gitignoré).

Usage (depuis la racine d'AstroSpectro, venv activé) :
    python export_doc_facts.py                       # cible par défaut : ../research-docs
    python export_doc_facts.py --target D:/x/research-docs --no-png --dry-run

Ce qui est copié : petits fichiers texte (json, txt, md, yaml, csv légers) et figures PNG
sous les plafonds de taille. Ce qui n'est JAMAIS copié : .pkl, .joblib, .fits(.gz), .env,
gros CSV (résumés à la place : en-tête + nombre de lignes). Tout ce qui est trouvé,
copié ou non, est listé dans MANIFEST.md.

ATTENTION : research-docs est public. Ne pointe pas ce script vers des dossiers sensibles.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

# Racines scannées (relatives à la racine d'AstroSpectro)
ROOTS = ["data/models", "data/reports", "data/catalog", "data/processed", "logs/shap", "logs/validation"]

# Plafonds par extension (octets) ; extension absente = jamais copiée
MAX_SIZE = {
    ".json": 2_000_000, ".txt": 2_000_000, ".md": 2_000_000,
    ".yaml": 500_000, ".yml": 500_000, ".csv": 1_000_000,
    ".png": 1_500_000, ".svg": 1_500_000,
}
NEVER = {".pkl", ".joblib", ".fits", ".gz", ".npy", ".npz", ".pt", ".pth", ".h5", ".env"}
SKIP_NAME_PARTS = ("secret", "token", "apikey", "api_key", ".env")
SUMMARIZE_CSV_OVER = 1_000_000          # gros CSV → résumé
SUMMARIZE_LATEST_FEATURES = 3           # data/processed : ne résumer que les N plus récents
MAX_TOTAL_MB = 150                      # garde-fou sur la taille totale exportée


def count_lines(path: Path) -> int:
    n = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            n += chunk.count(b"\n")
    return n


def csv_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n\r").split(",")
    return {
        "file": path.name,
        "size_mb": round(path.stat().st_size / 1e6, 2),
        "n_rows": max(count_lines(path) - 1, 0),
        "n_columns": len(header),
        "columns": header,
        "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=".", help="racine d'AstroSpectro")
    ap.add_argument("--target", default="../research-docs", help="racine de research-docs")
    ap.add_argument("--no-png", action="store_true", help="ne pas copier les figures")
    ap.add_argument("--include-archive", action="store_true", help="inclure les dossiers archive/")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = Path(args.source).resolve()
    out = Path(args.target).resolve() / "_facts" / "astrospectro"
    if not (src / "src" / "pipeline").exists():
        raise SystemExit(f"{src} ne ressemble pas à la racine d'AstroSpectro (src/pipeline absent).")

    copied, skipped, summaries = [], [], []
    total = 0

    # Fichiers features_*.csv : seuls les plus récents sont résumés
    processed = sorted((src / "data/processed").glob("features_*.csv"), key=lambda p: p.stat().st_mtime)
    latest_features = set(processed[-SUMMARIZE_LATEST_FEATURES:])

    for root in ROOTS:
        base = src / root
        if not base.exists():
            skipped.append((root, "racine absente"))
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(src).as_posix()
            ext = p.suffix.lower()
            size = p.stat().st_size
            name = p.name.lower()

            if not args.include_archive and "archive" in p.relative_to(src).parts:
                skipped.append((rel, "archive"))
                continue
            if ext in NEVER or any(s in name for s in SKIP_NAME_PARTS):
                skipped.append((rel, f"exclu ({ext or name})"))
                continue
            if ext == ".csv" and size > SUMMARIZE_CSV_OVER:
                if root == "data/processed" and p not in latest_features:
                    skipped.append((rel, f"gros CSV ancien, {size/1e6:.1f} Mo"))
                    continue
                summaries.append(csv_summary(p))
                skipped.append((rel, f"résumé seulement, {size/1e6:.1f} Mo"))
                continue
            if ext in (".png", ".svg") and args.no_png:
                skipped.append((rel, "--no-png"))
                continue
            limit = MAX_SIZE.get(ext)
            if limit is None or size > limit:
                skipped.append((rel, f"type/taille ({size/1e6:.2f} Mo)"))
                continue

            copied.append((rel, size))
            total += size
            if not args.dry_run:
                dest = out / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest)

    if total / 1e6 > MAX_TOTAL_MB:
        raise SystemExit(f"Export de {total/1e6:.0f} Mo > {MAX_TOTAL_MB} Mo : relance avec --no-png "
                         "ou baisse les plafonds. (Fichiers déjà copiés à supprimer à la main si besoin.)")

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        "# Faits AstroSpectro — export pour la documentation",
        "",
        f"- Exporté le : {stamp}",
        f"- Source : `{src.name}/` (data/ et logs/ sont gitignorés dans le dépôt de code)",
        f"- Fichiers copiés : {len(copied)} ({total/1e6:.1f} Mo)",
        "- **Règle** : tout chiffre publié dans la doc doit venir d'un fichier de ce dossier.",
        "",
        "## Copiés",
        "",
        *[f"- `{r}` ({s/1e3:.0f} Ko)" for r, s in copied],
        "",
        "## Présents localement mais non copiés",
        "",
        *[f"- `{r}` — {why}" for r, why in skipped],
    ]
    if args.dry_run:
        print("\n".join(lines[:8]))
        print(f"[dry-run] {len(copied)} fichiers à copier, {len(summaries)} CSV résumés, {len(skipped)} ignorés.")
        return

    out.mkdir(parents=True, exist_ok=True)
    (out / "MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "csv_summaries.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"OK → {out}\n{len(copied)} fichiers ({total/1e6:.1f} Mo), {len(summaries)} CSV résumés, "
          f"{len(skipped)} ignorés. Vérifie MANIFEST.md avant de committer.")


if __name__ == "__main__":
    main()
