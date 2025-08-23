from __future__ import annotations
import argparse, re, shutil
from pathlib import Path
import pandas as pd
from html import escape

FIG_DIR = Path("reports/figures")

# Known filename patterns -> destination builder
PATTERNS = [
    # benchmark (within): cm_within_bench_S01_<clf>.<ext>
    (re.compile(r"^cm_within_bench_(S\d{2})_([A-Za-z0-9_]+)\.(png|pdf|svg)$"),
     lambda m: Path(f"within/bench/{m.group(1)}/{m.group(2)}.{m.group(3)}"),
     "within_bench"),
    # benchmark (loso): cm_loso_bench_<clf>_S01.<ext>
    (re.compile(r"^cm_loso_bench_([A-Za-z0-9_]+)_(S\d{2})\.(png|pdf|svg)$"),
     lambda m: Path(f"loso/bench/{m.group(1)}/{m.group(2)}.{m.group(3)}"),
     "loso_bench"),
    # simple within: cm_within_S01_<clf>_<labelcol>.<ext>
    (re.compile(r"^cm_within_(S\d{2})_([A-Za-z0-9_]+)_([A-Za-z0-9_]+)\.(png|pdf|svg)$"),
     lambda m: Path(f"within/simple/{m.group(1)}/{m.group(2)}_{m.group(3)}.{m.group(4)}"),
     "within_simple"),
    # simple loso: cm_loso_<clf>_<labelcol>_S01.<ext>
    (re.compile(r"^cm_loso_([A-Za-z0-9_]+)_([A-Za-z0-9_]+)_(S\d{2})\.(png|pdf|svg)$"),
     lambda m: Path(f"loso/simple/{m.group(1)}/{m.group(2)}/{m.group(3)}.{m.group(4)}"),
     "loso_simple"),
]

def classify_and_dest(fname: str) -> tuple[str, Path]:
    for rx, builder, tag in PATTERNS:
        m = rx.match(fname)
        if m:
            return tag, builder(m)
    # unknown -> misc
    ext = fname.split(".")[-1]
    return "misc", Path(f"misc/{fname}")

def build_index_html(rows, out_html: Path):
    # Group by top-level (within/loso/misc)
    by_section = {}
    for r in rows:
        section = r["new_rel"].split("/")[0] if r["new_rel"] else "misc"
        by_section.setdefault(section, []).append(r)

    def img_tag(p):
        # Use only PNG for thumbnails; link to PDF/SVG if they exist
        if p.endswith(".png"):
            return f'<a href="{escape(p)}"><img src="{escape(p)}" style="max-width:260px; margin:6px; border:1px solid #ddd;"/></a>'
        return f'<a href="{escape(p)}">{escape(Path(p).name)}</a>'

    html = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Figures Index</title>",
        "<style>body{font-family:Segoe UI,Arial,sans-serif;padding:16px;} h2{margin-top:28px;} .blk{margin-bottom:12px;}</style>",
        "</head><body><h1>Figures Index</h1>"
    ]
    for section, items in sorted(by_section.items()):
        html.append(f"<h2>{escape(section.capitalize())}</h2>")
        # Further group within/loso
        # within: .../bench/<SUBJECT>/..., .../simple/<SUBJECT>/
        # loso:   .../bench/<CLF>/...,     .../simple/<CLF>/<LABELCOL>/
        groups = {}
        for r in items:
            parts = Path(r["new_rel"]).parts
            key = "/".join(parts[:3]) if len(parts) >= 3 else "/".join(parts)
            groups.setdefault(key, []).append(r)
        for gkey, glist in sorted(groups.items()):
            html.append(f"<div class='blk'><strong>{escape(gkey)}</strong><br/>")
            # show PNGs only in the gallery; link PDFs/SVGs are less visual
            imgs = [x for x in glist if x["new_rel"].endswith(".png")]
            if not imgs:
                # fallback: list links
                links = " | ".join(img_tag(x["new_rel"]) for x in glist)
                html.append(links)
            else:
                html.append("".join(img_tag(x["new_rel"]) for x in imgs))
            html.append("</div>")
    html.append("</body></html>")
    out_html.write_text("\n".join(html), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(FIG_DIR), help="Folder containing existing figures.")
    ap.add_argument("--apply", action="store_true", help="Actually move files. Default: dry-run.")
    ap.add_argument("--index", default="reports/figures_index.html", help="HTML index output")
    ap.add_argument("--manifest", default="reports/figures_manifest.csv", help="CSV manifest output")
    args = ap.parse_args()

    src = Path(args.source)
    files = [p for p in src.glob("*.*") if p.suffix.lower() in (".png",".pdf",".svg")]
    moves = []
    for f in files:
        tag, rel_dest = classify_and_dest(f.name)
        new_path = src / rel_dest
        moves.append({
            "old": str(f),
            "new": str(new_path),
            "new_rel": str(rel_dest).replace("\\","/"),
            "pattern": tag,
            "ext": f.suffix.lower().lstrip("."),
        })

    # Dry-run report
    df = pd.DataFrame(moves)
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.manifest, index=False)
    print(f"[INFO] Planned {len(df)} files. Manifest -> {args.manifest}")

    # Ensure folders & (optionally) move
    if args.apply:
        for r in moves:
            newp = Path(r["new"])
            newp.parent.mkdir(parents=True, exist_ok=True)
            if Path(r["old"]) == newp:
                continue
            try:
                shutil.move(r["old"], r["new"])
            except Exception as e:
                print(f"[WARN] Could not move {r['old']} -> {r['new']}: {e}")
        print("[OK] Files moved.")
    else:
        print("[DRY-RUN] No files moved. Use --apply to perform changes.")

    # Build HTML index (works for both dry-run and after move; if dry-run, links show old layout)
    build_index_html(moves, Path(args.index))
    print(f"[OK] HTML gallery -> {args.index}")

if __name__ == "__main__":
    main()
