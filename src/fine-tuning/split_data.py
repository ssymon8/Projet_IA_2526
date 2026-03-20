#!/usr/bin/env python3

import shutil
from pathlib import Path

SRC = Path("./Bach10_v1.1")
DST = Path("./Bach10_Clean")
STEMS = {
    "bassoon":   "bassoon",
    "clarinet":  "clarinet",
    "saxophone": "saxphone",  
    "violin":    "violin",
}
VAL_COUNT = 2

morceaux = sorted([d for d in SRC.iterdir() if d.is_dir()])
total = len(morceaux)

train_morceaux = morceaux[:total - VAL_COUNT]
val_morceaux   = morceaux[total - VAL_COUNT:]

def copier_morceau(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for stem_dst, stem_src in STEMS.items():
        src_file = src_dir / f"{src_dir.name}-{stem_src}.wav"
        dst_file = dst_dir / f"{stem_dst}.wav"
        if src_file.exists():
            shutil.copy2(src_file, dst_file)
        else:
            print(f"  [MANQUANT] {src_file}")

print(f"Morceaux : {total} total → {len(train_morceaux)} train / {len(val_morceaux)} val\n")

for morceau in train_morceaux:
    print(f"[train] {morceau.name}")
    copier_morceau(morceau, DST / "train" / morceau.name)

for morceau in val_morceaux:
    print(f"[val]   {morceau.name}")
    copier_morceau(morceau, DST / "val" / morceau.name)

print("\nRésumé :")
for split in ["train", "val"]:
    dossiers = list((DST / split).iterdir())
    complets   = [d for d in dossiers if len(list(d.glob("*.wav"))) == 4]
    incomplets = [d for d in dossiers if len(list(d.glob("*.wav"))) != 4]
    print(f"  {split} : {len(complets)} complets, {len(incomplets)} incomplets")
    for d in incomplets:
        manquants = [s + ".wav" for s in STEMS if not (d / f"{s}.wav").exists()]
        print(f"    {d.name} — manque : {', '.join(manquants)}")

print("\nDataset prêt.")