#!/usr/bin/env python3

import shutil
import numpy as np
import soundfile as sf
from pathlib import Path

SRC = Path.home() / "Projet_IA_2526/Bach10_Sorted"
DST = Path.home() / "Projet_IA_2526/Bach10_Clean_Augmented"
STEMS = ["bassoon", "clarinet", "saxophone", "violin"]

def get_reference_file(src_dir: Path) -> Path:
    for stem in STEMS:
        f = src_dir / f"{stem}.wav"
        if f.exists():
            return f
    raise FileNotFoundError(f"Aucun stem wav dans {src_dir}")

def create_silence(reference_file: Path, output_file: Path):
    audio, sr = sf.read(reference_file)
    silence = np.zeros_like(audio)
    sf.write(output_file, silence, sr)

def create_mixture(dst_dir: Path):
    """Crée mixture.wav en sommant les 4 stems."""
    mixture = None
    sr = None
    for stem in STEMS:
        audio, sr = sf.read(dst_dir / f"{stem}.wav")
        mixture = audio if mixture is None else mixture + audio
    mixture = mixture / (np.abs(mixture).max() + 1e-8) * 0.9
    sf.write(dst_dir / "mixture.wav", mixture, sr)

def copier_morceau(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    reference = get_reference_file(src_dir)
    for stem in STEMS:
        src_file = src_dir / f"{stem}.wav"
        dst_file = dst_dir / f"{stem}.wav"
        if src_file.exists():
            shutil.copy2(src_file, dst_file)
        else:
            create_silence(reference, dst_file)
            print(f"  [SILENCE] {dst_dir.name}/{stem}.wav")
    create_mixture(dst_dir)

# Collecte tous les morceaux
tous_les_morceaux = []
for split_dir in sorted(SRC.iterdir()):
    if split_dir.is_dir():
        for morceau in sorted(split_dir.iterdir()):
            if morceau.is_dir():
                tous_les_morceaux.append(morceau)

train_morceaux   = [m for m in tous_les_morceaux if not m.name.startswith("10-")]
valtest_morceaux = [m for m in tous_les_morceaux if m.name.startswith("10-")]


for morceau in train_morceaux:
    print(f"[train] {morceau.name}")
    copier_morceau(morceau, DST / "train" / morceau.name)

for morceau in valtest_morceaux:
    print(f"[valid] {morceau.name}")
    copier_morceau(morceau, DST / "valid" / morceau.name)
    print(f"[test]  {morceau.name}")
    copier_morceau(morceau, DST / "test" / morceau.name)

print("\nRésumé :")
for split in ["train", "valid", "test"]:
    split_dir = DST / split
    if not split_dir.exists():
        continue
    dossiers   = [d for d in split_dir.iterdir() if d.is_dir()]
    complets   = [d for d in dossiers if len(list(d.glob("*.wav"))) == 5]  # 4 stems + mixture
    incomplets = [d for d in dossiers if len(list(d.glob("*.wav"))) != 5]
    print(f"  {split} : {len(complets)} complets, {len(incomplets)} incomplets")

print("\nDataset prêt.")