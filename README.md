# 🎵 Projet IA 2025–2026 — Séparation de sources audio avec Demucs

> **Arnaud BARON · Guillaume DARNATIGUES · Simon DROUET · Théodore FISCHER · Mathis GROS**  
> ENSTA Paris — 3A — Projet d'Intelligence Artificielle

---

## 📋 Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Installation](#installation)
3. [Lancer l'application](#lancer-lapplication)
4. [Dataset MUSDB-HQ](#dataset-musdb-hq)
5. [Structure du code](#structure-du-code)
   - [`src/music_separation/`](#srcmusic_separation)
   - [`src/fine-tuning/`](#srcfine-tuning)
   - [`experiments/`](#experiments)
   - [`app.py`](#apppy)
6. [Les 4 modèles Demucs](#les-4-modèles-demucs)
7. [Sources audio alternatives](#sources-audio-alternatives)

---

## Vue d'ensemble

Ce projet explore la **séparation de sources audio** (music source separation) à l'aide du modèle [Demucs](https://github.com/facebookresearch/demucs) (Meta AI Research).

L'objectif est de séparer un morceau de musique en **4 stems** :
- 🎤 **Vocals** — voix chantée
- 🥁 **Drums** — batterie et percussions
- 🎸 **Bass** — basse
- 🎹 **Other** — tous les autres instruments

Le projet comprend :
- Une **application web** Streamlit pour la séparation interactive
- Un **package Python** modulaire (`src/music_separation`) avec séparation, évaluation et visualisation
- Des **notebooks d'analyse** pour comparer les 4 variantes de Demucs
- Des scripts de **fine-tuning** sur le dataset Bach10

---

## Installation

```bash
# Cloner le dépôt
git clone <url-du-repo>
cd Projet_IA_2526

# Installer les dépendances
pip install -r requirements.txt
```

**Dépendances principales :**

| Package | Version | Rôle |
|---------|---------|------|
| `demucs` | 4.0.1 | Modèle de séparation audio |
| `torch` | 2.10.0 | Backend deep learning |
| `librosa` | 0.11.0 | Analyse et traitement audio |
| `soundfile` | 0.13.1 | Lecture/écriture WAV/FLAC |
| `streamlit` | 1.43.0 | Interface web |
| `museval` | 0.4.1 | Métriques BSS (SDR, ISR, SIR, SAR) |
| `mir_eval` | 0.8.2 | Métriques alternatives |

---

## Lancer l'application

```bash
streamlit run app.py --server.fileWatcherType none
```

L'application permet de :
1. Uploader un fichier audio (MP3, WAV, FLAC, OGG, M4A, MP4)
2. Choisir le modèle Demucs
3. Lancer la séparation et écouter chaque stem
4. Télécharger les stems individuellement ou en `.zip`
5. Visualiser les spectrogrammes de chaque piste

---

## Dataset MUSDB-HQ

### Présentation

[MUSDB-HQ](https://sigsep.github.io/datasets/musdb.html) est le dataset de référence pour l'évaluation des systèmes de séparation de sources musicales. Il est utilisé dans la compétition [SiSEC](https://sisec.inria.fr/).

| Caractéristique | Valeur |
|----------------|--------|
| Nb de morceaux | 150 (100 train + 50 test) |
| Format | WAV stéréo, 44 100 Hz |
| Durée totale | ~10 heures |
| Stems disponibles | `vocals`, `drums`, `bass`, `other`, `mixture` |
| Licence | Recherche uniquement |

### Structure attendue

```
dataset/
└── musdb18hq/
    ├── train/
    │   ├── A Classic Education - NightOwl/
    │   │   ├── mixture.wav
    │   │   ├── vocals.wav
    │   │   ├── drums.wav
    │   │   ├── bass.wav
    │   │   └── other.wav
    │   └── ...
    └── test/
        ├── Arise - Run Run Run/
        │   ├── mixture.wav
        │   ├── vocals.wav
        │   ...
        └── ...
```

### Utilisation dans le projet

Le dataset MUSDB est utilisé pour :
- **Évaluer** les modèles Demucs avec les métriques officielles (SDR, ISR, SIR, SAR)
- **Activer** la section 10 du notebook de comparaison (`USE_MUSDB = True`)
- **Fine-tuner** les modèles (section `src/fine-tuning`)

> **Attention** : MUSDB-HQ n'est pas inclus dans le dépôt pour des raisons de droits. Il doit être téléchargé séparément sur [Zenodo](https://zenodo.org/record/3338373).

### Dataset Bach10

En complément de MUSDB, le projet utilise **Bach10** pour le fine-tuning sur un domaine différent (musique classique avec instruments à vent). Les données pré-traitées sont dans `Bach10_v1.1/`.

---

## Structure du code

```
Projet_IA_2526/
│
├── app.py                        # Application web Streamlit
├── requirements.txt              # Dépendances Python
│
├── src/
│   ├── music_separation/         # 📦 Package principal de séparation audio
│   │   ├── __init__.py
│   │   ├── config.py             # Constantes et paramètres globaux
│   │   ├── separate.py           # Classe AudioSeparator
│   │   ├── evaluate.py           # Classe AudioEvaluator (BSS metrics)
│   │   ├── visualization.py      # Classe Visualizer (spectrogrammes, waveforms)
│   │   ├── audio_utils.py        # Utilitaires bas-niveau (load, save, resample...)
│   │   ├── analysis_tools.py     # Métriques avancées & visualisations comparatives
│   │   ├── loader.py             # Classe AudioLoader (fichiers locaux, MUSDB)
│   │   ├── cache_manager.py      # Gestion des fichiers temporaires
│   │   ├── benchmark.py          # Benchmark sur un grand nombre de pistes
│   │   └── compare.py            # Comparaison multi-modèles sur une piste
│   │
│   └── fine-tuning/              # Scripts de fine-tuning sur Bach10
│       ├── augmente_data.py      # Augmentation de données (mixage de stems)
│       └── split_data.py         # Découpage train/val du dataset
│
├── experiments/                  # Notebooks d'analyse
│   ├── demucs_model_comparison.ipynb   # 🌟 Comparaison des 4 modèles (principal)
│   ├── test_museval.ipynb              # Tests métriques museval
│   ├── test_museval_analyse_spectrale.ipynb
│   └── spleeter.ipynb                  # Comparaison avec Spleeter
│
├── data/
│   ├── input/                    # Fichiers audio d'entrée
│   └── output_demucs*/           # Stems séparés + résultats
│
└── Bach10_v1.1/                  # Dataset Bach10 (musique classique)
```

---

## `src/music_separation/`

### `config.py`
Centralise toutes les constantes du projet :
- `DEFAULT_SAMPLE_RATE = 44100` Hz
- `DEFAULT_DEVICE` — auto-détecté (`cuda` si GPU disponible, sinon `cpu`)
- `SUPPORTED_MODELS = ["htdemucs", "htdemucs_ft", "mdx_extra", "mdx_extra_q"]`
- Chemins racine du projet (`ROOT_DIR`, `DATA_DIR`, etc.)

### `separate.py` — `AudioSeparator`
Classe principale pour la séparation audio. Gère le cycle complet :

```python
from src.music_separation import AudioSeparator

sep = AudioSeparator(model_name="htdemucs")

# Séparation vers des fichiers
saved_paths = sep.process_file("input.mp3", "output/")

# Traitement par batch
results = sep.process_batch(["track1.wav", "track2.wav"], "output/")

# Mode mémoire (pour Streamlit)
stems = sep.process_file_in_memory("input.mp3")
# → [(nom, bytes_audio, filename), ...]
```

### `evaluate.py` — `AudioEvaluator`
Calcule les métriques de qualité BSS en comparant les stems estimés à la vérité terrain :

```python
from src.music_separation import AudioEvaluator

ev = AudioEvaluator()
metrics = ev.compute_bss_metrics(
    reference_stems=["vocals_gt.wav", "drums_gt.wav", "bass_gt.wav", "other_gt.wav"],
    estimated_stems=["vocals_est.wav", "drums_est.wav", "bass_est.wav", "other_est.wav"]
)
# metrics = {"SDR": [...], "ISR": [...], "SIR": [...], "SAR": [...]}
```

| Métrique | Signification | Interprétation |
|----------|--------------|----------------|
| **SDR** | Signal-to-Distortion Ratio | Qualité globale (↑ meilleur) |
| **ISR** | Image-to-Spatial distortion Ratio | Précision spatiale |
| **SIR** | Source-to-Interferences Ratio | Isolation des sources |
| **SAR** | Sources-to-Artifacts Ratio | Absence d'artefacts |

Utilise `museval` si disponible (inclut ISR), sinon `mir_eval`.

### `visualization.py` — `Visualizer`
Génère des visualisations audio :

```python
from src.music_separation import Visualizer

viz = Visualizer()

# Spectrogramme depuis un fichier → PNG bytes
png = viz.spectrogram_from_path("vocals.wav", title="Vocals")

# Spectrogramme depuis des bytes (audio en mémoire)
png = viz.spectrogram_from_bytes(audio_bytes, title="Drums")

# Comparaison de spectrogrammes (multi-fichiers)
fig = viz.plot_spectrograms({"Mix": "mixture.wav", "Voix": "vocals.wav"})

# Forme d'onde
fig = viz.plot_waveform("bass.wav", title="Basse")
```

### `audio_utils.py`
Fonctions bas-niveau sans état :
- `load_audio(path, sr, mono, force_stereo)` — WAV/FLAC via soundfile, MP3/M4A via librosa
- `save_audio(path, audio, sr)` — sauvegarde en WAV
- `normalize_audio`, `to_mono`, `to_stereo`, `slice_audio`
- `mix_stems(stems)` — reconstruit le mixture depuis les stems
- `get_duration(path)` — durée en secondes

### `analysis_tools.py`
Métriques avancées et visualisations pour la comparaison des modèles (utilisé dans le notebook) :

**Métriques :**
- `compute_snr(ref, est)` — Signal-to-Noise Ratio (dB)
- `compute_si_snr(ref, est)` — Scale-Invariant SNR (invariant au gain)
- `compute_rms_energy(audio)` — Énergie RMS
- `compute_spectral_centroid(audio, sr)` — Brillance spectrale (Hz)
- `compute_spectral_rolloff(audio, sr)` — Fréquence de roll-off
- `compute_spectral_flux(audio, sr)` — Variabilité temporelle du spectre
- `compute_mfcc_distance(ref, est, sr)` — Similarité timbrale via MFCC
- `compute_frequency_band_energy(audio, sr)` — Énergie par bande (sub-bass, bass, mid, high)

**Visualisations (dark mode, palette cohérente) :**
- `plot_waveforms_comparison(...)` — formes d'onde côte-à-côte
- `plot_spectrograms_grid(...)` — grille de spectrogrammes par modèle
- `plot_metrics_bar_per_stem(...)` — barres groupées (SDR/SIR/SAR par stem)
- `plot_inference_time_comparison(...)` — temps + Real-Time Factor
- `plot_si_snr_comparison(...)` — SI-SNR par stem
- `plot_mfcc_distance_matrix(...)` — heatmap de similarité timbrale
- `plot_frequency_bands_heatmap(...)` — énergie par bande
- `plot_spectral_features_comparison(...)` — centroïde, rolloff, flux
- `plot_metrics_radar(...)` — spider chart normalisé
- `plot_sdr_evolution_summary(...)` — vue synthèse SDR

### `benchmark.py` et `compare.py`
Fonctions de haut niveau pour l'évaluation à grande échelle :

```python
from src.music_separation import run_benchmark, compare_models_on_track

# Benchmark d'un modèle sur plusieurs pistes
results = run_benchmark("htdemucs", tracks=[...], gt_stems_dir=Path("musdb18hq/test"))

# Comparaison de plusieurs modèles sur une piste
comparison = compare_models_on_track(
    track_path=Path("musdb18hq/test/Arise - Run Run Run/mixture.wav"),
    gt_stems_dir=Path("musdb18hq/test"),
    models=["htdemucs", "htdemucs_ft", "mdx_extra", "mdx_extra_q"]
)
```

---

## `src/fine-tuning/`

Scripts pour adapter Demucs à un nouveau domaine (Bach10 — instruments classiques).

| Script | Rôle |
|--------|------|
| `augmente_data.py` | Génère des mixtures synthétiques à partir des stems isolés (data augmentation) |
| `split_data.py` | Sépare les données en ensembles train/validation |

Le fine-tuning est lancé via le script SLURM (`submit_fine-tune.slurm`) sur un cluster GPU.

---

## `experiments/`

| Notebook | Description |
|----------|-------------|
| `demucs_model_comparison.ipynb` | **Comparaison complète** des 4 modèles : temps, formes d'onde, spectrogrammes, MFCC, SI-SNR, radar, classement final |
| `test_museval.ipynb` | Tests des métriques BSS avec `museval` |
| `test_museval_analyse_spectrale.ipynb` | Analyse spectrale + métriques sur des morceaux MUSDB |
| `spleeter.ipynb` | Comparaison avec le modèle Spleeter (Deezer) |

Les résultats (PNG + CSV) sont sauvegardés dans `data/output_demucs_comparison/results/`.

---

## Les 4 modèles Demucs

| Modèle | Architecture | Caractéristiques |
|--------|-------------|-----------------|
| `htdemucs` | Hybrid Transformer Demucs | Modèle de référence, bon équilibre qualité/vitesse |
| `htdemucs_ft` | HTDemucs fine-tuné | Fine-tuné sur MUSDB-HQ, meilleure qualité vocale |
| `mdx_extra` | MDX-Net convolutif | Gagnant de la compétition MDX, très bonne qualité |
| `mdx_extra_q` | MDX-Net quantifié | Version allégée, consomme moins de mémoire GPU |

---

## Sources audio alternatives

### YouTube Studio Audio Library
Pour tester la séparation sans MUSDB :

🔗 https://www.youtube.com/audiolibrary

- Musiques libres de droits, téléchargeables en MP3
- **Limite** : fichiers déjà mixés, pas de stems disponibles → pas d'évaluation avec métriques BSS

### Fichiers de démonstration
Des exemples sont disponibles dans `data/input/` :
- `good for the ghost - Alge.mp3`
- `Hello - ssymon edit.mp3`
