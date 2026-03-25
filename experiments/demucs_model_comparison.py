#!/usr/bin/env python3
"""
🎵 Comparaison des 4 modèles Demucs
====================================
Analyse comparative — Séparation de sources audio avec Demucs sur MUSDB18-HQ

| Modèle        | Description                              |
|---------------|------------------------------------------|
| htdemucs      | Hybrid Transformer Demucs (baseline)     |
| htdemucs_ft   | HTDemucs fine-tuné sur MUSDB-HQ          |
| mdx_extra     | MDX-Net (architecture convolutive)       |
| mdx_extra_q   | MDX-Net quantifié (plus léger)           |

Projet IA — ENSTA Paris 2025/2026
"""

# ─────────────────────────────────────────────────────────────────────────────
# ⚙️  1. Configuration & Imports
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import torch

# ─── Dark mode global ────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117', 'axes.facecolor': '#0d1117',
    'text.color': 'white', 'axes.labelcolor': 'white',
    'xtick.color': 'white', 'ytick.color': 'white',
    'axes.edgecolor': '#333333', 'grid.color': '#333333',
    'legend.facecolor': '#1a1a2e', 'legend.edgecolor': '#444444',
    'font.family': 'DejaVu Sans', 'font.size': 11,
})

# ─── Imports projet ──────────────────────────────────────────────────────────
from src.music_separation import AudioSeparator, AudioEvaluator, audio_utils, analysis_tools
from src.music_separation.config import SUPPORTED_MODELS, DEFAULT_SAMPLE_RATE
from src.music_separation.analysis_tools import (
    Timer, MODEL_COLORS, MODEL_LABELS, STEM_NAMES, STEM_COLORS,
    compute_snr, compute_si_snr, compute_rms_energy,
    compute_spectral_centroid, compute_spectral_rolloff, compute_spectral_flux,
    compute_mfcc_distance, compute_frequency_band_energy,
    plot_metrics_bar_per_stem, plot_waveforms_comparison, plot_spectrograms_grid,
    plot_frequency_bands_heatmap, plot_inference_time_comparison,
    plot_si_snr_comparison, plot_mfcc_distance_matrix,
    plot_spectral_features_comparison, plot_sdr_evolution_summary, plot_metrics_radar
)

print(f"✅ Imports OK — device : {'cuda' if torch.cuda.is_available() else 'cpu'}")
print(f"📂 Racine du projet : {PROJECT_ROOT}")


# ─────────────────────────────────────────────────────────────────────────────
# 📂 2. Sélection du dataset & Helper
# ─────────────────────────────────────────────────────────────────────────────

USE_MUSDB = True 

# ── Sélection des morceaux valides ─────────────────────────────────────────────
MUSDB_TEST_DIR = Path(PROJECT_ROOT) / 'dataset' / 'musdb18hq' / 'test'
all_track_dirs = sorted([d for d in MUSDB_TEST_DIR.iterdir() if d.is_dir()])

TRACK_DIRS = []
MAX_TRACKS = 3

for td in all_track_dirs:
    mix_file = td / 'mixture.wav'
    # On vérifie que le mix existe, n'est pas vide, ET que les stems GT sont là
    if mix_file.exists() and mix_file.stat().st_size > 0:
        if all((td / f"{s}.wav").exists() and (td / f"{s}.wav").stat().st_size > 0 for s in STEM_NAMES):
            TRACK_DIRS.append(td)
    if len(TRACK_DIRS) >= MAX_TRACKS:
        break

if not TRACK_DIRS:
    print(f"❌ Aucun morceau valide (mixture.wav + stems) trouvé dans {MUSDB_TEST_DIR}")
    sys.exit(1)

OUTPUT_DIR  = Path(PROJECT_ROOT) / 'data' / 'output_demucs_comparison'
RESULTS_DIR = OUTPUT_DIR / 'results'
PLOTS_DIR   = RESULTS_DIR / 'plots'
for d in [OUTPUT_DIR, RESULTS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def save_fig(fig: plt.Figure, name: str, dpi: int = 150) -> None:
    out = PLOTS_DIR / f"{name}.png"
    fig.savefig(out, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  💾 Sauvegardé → {out.relative_to(PROJECT_ROOT)}")

MODELS = SUPPORTED_MODELS
total_duration = 0
for td in TRACK_DIRS:
    total_duration += audio_utils.get_duration(td / 'mixture.wav')

print(f"🎵 Évaluation sur {len(TRACK_DIRS)} morceaux")


# ─────────────────────────────────────────────────────────────────────────────
# ⏱️  3. Boucle de Traitement (Modèle > Piste)
# ─────────────────────────────────────────────────────────────────────────────

all_inference_times = {m: [] for m in MODELS}
all_load_times      = {m: [] for m in MODELS}
all_si_snr          = {m: {s: [] for s in STEM_NAMES} for m in MODELS}
all_mfcc            = {m: {s: [] for s in STEM_NAMES} for m in MODELS}
all_rms             = {m: {s: [] for s in STEM_NAMES} for m in MODELS}
all_spectral        = {m: {s: {'centroid': [], 'flux': []} for s in STEM_NAMES} for m in MODELS}
all_band_energies   = {m: {s: {} for s in STEM_NAMES} for m in MODELS}
all_bss             = {m: {met: {s: [] for s in STEM_NAMES} for met in ['SDR', 'SIR', 'SAR', 'ISR']} for m in MODELS}

example_stem_audio_arrays = {}

for model_name in MODELS:
    print(f"\n{'='*60}\n🤖 MODÈLE : {MODEL_LABELS[model_name]}\n{'='*60}")
    
    with Timer() as t_load:
        separator = AudioSeparator(model_name=model_name)
    all_load_times[model_name].append(t_load.elapsed)
    print(f"⏳ Modèle chargé en {t_load.elapsed:.2f}s")

    for t_idx, track_dir in enumerate(TRACK_DIRS):
        track_name = track_dir.name
        audio_file = track_dir / 'mixture.wav'
        print(f"\n  [Track {t_idx+1}/{len(TRACK_DIRS)}] {track_name}")

        gt_stems = {s: track_dir / f"{s}.wav" for s in STEM_NAMES}
        model_out_dir = OUTPUT_DIR / track_name / model_name
        model_out_dir.mkdir(parents=True, exist_ok=True)

        with Timer() as t_inf:
            saved_paths = separator.process_file(audio_file, model_out_dir)
        all_inference_times[model_name].append(t_inf.elapsed)
        
        path_map = {p.stem.split('_')[-1]: p for p in saved_paths}
        current_track_stems = {}

        for stem_name in STEM_NAMES:
            path = path_map.get(stem_name)
            if path and path.exists():
                audio, _ = audio_utils.load_audio(path, sr=DEFAULT_SAMPLE_RATE, mono=False)
                current_track_stems[stem_name] = audio
                
                # Métriques spectrales
                all_rms[model_name][stem_name].append(compute_rms_energy(audio))
                all_spectral[model_name][stem_name]['centroid'].append(compute_spectral_centroid(audio, sr=DEFAULT_SAMPLE_RATE))
                all_spectral[model_name][stem_name]['flux'].append(compute_spectral_flux(audio, sr=DEFAULT_SAMPLE_RATE))
                
                # Énergie par bandes
                be = compute_frequency_band_energy(audio, sr=DEFAULT_SAMPLE_RATE)
                for band, val in be.items():
                    if band not in all_band_energies[model_name][stem_name]:
                        all_band_energies[model_name][stem_name][band] = []
                    all_band_energies[model_name][stem_name][band].append(val)

        # SI-SNR & MFCC vs GT
        for stem_name in STEM_NAMES:
            ref_path = gt_stems.get(stem_name)
            est_audio = current_track_stems.get(stem_name)
            if ref_path and ref_path.exists() and est_audio is not None:
                ref_audio, _ = audio_utils.load_audio(ref_path, sr=DEFAULT_SAMPLE_RATE, mono=False)
                min_len = min(ref_audio.shape[-1], est_audio.shape[-1])
                ref_audio = ref_audio[..., :min_len]
                est_audio = est_audio[..., :min_len]
                all_si_snr[model_name][stem_name].append(compute_si_snr(ref_audio, est_audio))
                all_mfcc[model_name][stem_name].append(compute_mfcc_distance(ref_audio, est_audio, sr=DEFAULT_SAMPLE_RATE))

        # BSS Metrics
        evaluator = AudioEvaluator(sample_rate=DEFAULT_SAMPLE_RATE)
        gt_paths_list = [gt_stems[s] for s in STEM_NAMES if gt_stems[s].exists()]
        pred_paths_list = [path_map[s] for s in STEM_NAMES if s in path_map]
        try:
            metrics = evaluator.compute_bss_metrics(gt_paths_list, pred_paths_list)
            for met in ['SDR', 'SIR', 'SAR', 'ISR']:
                if met in metrics:
                    for i, s in enumerate(STEM_NAMES):
                        if i < len(metrics[met]):
                            all_bss[model_name][met][s].append(float(metrics[met][i]))
        except Exception as e:
            pass

        if t_idx == 0:
            example_stem_audio_arrays[model_name] = current_track_stems

print("\n✅ Évaluation terminée !")


# ─────────────────────────────────────────────────────────────────────────────
# 📐 4. Agrégation & Moyennes
# ─────────────────────────────────────────────────────────────────────────────

inference_times = {m: np.mean(all_inference_times[m]) for m in MODELS}
load_times      = {m: np.mean(all_load_times[m]) for m in MODELS}
duration        = total_duration / len(TRACK_DIRS)
stem_audio_arrays = example_stem_audio_arrays

si_snr_per_model = {m: {s: np.mean(all_si_snr[m][s]) for s in STEM_NAMES if all_si_snr[m][s]} for m in MODELS}
mfcc_distances   = {m: {s: np.mean(all_mfcc[m][s]) for s in STEM_NAMES if all_mfcc[m][s]} for m in MODELS}
rms_per_model    = {m: {s: np.mean(all_rms[m][s]) for s in STEM_NAMES if all_rms[m][s]} for m in MODELS}
spectral_features = {m: {s: {
    'centroid': np.mean(all_spectral[m][s]['centroid']),
    'flux': np.mean(all_spectral[m][s]['flux']),
    'rolloff': 0,
} for s in STEM_NAMES if all_spectral[m][s]['centroid']} for m in MODELS}

band_energies = {m: {s: {b: np.mean(v) for b, v in all_band_energies[m][s].items()} 
                      for s in STEM_NAMES if all_band_energies[m][s]} for m in MODELS}

bss_results = {m: {met: [np.mean(all_bss[m][met][s]) for s in STEM_NAMES] for met in ['SDR', 'SIR', 'SAR', 'ISR']} for m in MODELS}
HAS_BSS = True


# ─────────────────────────────────────────────────────────────────────────────
# 📈 5. Visualisations (Averages)
# ─────────────────────────────────────────────────────────────────────────────

# --- Inférence ---
fig = plot_inference_time_comparison(inference_times, duration)
save_fig(fig, '01_inference_times')

df_times = pd.DataFrame({
    'Modèle': [MODEL_LABELS[m] for m in MODELS],
    'Chargement (s)': [round(load_times[m], 2) for m in MODELS],
    'Inférence (s)': [round(inference_times[m], 2) for m in MODELS],
    'RTF': [round(inference_times[m] / duration, 3) for m in MODELS],
}).set_index('Modèle')
print("\n### 📋 Tableau des temps (Moyennes)")
print(df_times.to_string())

# --- Formes d'onde & Spectrogrammes (Exemple index 0) ---
for stem_name in STEM_NAMES:
    fig = plot_waveforms_comparison(stem_audio_arrays, stem_name=stem_name, sr=DEFAULT_SAMPLE_RATE, max_seconds=15.0)
    save_fig(fig, f'02_waveform_{stem_name}')
    fig = plot_spectrograms_grid(stem_audio_arrays, stem_name=stem_name, sr=DEFAULT_SAMPLE_RATE)
    save_fig(fig, f'03_spectrogram_{stem_name}')

# --- RMS ---
df_rms = pd.DataFrame(rms_per_model).T.round(5).rename(index=MODEL_LABELS)
print("\n### 📋 Énergie RMS par stem")
print(df_rms.to_string())

# --- SI-SNR ---
fig = plot_si_snr_comparison(si_snr_per_model, stem_names=STEM_NAMES)
save_fig(fig, '08_si_snr')

# --- MFCC ---
df_mfcc = pd.DataFrame(mfcc_distances).T.round(2).rename(index=MODEL_LABELS)
print("\n### 📋 Distance MFCC (vs GT)")
print(df_mfcc.to_string())

# --- BSS ---
for metric_name in ['SDR', 'SIR', 'SAR', 'ISR']:
    fig = plot_metrics_bar_per_stem(bss_results, metric_name=metric_name, stem_names=STEM_NAMES)
    save_fig(fig, f'09_bss_{metric_name.lower()}')


# ─────────────────────────────────────────────────────────────────────────────
# 🕸️ 6. Radar & Classement
# ─────────────────────────────────────────────────────────────────────────────

def norm(values: dict, higher_is_better: bool = True) -> dict:
    vals = np.array(list(values.values()), dtype=float)
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    if vmax == vmin: return {m: 0.5 for m in values}
    normed = (vals - vmin) / (vmax - vmin)
    return {m: float(normed[i] if higher_is_better else 1-normed[i]) for i, m in enumerate(values)}

sisnr_global = {m: np.nanmean(list(si_snr_per_model[m].values())) for m in MODELS}
flux_global = {m: np.nanmean([spectral_features[m][s]['flux'] for s in STEM_NAMES if s in spectral_features[m]]) for m in MODELS}
bass_energy = {m: band_energies[m].get('bass', {}).get('bass (80-250 Hz)', 0.0) for m in MODELS}
vocals_mid = {m: band_energies[m].get('vocals', {}).get('midrange (250-2kHz)', 0.0) for m in MODELS}
mfcc_global = {m: np.nanmean(list(mfcc_distances[m].values())) if mfcc_distances[m] else 0.0 for m in MODELS}

radar_metrics = {}
for m in MODELS:
    radar_metrics[m] = {
        'SI-SNR': norm(sisnr_global, True)[m],
        'Vitesse': norm(inference_times, False)[m],
        'Propreté\nspectrale': norm(flux_global, False)[m],
        'Clarté\nbasses': norm(bass_energy, True)[m],
        'Clarté\nvoix': norm(vocals_mid, True)[m],
        'Similarité\ntimbrale': norm(mfcc_global, False)[m],
    }

fig = plot_metrics_radar(radar_metrics, title="Synthèse Radar (Moyenne sur 5 morceaux)")
save_fig(fig, '10_radar')

# --- Final ---
weights = {'SI-SNR': 0.25, 'Vitesse': 0.20, 'Propreté\nspectrale': 0.15, 'Clarté\nbasses': 0.15, 'Clarté\nvoix': 0.10, 'Similarité\ntimbrale': 0.15}
final_scores = {m: round(sum(radar_metrics[m][k] * w for k, w in weights.items()), 4) for m in MODELS}
ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

print("\n🏆 CLASSEMENT FINAL")
for i, (model, score) in enumerate(ranked):
    print(f"  {i+1}. {MODEL_LABELS[model]:<20} → {score:.4f}")

# Final PNG for bar chart omitted for brevity, but results saved to CSV
df_times.to_csv(RESULTS_DIR / 'average_inference_times.csv')
pd.DataFrame(list(final_scores.items()), columns=['model', 'score']).to_csv(RESULTS_DIR / 'final_scores_averaged.csv')
print(f"✅ Résultats sauvegardés dans {RESULTS_DIR}")
