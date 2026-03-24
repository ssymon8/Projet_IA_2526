"""
analysis_tools.py
=================
Outils d'analyse avancée pour la comparaison des modèles Demucs.
Fournit des métriques spectrales, temporelles et perceptuelles supplémentaires
qui complètent les métriques BSS classiques (SDR/SIR/SAR).
"""

import time
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from scipy import signal

from .audio_utils import load_audio
from .config import DEFAULT_SAMPLE_RATE


# ──────────────────────────────────────────────
# Palette de couleurs cohérente pour les 4 modèles
# ──────────────────────────────────────────────

MODEL_COLORS = {
    "htdemucs":    "#4C9BE8",   # Bleu
    "htdemucs_ft": "#F4845F",   # Orange
    "mdx_extra":   "#56C596",   # Vert
    "mdx_extra_q": "#A77BCA",   # Violet
}

STEM_COLORS = {
    "vocals": "#E8624C",
    "drums":  "#4C9BE8",
    "bass":   "#56C596",
    "other":  "#F4D35E",
}

MODEL_LABELS = {
    "htdemucs":    "HTDemucs",
    "htdemucs_ft": "HTDemucs FT",
    "mdx_extra":   "MDX Extra",
    "mdx_extra_q": "MDX Extra Q",
}

STEM_NAMES = ["vocals", "drums", "bass", "other"]


# ──────────────────────────────────────────────
# Timer context manager
# ──────────────────────────────────────────────

class Timer:
    """Context manager simple pour mesurer le temps d'exécution."""

    def __init__(self):
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start


# ──────────────────────────────────────────────
# Métriques spectrales et temporelles
# ──────────────────────────────────────────────

def compute_spectral_centroid(audio: np.ndarray, sr: int = DEFAULT_SAMPLE_RATE) -> float:
    """Centroïde spectral moyen (Hz) — caractérise la 'brillance' du son."""
    y = audio.mean(axis=0) if audio.ndim == 2 else audio
    centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    return float(np.nanmean(centroids))


def compute_spectral_rolloff(audio: np.ndarray, sr: int = DEFAULT_SAMPLE_RATE, roll_percent: float = 0.85) -> float:
    """Fréquence de roll-off spectral moyen (Hz)."""
    y = audio.mean(axis=0) if audio.ndim == 2 else audio
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=roll_percent)[0]
    return float(np.nanmean(rolloff))


def compute_rms_energy(audio: np.ndarray) -> float:
    """Énergie RMS du signal."""
    return float(np.sqrt(np.mean(audio ** 2)))


def compute_snr(reference: np.ndarray, estimated: np.ndarray) -> float:
    """
    Signal-to-Noise Ratio (dB) entre le signal de référence et l'estimation.
    SNR = 10 * log10(||ref||² / ||ref - est||²)
    """
    min_len = min(reference.shape[-1], estimated.shape[-1])
    ref = reference[..., :min_len]
    est = estimated[..., :min_len]
    noise = ref - est
    power_signal = np.mean(ref ** 2)
    power_noise = np.mean(noise ** 2)
    if power_noise < 1e-10:
        return float("inf")
    return 10.0 * np.log10(power_signal / power_noise)


def compute_si_snr(reference: np.ndarray, estimated: np.ndarray) -> float:
    """Scale-Invariant SNR (dB) — invariant au gain."""
    min_len = min(reference.shape[-1], estimated.shape[-1])
    ref = reference[..., :min_len].flatten()
    est = estimated[..., :min_len].flatten()
    # Centrée
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    # Projection
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-10)
    s_target = alpha * ref
    noise = est - s_target
    return 10.0 * np.log10(np.sum(s_target ** 2) / (np.sum(noise ** 2) + 1e-10))


def compute_spectral_flux(audio: np.ndarray, sr: int = DEFAULT_SAMPLE_RATE) -> float:
    """Flux spectral moyen — mesure les variations temporelles du spectre."""
    y = audio.mean(axis=0) if audio.ndim == 2 else audio
    S = np.abs(librosa.stft(y))
    flux = np.sqrt(np.mean(np.diff(S, axis=1) ** 2))
    return float(flux)


def compute_loudness_lufs(audio: np.ndarray, sr: int = DEFAULT_SAMPLE_RATE) -> float:
    """
    Approximation simplifiée de la loudness intégrée (LUFS-like) via filtre K-weighting simplifié.
    Valeur en dB relatif.
    """
    y = audio.mean(axis=0) if audio.ndim == 2 else audio
    # K-weighting simplifié: pre-filter + RLB (highpass)
    b_pre, a_pre = signal.butter(2, 1681.97 / (sr / 2), btype='high')
    y_filtered = signal.lfilter(b_pre, a_pre, y)
    mean_sq = np.mean(y_filtered ** 2) + 1e-10
    return float(-0.691 + 10.0 * np.log10(mean_sq))


def compute_mfcc_distance(ref: np.ndarray, est: np.ndarray, sr: int = DEFAULT_SAMPLE_RATE, n_mfcc: int = 13) -> float:
    """
    Distance euclidienne moyenne entre les MFCC du signal de référence et de l'estimation.
    Mesure la similarité timbrale.
    """
    min_len = min(ref.shape[-1], est.shape[-1])
    ref_y = (ref.mean(axis=0) if ref.ndim == 2 else ref)[:min_len]
    est_y = (est.mean(axis=0) if est.ndim == 2 else est)[:min_len]

    mfcc_ref = librosa.feature.mfcc(y=ref_y, sr=sr, n_mfcc=n_mfcc)
    mfcc_est = librosa.feature.mfcc(y=est_y, sr=sr, n_mfcc=n_mfcc)

    min_frames = min(mfcc_ref.shape[1], mfcc_est.shape[1])
    dist = np.linalg.norm(mfcc_ref[:, :min_frames] - mfcc_est[:, :min_frames], axis=0)
    return float(np.mean(dist))


def compute_stem_leakage(stems: Dict[str, np.ndarray], stem_name: str) -> Dict[str, float]:
    """
    Évalue le 'leakage' d'un stem vers les autres : corrélation croisée normalisée.
    Utile pour vérifier si les basses se retrouvent dans la piste voix, etc.
    """
    target = stems[stem_name].flatten()
    leakage = {}
    for name, audio in stems.items():
        if name == stem_name:
            continue
        other = audio.flatten()
        min_len = min(len(target), len(other))
        corr = np.corrcoef(target[:min_len], other[:min_len])[0, 1]
        leakage[name] = abs(float(corr))
    return leakage


def compute_frequency_band_energy(audio: np.ndarray, sr: int = DEFAULT_SAMPLE_RATE) -> Dict[str, float]:
    """
    Énergie par bandes de fréquences (sub-bass, bass, mid, high).
    Retourne un dict {bande: énergie_relative}.
    """
    y = audio.mean(axis=0) if audio.ndim == 2 else audio
    fft_vals = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1.0 / sr)

    bands = {
        "sub-bass (20-80 Hz)":  (20, 80),
        "bass (80-250 Hz)":     (80, 250),
        "midrange (250-2kHz)":  (250, 2000),
        "high (2k-20kHz)":      (2000, 20000),
    }

    total = np.sum(fft_vals ** 2) + 1e-10
    result = {}
    for band_name, (f_lo, f_hi) in bands.items():
        mask = (freqs >= f_lo) & (freqs < f_hi)
        energy = np.sum(fft_vals[mask] ** 2) / total
        result[band_name] = float(energy)
    return result


# ──────────────────────────────────────────────
# Fonctions de visualisation avancée
# ──────────────────────────────────────────────

def plot_metrics_radar(
    metrics_per_model: Dict[str, Dict[str, float]],
    title: str = "Comparaison radar des métriques",
    figsize: Tuple[int, int] = (8, 8)
) -> plt.Figure:
    """
    Graphique radar (spider chart) pour comparer plusieurs modèles sur plusieurs métriques.
    
    Args:
        metrics_per_model: {model_name: {metric_name: value}}
    """
    import numpy as np
    
    models = list(metrics_per_model.keys())
    metric_names = list(next(iter(metrics_per_model.values())).keys())
    N = len(metric_names)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    for model in models:
        values = [metrics_per_model[model].get(m, 0) for m in metric_names]
        values += values[:1]
        color = MODEL_COLORS.get(model, "white")
        ax.plot(angles, values, color=color, linewidth=2, label=MODEL_LABELS.get(model, model))
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, color="white", size=11)
    ax.set_yticklabels([])
    ax.spines["polar"].set_color("#333")
    ax.grid(color="#333", linewidth=0.8)
    ax.set_title(title, color="white", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), facecolor="#1a1a2e", 
              edgecolor="#444", labelcolor="white")
    return fig


def plot_metrics_bar_per_stem(
    metrics_per_model: Dict[str, Dict[str, np.ndarray]],
    metric_name: str = "SDR",
    stem_names: List[str] = STEM_NAMES,
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    Barres groupées : un groupe par stem, une barre par modèle.
    """
    models = list(metrics_per_model.keys())
    n_models = len(models)
    n_stems = len(stem_names)
    x = np.arange(n_stems)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    for i, model in enumerate(models):
        vals = metrics_per_model[model].get(metric_name, np.zeros(n_stems))
        if isinstance(vals, np.ndarray):
            vals = [float(v) if not np.isnan(v) else 0.0 for v in vals]
        else:
            vals = [float(vals)] * n_stems
        
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9, 
                      label=MODEL_LABELS.get(model, model),
                      color=MODEL_COLORS.get(model, "gray"),
                      alpha=0.85, edgecolor="white", linewidth=0.4)
        # Labels de valeur sur les barres
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{v:.1f}", ha='center', va='bottom', fontsize=8, color='white')

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in stem_names], color="white", fontsize=12)
    ax.set_ylabel(f"{metric_name} (dB)", color="white", fontsize=12)
    ax.set_title(f"{metric_name} par stem et par modèle", color="white", fontsize=14)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")
    ax.yaxis.grid(True, color="#333", linewidth=0.8, linestyle="--")
    ax.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")
    return fig


def plot_waveforms_comparison(
    stems_per_model: Dict[str, Dict[str, np.ndarray]],
    stem_name: str,
    sr: int = DEFAULT_SAMPLE_RATE,
    max_seconds: float = 10.0,
    figsize: Tuple[int, int] = (16, 10)
) -> plt.Figure:
    """
    Affiche les formes d'onde d'un stem donné pour chaque modèle, côte à côte.
    """
    models = list(stems_per_model.keys())
    n = len(models)
    max_samples = int(max_seconds * sr)

    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
    fig.patch.set_facecolor("#0d1117")
    if n == 1:
        axes = [axes]

    t = None
    for ax, model in zip(axes, models):
        ax.set_facecolor("#0d1117")
        audio = stems_per_model[model].get(stem_name)
        if audio is None:
            ax.set_title(f"{MODEL_LABELS.get(model, model)} — {stem_name} (manquant)", 
                         color="white")
            continue
        y = audio.mean(axis=0) if audio.ndim == 2 else audio
        y = y[:max_samples]
        if t is None:
            t = np.arange(len(y)) / sr
        color = MODEL_COLORS.get(model, "white")
        ax.plot(t, y, color=color, linewidth=0.6, alpha=0.8)
        ax.fill_between(t, y, alpha=0.2, color=color)
        ax.set_title(f"{MODEL_LABELS.get(model, model)}", color="white", fontsize=11)
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#333")
        ax.set_ylabel("Amplitude", color="white", fontsize=9)
        ax.yaxis.grid(True, color="#1a1a2e", linewidth=0.5)

    axes[-1].set_xlabel("Temps (s)", color="white", fontsize=11)
    fig.suptitle(f"Formes d'onde — {stem_name.capitalize()}", color="white", fontsize=15, y=1.01)
    plt.tight_layout()
    return fig


def plot_spectrograms_grid(
    stems_per_model: Dict[str, Dict[str, np.ndarray]],
    stem_name: str,
    sr: int = DEFAULT_SAMPLE_RATE,
    figsize: Tuple[int, int] = (18, 10)
) -> plt.Figure:
    """
    Grille de spectrogrammes log-mel : 1 colonne par modèle pour un même stem.
    """
    models = list(stems_per_model.keys())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True)
    fig.patch.set_facecolor("#0d1117")
    if n == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        ax.set_facecolor("#0d1117")
        audio = stems_per_model[model].get(stem_name)
        if audio is None:
            ax.set_title(f"{MODEL_LABELS.get(model, model)}\n(manquant)", color="white")
            continue
        y = audio.mean(axis=0) if audio.ndim == 2 else audio
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax, cmap='magma')
        ax.set_title(MODEL_LABELS.get(model, model), color="white", fontsize=12)
        ax.tick_params(colors="white")
        ax.set_xlabel("Temps (s)", color="white")
        ax.yaxis.label.set_color("white")

    fig.colorbar(img, ax=axes, format="%+2.0f dB", label="dB")
    fig.suptitle(f"Spectrogrammes — {stem_name.capitalize()}", color="white", fontsize=15)
    return fig


def plot_frequency_bands_heatmap(
    band_energies_per_model: Dict[str, Dict[str, Dict[str, float]]],
    stem_name: str,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Heatmap de l'énergie par bande de fréquences et par modèle pour un stem.
    """
    models = list(band_energies_per_model.keys())
    bands = list(next(iter(next(iter(band_energies_per_model.values())).values())).keys())

    data = np.array([
        [band_energies_per_model[m][stem_name].get(b, 0.0) for b in bands]
        for m in models
    ])

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([b.split("(")[0].strip() for b in bands], color="white", fontsize=10)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in models], color="white", fontsize=11)
    ax.set_title(f"Énergie par bande — {stem_name.capitalize()}", color="white", fontsize=13)

    for i in range(len(models)):
        for j in range(len(bands)):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", 
                    fontsize=9, color="black" if data[i, j] > 0.3 else "white")

    fig.colorbar(im, ax=ax, label="Énergie relative")
    plt.tight_layout()
    return fig


def plot_inference_time_comparison(
    inference_times: Dict[str, float],
    audio_duration: float,
    figsize: Tuple[int, int] = (10, 5)
) -> plt.Figure:
    """
    Graphique barres horizontales du temps d'inférence par modèle.
    Ajoute le facteur temps réel (RTF).
    """
    models = list(inference_times.keys())
    times = [inference_times[m] for m in models]
    rtf = [t / audio_duration for t in times]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor("#0d1117")

    for ax in (ax1, ax2):
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#333")

    colors = [MODEL_COLORS.get(m, "gray") for m in models]
    labels = [MODEL_LABELS.get(m, m) for m in models]

    # Temps absolu
    bars = ax1.barh(labels, times, color=colors, edgecolor="white", linewidth=0.5, alpha=0.85)
    for bar, t in zip(bars, times):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, 
                 f"{t:.1f}s", va="center", ha="left", color="white", fontsize=10)
    ax1.set_xlabel("Temps d'inférence (s)", color="white")
    ax1.set_title("Durée de séparation", color="white", fontsize=12)
    ax1.xaxis.grid(True, color="#333", linestyle="--", linewidth=0.8)
    ax1.set_yticklabels(labels, color="white")

    # Real-Time Factor
    bars2 = ax2.barh(labels, rtf, color=colors, edgecolor="white", linewidth=0.5, alpha=0.85)
    ax2.axvline(x=1.0, color="red", linewidth=1.5, linestyle="--", label="Temps réel (RTF=1)")
    for bar, r in zip(bars2, rtf):
        ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2, 
                 f"×{r:.2f}", va="center", ha="left", color="white", fontsize=10)
    ax2.set_xlabel("Real-Time Factor (RTF)", color="white")
    ax2.set_title("Facteur temps réel", color="white", fontsize=12)
    ax2.xaxis.grid(True, color="#333", linestyle="--", linewidth=0.8)
    ax2.set_yticklabels(labels, color="white")
    ax2.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")

    plt.tight_layout()
    return fig


def plot_si_snr_comparison(
    si_snr_per_model: Dict[str, Dict[str, float]],
    stem_names: List[str] = STEM_NAMES,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """Violin/box plot du SI-SNR par modèle et par stem."""
    n_stems = len(stem_names)
    fig, axes = plt.subplots(1, n_stems, figsize=figsize, sharey=True)
    fig.patch.set_facecolor("#0d1117")
    if n_stems == 1:
        axes = [axes]

    models = list(si_snr_per_model.keys())
    for ax, stem in zip(axes, stem_names):
        ax.set_facecolor("#0d1117")
        vals = [si_snr_per_model[m].get(stem, 0.0) for m in models]
        colors = [MODEL_COLORS.get(m, "gray") for m in models]
        bars = ax.bar([MODEL_LABELS.get(m, m) for m in models], vals, 
                      color=colors, edgecolor="white", linewidth=0.5, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{v:.1f}", ha='center', va='bottom', fontsize=9, color='white')
        ax.set_title(stem.capitalize(), color="white", fontsize=11)
        ax.tick_params(colors="white", axis="both")
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], 
                            rotation=25, ha="right", color="white", fontsize=8)
        ax.spines[:].set_color("#333")
        ax.yaxis.grid(True, color="#333", linestyle="--", linewidth=0.8)
        ax.set_ylabel("SI-SNR (dB)", color="white")

    fig.suptitle("SI-SNR par stem et par modèle", color="white", fontsize=14)
    plt.tight_layout()
    return fig


def plot_mfcc_distance_matrix(
    mfcc_distances: Dict[str, Dict[str, float]],
    stem_names: List[str] = STEM_NAMES,
    figsize: Tuple[int, int] = (12, 4)
) -> plt.Figure:
    """Heatmap de la distance MFCC (similarité timbrale) — plus c'est bas, meilleure est la reconstruction."""
    models = list(mfcc_distances.keys())
    data = np.array([[mfcc_distances[m].get(s, np.nan) for s in stem_names] for m in models])

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(stem_names)))
    ax.set_xticklabels([s.capitalize() for s in stem_names], color="white", fontsize=11)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in models], color="white", fontsize=11)
    ax.set_title("Distance MFCC (↓ meilleur)", color="white", fontsize=13)

    for i in range(len(models)):
        for j in range(len(stem_names)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=10,
                        color="white" if val > np.nanmedian(data) else "black")

    fig.colorbar(im, ax=ax, label="Distance MFCC")
    plt.tight_layout()
    return fig


def plot_spectral_features_comparison(
    spectral_features: Dict[str, Dict[str, Dict[str, float]]],
    stem_names: List[str] = STEM_NAMES,
    figsize: Tuple[int, int] = (16, 8)
) -> plt.Figure:
    """
    Compare les features spectrales (centroïde, rolloff, flux) par modèle et par stem.
    """
    features = ["centroid", "rolloff", "flux"]
    feature_labels = ["Centroïde spectral (Hz)", "Roll-off 85% (Hz)", "Flux spectral"]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.patch.set_facecolor("#0d1117")

    models = list(spectral_features.keys())
    x = np.arange(len(stem_names))
    width = 0.8 / len(models)

    for ax, feat, feat_label in zip(axes, features, feature_labels):
        ax.set_facecolor("#0d1117")
        for i, model in enumerate(models):
            vals = [spectral_features[model].get(s, {}).get(feat, 0.0) for s in stem_names]
            offset = (i - len(models)/2 + 0.5) * width
            ax.bar(x + offset, vals, width * 0.9,
                   label=MODEL_LABELS.get(model, model),
                   color=MODEL_COLORS.get(model, "gray"),
                   alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels([s.capitalize() for s in stem_names], color="white", fontsize=10)
        ax.set_title(feat_label, color="white", fontsize=11)
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#333")
        ax.yaxis.grid(True, color="#333", linestyle="--", linewidth=0.8)

    axes[0].legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", fontsize=9)
    fig.suptitle("Comparaison des caractéristiques spectrales", color="white", fontsize=14)
    plt.tight_layout()
    return fig


def plot_stems_energy_pie(
    rms_per_stem: Dict[str, float],
    model_name: str,
    figsize: Tuple[int, int] = (6, 6)
) -> plt.Figure:
    """Camembert de la distribution d'énergie entre les stems pour un modèle."""
    stems = list(rms_per_stem.keys())
    energies = [max(v, 0) for v in rms_per_stem.values()]
    colors = [STEM_COLORS.get(s, "gray") for s in stems]

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    wedges, texts, autotexts = ax.pie(
        energies, labels=[s.capitalize() for s in stems],
        colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(edgecolor="white", linewidth=1.2)
    )
    for t in texts:
        t.set_color("white")
        t.set_fontsize(12)
    for at in autotexts:
        at.set_color("black")
        at.set_fontsize(10)

    ax.set_title(f"Distribution d'énergie\n{MODEL_LABELS.get(model_name, model_name)}", 
                 color="white", fontsize=13)
    return fig


def plot_sdr_evolution_summary(
    all_results: Dict[str, Dict[str, np.ndarray]],
    stem_names: List[str] = STEM_NAMES,
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Vue synthèse : SDR moyen global et par stem, avec barres d'incertitude (std si plusieurs tracks).
    """
    models = list(all_results.keys())
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor("#0d1117")

    # ── gauche : SDR global moyen ──
    ax = axes[0]
    ax.set_facecolor("#0d1117")
    global_sdrs = []
    for m in models:
        sdr = all_results[m].get("SDR", np.array([]))
        global_sdrs.append(float(np.nanmean(sdr)) if len(sdr) else 0.0)

    colors = [MODEL_COLORS.get(m, "gray") for m in models]
    bars = ax.bar([MODEL_LABELS.get(m, m) for m in models], global_sdrs,
                  color=colors, edgecolor="white", linewidth=0.5, alpha=0.85)
    for bar, v in zip(bars, global_sdrs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{v:.2f} dB", ha='center', va='bottom', fontsize=11, 
                color='white', fontweight='bold')
    ax.set_title("SDR global moyen", color="white", fontsize=13)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")
    ax.yaxis.grid(True, color="#333", linestyle="--", linewidth=0.8)
    ax.set_ylabel("SDR (dB)", color="white")

    # ── droite : SDR par stem ──
    ax2 = axes[1]
    ax2.set_facecolor("#0d1117")
    x = np.arange(len(stem_names))
    width = 0.8 / len(models)
    for i, m in enumerate(models):
        sdr_arr = all_results[m].get("SDR", np.zeros(len(stem_names)))
        if isinstance(sdr_arr, np.ndarray) and len(sdr_arr) >= len(stem_names):
            vals = [float(v) for v in sdr_arr[:len(stem_names)]]
        else:
            vals = [0.0] * len(stem_names)
        offset = (i - len(models)/2 + 0.5) * width
        ax2.bar(x + offset, vals, width * 0.9,
                label=MODEL_LABELS.get(m, m),
                color=MODEL_COLORS.get(m, "gray"),
                edgecolor="white", linewidth=0.3, alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.capitalize() for s in stem_names], color="white", fontsize=11)
    ax2.set_title("SDR par stem", color="white", fontsize=13)
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("#333")
    ax2.yaxis.grid(True, color="#333", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("SDR (dB)", color="white")
    ax2.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")

    plt.tight_layout()
    return fig
