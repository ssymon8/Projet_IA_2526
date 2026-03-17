import numpy as np
import librosa
import soundfile as sf
from typing import Union, List, Tuple
from pathlib import Path


def load_audio(path: Union[str, Path], sr: int = 44100, mono: bool = False, force_stereo: bool = False) -> Tuple[np.ndarray, int]:
    """
    Charge un fichier audio.
    
    Args:
        path: Chemin vers le fichier.
        sr: Fréquence d'échantillonnage cible (44100 par défaut).
        mono: Si True, force le fichier en mono (1D array: time,).
        force_stereo: Si True et que l'audio est mono, le duplique sur 2 canaux (2, time).
        
    Returns:
        (audio_array, sample_rate)
    """
    audio, loaded_sr = librosa.load(str(path), sr=sr, mono=mono)
    
    # librosa retourne (time,) pour mono et (channels, time) pour stéréo quand mono=False.
    if force_stereo and audio.ndim == 1:
        audio = to_stereo(audio)
    elif not mono and not force_stereo and audio.ndim == 1:
        # Par défaut, on s'assure d'avoir au moins (1, time) si ce n'est pas explicitement mono
        audio = np.expand_dims(audio, axis=0)
        
    return audio, loaded_sr


def save_audio(path: Union[str, Path], audio: np.ndarray, sr: int = 44100):
    """
    Sauvegarde un tableau numpy en fichier audio WAV.
    
    Args:
        path: Chemin de sauvegarde.
        audio: Tableau contenant l'audio. Peut être (time,) ou (channels, time).
        sr: Fréquence d'échantillonnage.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # soundfile attend un array (time, channels) pour l'écriture
    if audio.ndim == 2:
        audio = audio.T
        
    sf.write(str(path), audio, sr)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalise le volume de l'audio (Peak Normalization) entre -1.0 et 1.0.
    """
    peak = np.abs(audio).max()
    if peak > 0:
        return audio / peak
    return audio


def to_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convertit un signal (channels, time) en (time,) en moyennant les canaux.
    """
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=0)


def to_stereo(audio: np.ndarray) -> np.ndarray:
    """
    Convertit un signal mono (time,) ou (1, time) en stéréo (2, time).
    """
    if audio.ndim == 1:
        return np.vstack((audio, audio))
    elif audio.shape[0] == 1:
        return np.vstack((audio[0], audio[0]))
    return audio


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Change la fréquence d'échantillonnage de l'audio.
    """
    if orig_sr == target_sr:
        return audio
        
    # librosa.resample s'attend à (..., time)
    return librosa.resample(y=audio, orig_sr=orig_sr, target_sr=target_sr)


def slice_audio(audio: np.ndarray, start_sec: float, end_sec: float, sr: int = 44100) -> np.ndarray:
    """
    Découpe une portion temporelle de l'audio.
    """
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    
    if audio.ndim == 1:
        return audio[start_sample:end_sample]
    else:
        return audio[:, start_sample:end_sample]


def mix_stems(stems: List[np.ndarray], normalize: bool = True) -> np.ndarray:
    """
    Mélange plusieurs stems audios (somme des tableaux) pour reformer le mix complet.
    
    Args:
        stems: Liste de tableaux numpy représentant les pistes.
        normalize: Si True, normalise le résultat final pour éviter la saturation.
    """
    if not stems:
        raise ValueError("La liste de stems est vide.")
        
    mix = sum(stems)
    if normalize:
        mix = normalize_audio(mix)
    return mix


def get_duration(audio_or_path: Union[str, Path, np.ndarray], sr: int = 44100) -> float:
    """
    Calcule la durée de l'audio en secondes.
    """
    if isinstance(audio_or_path, (str, Path)):
        return librosa.get_duration(path=str(audio_or_path))
    else:
        # Array numpy
        time_axis_len = audio_or_path.shape[-1]
        return time_axis_len / sr
