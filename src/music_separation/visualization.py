import io
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from typing import Dict, Union
from pathlib import Path

from .audio_utils import load_audio
from .config import DEFAULT_SAMPLE_RATE

class Visualizer:
    """Outils de visualisation audio implémentant spectrogrammes et ondes."""
    
    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE):
        self.sample_rate = sample_rate

    def load_mono(self, path: Union[str, Path]) -> np.ndarray:
        """Charge spécifiquement en mono pour la visualisation."""
        audio, _ = load_audio(path, sr=self.sample_rate, mono=True)
        return audio

    def _audio_array_to_mono(self, audio_bytes: bytes) -> np.ndarray:
        """Charge des bytes audio et les convertit en array mono."""
        buf = io.BytesIO(audio_bytes)
        y, _ = librosa.load(buf, sr=self.sample_rate, mono=True)
        return y

    def _render_spectrogram(self, y: np.ndarray, title: str) -> bytes:
        """Génère un spectrogramme à partir d'un array audio et le retourne en PNG bytes."""
        fig, ax = plt.subplots(figsize=(12, 4))
        D = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img = librosa.display.specshow(S_db, sr=self.sample_rate, x_axis='time', y_axis='log', ax=ax, cmap='magma')
        ax.set_title(f"Spectrogramme : {title}")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    def spectrogram_from_path(self, path: Union[str, Path], title: str) -> bytes:
        """Génère un spectrogramme depuis un fichier audio et retourne les bytes PNG."""
        y = self.load_mono(path)
        return self._render_spectrogram(y, title)

    def spectrogram_from_bytes(self, audio_bytes: bytes, title: str) -> bytes:
        """Génère un spectrogramme depuis des bytes audio et retourne les bytes PNG."""
        y = self._audio_array_to_mono(audio_bytes)
        return self._render_spectrogram(y, title)

    def plot_spectrograms(self, audio_paths_dict: Dict[str, Union[str, Path]], output_path: Union[str, Path] = None):
        """
        Génère une figure comparant les spectrogrammes de plusieurs fichiers.
        
        Args:
            audio_paths_dict: Dictionnaire { 'Titre du graphe': 'chemin/vers/audio.wav' }
            output_path: Chemin où sauvegarder l'image générée (optionnel)
        """
        n_plots = len(audio_paths_dict)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots), sharex=True, sharey=True)
        
        if n_plots == 1:
            axes = [axes]
            
        for ax, (title, path) in zip(axes, audio_paths_dict.items()):
            try:
                y = self.load_mono(path)
                D = librosa.stft(y)
                S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                
                img = librosa.display.specshow(S_db, sr=self.sample_rate, x_axis='time', y_axis='log', ax=ax, cmap='magma')
                ax.set_title(f"Spectrogramme : {title}")
                fig.colorbar(img, ax=ax, format="%+2.0f dB")
                
            except Exception as e:
                ax.set_title(f"Erreur avec {title} : {e}")
                
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150)
            
        return fig

    def plot_waveform(self, path: Union[str, Path], title: str = "Waveform", output_path: Union[str, Path] = None):
        """
        Affiche la forme d'onde temporelle d'un fichier audio.
        """
        y = self.load_mono(path)
        fig, ax = plt.subplots(figsize=(12, 3))
        librosa.display.waveshow(y, sr=self.sample_rate, ax=ax, color="blue", alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("Amplitude")
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150)
            
        return fig
