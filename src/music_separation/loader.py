import os
from pathlib import Path
from typing import Union, Tuple
import numpy as np

from .audio_utils import load_audio
from .config import DEFAULT_SAMPLE_RATE

class AudioLoader:
    """Classe chargeant l'audio depuis différentes sources (local, datasets)."""
    
    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE):
        self.sample_rate = sample_rate

    def load_local(self, file_path: Union[str, Path], mono: bool = False, force_stereo: bool = False) -> Tuple[np.ndarray, int]:
        """Charge un fichier audio local."""
        return load_audio(file_path, sr=self.sample_rate, mono=mono, force_stereo=force_stereo)

    def load_musdb(self, dataset_path: Union[str, Path], track_name: str) -> Tuple[np.ndarray, int]:
        """
        Charge un titre depuis la base MUSDB structurée.
        """
        dataset_path = Path(dataset_path)
        track_dir = dataset_path / track_name
        
        if not track_dir.exists():
            # Alternative: c'est un seul fichier .wav (musdb_hq)
            track_file = dataset_path / f"{track_name}.wav"
            if track_file.exists():
                return self.load_local(track_file, force_stereo=True)
            raise FileNotFoundError(f"Track introuvable pour {track_name}")
            
        # Si c'est un dossier (stems séparés), on charge le mix principal
        mixture_path = track_dir / "mixture.wav"
        if not mixture_path.exists():
            raise FileNotFoundError(f"mixture.wav introuvable dans {track_dir}")
            
        return self.load_local(mixture_path, force_stereo=True)
