import numpy as np
from pathlib import Path
from typing import List, Dict, Union

from . import audio_utils
from .config import DEFAULT_SAMPLE_RATE

try:
    import museval
    HAS_MUSEVAL = True
except ImportError:
    HAS_MUSEVAL = False

try:
    import mir_eval
    HAS_MIR_EVAL = True
except ImportError:
    HAS_MIR_EVAL = False


class AudioEvaluator:
    """
    Classe permettant d'évaluer la qualité de la séparation audio.
    Gère le calcul des métriques (SDR, SIR, SAR, ISR).
    Les visualisations ont été déplacées vers visualization.py.
    """

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE):
        self.sample_rate = sample_rate

    def load_audio(self, path: Union[str, Path], mono: bool = False) -> np.ndarray:
        """Charge un fichier audio avec audio_utils."""
        audio, _ = audio_utils.load_audio(str(path), sr=self.sample_rate, mono=mono)
        return audio

    def compute_bss_metrics(self, reference_stems: List[Union[str, Path]], estimated_stems: List[Union[str, Path]]) -> Dict[str, np.ndarray]:
        """
        Calcule les métriques SDR, ISR, SIR, SAR.
        Nécessite les stems "Vérité Terrain" pour évaluer.
        """
        if not HAS_MUSEVAL and not HAS_MIR_EVAL:
            raise ImportError(
                "Vous devez installer 'museval' (pour inclure l'ISR) ou 'mir_eval'."
            )

        print("Chargement des audios pour l'évaluation...")
        refs = [self.load_audio(p, mono=False) for p in reference_stems]
        ests = [self.load_audio(p, mono=False) for p in estimated_stems]

        def format_for_eval(audio_list):
            formatted = []
            for a in audio_list:
                if a.ndim == 1:
                    a = np.expand_dims(a, axis=1)  # (time, 1)
                else:
                    a = a.T  # (time, channels)
                formatted.append(a)
            
            min_len = min([x.shape[0] for x in formatted])
            return np.stack([x[:min_len, :] for x in formatted])

        refs_np = format_for_eval(refs)
        ests_np = format_for_eval(ests)

        results = {}
        if HAS_MUSEVAL:
            print("Calcul des métriques avec museval (SDR, ISR, SIR, SAR)...")
            sdr, isr, sir, sar = museval.metrics.bss_eval(
                refs_np, ests_np,
                window=self.sample_rate * 1,
                hop=self.sample_rate * 1
            )
            results = {
                "SDR": np.nanmedian(sdr, axis=1),
                "ISR": np.nanmedian(isr, axis=1),
                "SIR": np.nanmedian(sir, axis=1),
                "SAR": np.nanmedian(sar, axis=1)
            }
        else:
            print("Calcul des métriques avec mir_eval (SDR, SIR, SAR)...")
            refs_mono = np.array([x.mean(axis=1) for x in refs_np])
            ests_mono = np.array([x.mean(axis=1) for x in ests_np])
            
            sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(refs_mono, ests_mono)
            results = {
                "SDR": sdr,
                "SIR": sir,
                "SAR": sar
            }

        return results
