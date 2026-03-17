import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Union
import audio_utils

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
    Comprend le calcul des métriques (SDR, SIR, SAR, ISR) et la génération de visualisations (spectrogrammes).
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def load_audio(self, path: Union[str, Path], mono: bool = False) -> np.ndarray:
        """Charge un fichier audio avec audio_utils."""
        audio, _ = audio_utils.load_audio(str(path), sr=self.sample_rate, mono=mono)
        return audio

    def compute_bss_metrics(self, reference_stems: List[Union[str, Path]], estimated_stems: List[Union[str, Path]]) -> Dict[str, np.ndarray]:
        """
        Calcule les métriques SDR, ISR, SIR, SAR.
        
        ⚠️ IMPORTANT : Ces métriques nécessitent les stems "Vérité Terrain" (Ground Truth) 
        pour comparer avec ce que le modèle a produit ! On ne peut pas calculer un SDR 
        sans avoir la piste originale isolée parfaite correspondante.
        
        Args:
            reference_stems: Liste des chemins vers les vrais stems isolés originaux.
            estimated_stems: Liste des chemins vers les stems produits par le modèle Demucs.
            
        Returns:
            Dict: Dictionnaire contenant les scores SDR, ISR, SIR, SAR médians pour chaque stem.
        """
        if not HAS_MUSEVAL and not HAS_MIR_EVAL:
            raise ImportError(
                "Vous devez installer 'museval' (pour inclure l'ISR) ou 'mir_eval'. "
                "Tapez: pip install museval"
            )

        print("Chargement des audios pour l'évaluation...")
        refs = [self.load_audio(p, mono=False) for p in reference_stems]
        ests = [self.load_audio(p, mono=False) for p in estimated_stems]

        # Formatage pour museval (nécessite (n_samples, n_channels))
        def format_for_eval(audio_list):
            formatted = []
            for a in audio_list:
                if a.ndim == 1:
                    a = np.expand_dims(a, axis=1)  # (time, 1)
                else:
                    a = a.T  # (time, channels)
                formatted.append(a)
            
            # Tronque à la taille minimale pour éviter les problèmes de dimensions
            min_len = min([x.shape[0] for x in formatted])
            return np.stack([x[:min_len, :] for x in formatted])

        refs_np = format_for_eval(refs)
        ests_np = format_for_eval(ests)

        results = {}
        if HAS_MUSEVAL:
            # museval BSS_eval_v4 (SDR, ISR, SIR, SAR)
            # Paramètres par défaut : fenêtres de 1 seconde
            print("Calcul des métriques avec museval (SDR, ISR, SIR, SAR) en cours...")
            sdr, isr, sir, sar = museval.metrics.bss_eval(
                refs_np, ests_np,
                window=self.sample_rate * 1,
                hop=self.sample_rate * 1
            )
            # On prend la médiane sur toutes les fenêtres temporelles pour chaque source
            results = {
                "SDR": np.nanmedian(sdr, axis=1),
                "ISR": np.nanmedian(isr, axis=1),
                "SIR": np.nanmedian(sir, axis=1),
                "SAR": np.nanmedian(sar, axis=1)
            }
        else:
            # Fallback sur mir_eval (SDR, SIR, SAR uniquement, pas d'ISR natif)
            print("Calcul des métriques avec mir_eval (SDR, SIR, SAR) en cours...")
            
            # mir_eval prend (sources, canaux * temps) ou mono (sources, temps)
            # Simplification : on moyennise en mono pour mir_eval classique
            refs_mono = np.array([x.mean(axis=1) for x in refs_np])
            ests_mono = np.array([x.mean(axis=1) for x in ests_np])
            
            sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(refs_mono, ests_mono)
            results = {
                "SDR": sdr,
                "SIR": sir,
                "SAR": sar
            }

        return results

    def plot_spectrograms(self, audio_paths_dict: Dict[str, Union[str, Path]], output_path: Union[str, Path] = None):
        """
        Génère une figure comparant les spectrogrammes de plusieurs fichiers (ex: Mix, Bass, Drums, etc).
        Les spectrogrammes permettent de visuellement analyser le spectre des fréquences.
        
        Args:
            audio_paths_dict: Un dict { 'Titre du graphe': 'chemin/vers/audio.wav' }
            output_path: Chemin où sauvegarder l'image générée (optionnel)
            
        Returns:
            matplotlib.figure.Figure: L'objet Figure.
        """
        n_plots = len(audio_paths_dict)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots), sharex=True, sharey=True)
        
        if n_plots == 1:
            axes = [axes]
            
        for ax, (title, path) in zip(axes, audio_paths_dict.items()):
            try:
                # Charger en mono pour le spectrogramme
                y = self.load_audio(path, mono=True)
                
                # Transformée de Fourier à Court Terme (STFT)
                D = librosa.stft(y)
                S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                
                # Tracé du spectrogramme
                img = librosa.display.specshow(S_db, sr=self.sample_rate, x_axis='time', y_axis='log', ax=ax, cmap='magma')
                ax.set_title(f"Spectrogramme : {title}")
                fig.colorbar(img, ax=ax, format="%+2.0f dB")
                
            except Exception as e:
                ax.set_title(f"Erreur avec {title} : {e}")
                
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150)
            print(f"Spectrogrammes sauvegardés sous : {output_path}")
            
        return fig

# Exemple d'utilisation
if __name__ == "__main__":
    evaluator = AudioEvaluator()
    
    # --- 1. Exemple Visuel (Spectrogrammes) ---
    # Ici, tu passes juste le mix original et les outputs générés ! Pas besoin de vérité terrain.
    # a_tester = {
    #     "Mix Original": "chanson1.wav",
    #     "Voix (Séparée)": "output/chanson1_vocals.wav",
    #     "Batterie (Séparée)": "output/chanson1_drums.wav"
    # }
    # evaluator.plot_spectrograms(a_tester, output_path="spectro_chanson1.png")
    
    # --- 2. Exemple Métriques (SDR, SIR...) ---
    # /!\ ATTENTION : Il faut avoir les VRAIES pistes isolées de la chanson (Ground Truth) !
    # true_stems = ["true_vocals.wav", "true_drums.wav", "true_bass.wav", "true_other.wav"]
    # estimated_stems = ["demucs_vocals.wav", "demucs_drums.wav", "demucs_bass.wav", "demucs_other.wav"]
    # metrics = evaluator.compute_bss_metrics(true_stems, estimated_stems)
    # print(metrics)
