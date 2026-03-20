import torch
import numpy as np
import tempfile
from pathlib import Path
from typing import List, Tuple, Union, Dict

from demucs.pretrained import get_model
from demucs.apply import apply_model

from . import audio_utils
from .config import DEFAULT_SAMPLE_RATE, DEFAULT_DEVICE, DEFAULT_MODEL

class AudioSeparator:
    """
    Classe permettant de gérer la séparation de sources audio (stems) avec le modèle Demucs.
    Idéale pour être instanciée dans un notebook pour du batch processing ou dans une app UI.
    """
    
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = None):
        """
        Initialise l'AudioSeparator.
        
        Args:
            model_name (str): Nom du modèle Demucs à utiliser.
            device (str, optional): 'cuda' ou 'cpu'.
        """
        self.model_name = model_name
        self.device = device or DEFAULT_DEVICE
        self.model = self._load_model()
        self.sample_rate = DEFAULT_SAMPLE_RATE
        
    def _load_model(self):
        """Charge le modèle Demucs sur le device (CPU/GPU)."""
        model = get_model(self.model_name)
        model.to(self.device)
        return model

    def load_audio(self, file_path: Union[str, Path]) -> torch.Tensor:
        """Charge un fichier audio et le prépare pour l'inférence."""
        waveform, _ = audio_utils.load_audio(file_path, sr=self.sample_rate, force_stereo=True)
            
        # Conversion en Tensor PyTorch et ajout de la dimension batch: attendu (1, channels, time)
        waveform_tensor = torch.tensor(waveform).float().unsqueeze(0).to(self.device)
        return waveform_tensor

    def separate(self, waveform_tensor: torch.Tensor) -> torch.Tensor:
        """Effectue la séparation des stems."""
        with torch.no_grad():
            # apply_model retourne une liste de batchs, on prend le premier élément [0]
            sources = apply_model(self.model, waveform_tensor)[0]
        return sources

    def save_stems(self, sources: torch.Tensor, original_file_name: str, output_dir: Union[str, Path]) -> List[Path]:
        """Sauvegarde les stems séparés dans un répertoire donné."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        
        for source_tensor, name in zip(sources, self.model.sources):
            out_path = out_dir / f"{original_file_name}_{name}.wav"
            
            audio_utils.save_audio(out_path, source_tensor.cpu().numpy(), sr=self.sample_rate)
            saved_paths.append(out_path)
            
        return saved_paths

    def process_file(self, input_path: Union[str, Path], output_dir: Union[str, Path]) -> List[Path]:
        """Méthode principale de traitement d'un fichier audio."""
        input_file = Path(input_path)
        waveform_tensor = self.load_audio(input_file)
        sources = self.separate(waveform_tensor)
        saved_paths = self.save_stems(sources, input_file.stem, output_dir)
        return saved_paths

    def process_batch(self, input_paths: List[Union[str, Path]], base_output_dir: Union[str, Path]) -> Dict[str, List[Path]]:
        """Traite une liste complète de fichiers."""
        results = {}
        for idx, input_path in enumerate(input_paths):
            file_path = Path(input_path)
            print(f"[{idx+1}/{len(input_paths)}] Traitement en cours : {file_path.name}")
            
            file_output_dir = Path(base_output_dir) / file_path.stem
            saved_paths = self.process_file(file_path, file_output_dir)
            results[str(file_path)] = saved_paths
            
        return results

    def process_file_in_memory(self, input_path: Union[str, Path]) -> List[Tuple[str, bytes, str]]:
        """
        Méthode spécifiquement pensée pour une application UI (type Streamlit).
        Elle exécute tout le flux mais renvoie les fichiers directement sous forme de bytes.
        """
        input_file = Path(input_path)
        waveform_tensor = self.load_audio(input_file)
        sources = self.separate(waveform_tensor)
        
        results = []
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            for source_tensor, name in zip(sources, self.model.sources):
                file_name = f"{input_file.stem}_{name}.wav"
                out_path = out_dir / file_name
                
                audio_utils.save_audio(out_path, source_tensor.cpu().numpy(), sr=self.sample_rate)
                
                with open(out_path, "rb") as f:
                    audio_bytes = f.read()
                    
                results.append((name, audio_bytes, file_name))
                
        return results
