import torch
import librosa
import numpy as np
import soundfile as sf
import tempfile
from pathlib import Path
from typing import List, Tuple, Union, Dict

from demucs.pretrained import get_model
from demucs.apply import apply_model


class AudioSeparator:
    """
    Classe permettant de gérer la séparation de sources audio (stems) avec le modèle Demucs.
    Idéale pour être instanciée dans un notebook pour du batch processing ou dans une app UI.
    """
    
    def __init__(self, model_name: str = "htdemucs", device: str = None):
        """
        Initialise l'AudioSeparator.
        
        Args:
            model_name (str): Nom du modèle Demucs à utiliser (ex: 'htdemucs', 'htdemucs_ft', 'mdx_extra').
            device (str, optional): 'cuda' ou 'cpu'. S'il n'est pas spécifié, détectera automatiquement.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.sample_rate = 44100  # Fréquence d'échantillonnage par défaut de Demucs
        
    def _load_model(self):
        """Charge le modèle Demucs sur le device (CPU/GPU)."""
        model = get_model(self.model_name)
        model.to(self.device)
        return model

    def load_audio(self, file_path: Union[str, Path]) -> torch.Tensor:
        """
        Charge un fichier audio et le prépare pour l'inférence.
        
        Args:
            file_path (str | Path): Chemin vers le fichier audio d'origine.
            
        Returns:
            torch.Tensor: Tensor du waveform prêt pour Demucs de forme (batch, channels, time).
        """
        # Chargement et resampling forcé à 44100 Hz
        waveform, _ = librosa.load(str(file_path), sr=self.sample_rate, mono=False)
        
        # Si le fichier est mono, on rajoute une dimension pour avoir (channels, time)
        if waveform.ndim == 1:
            waveform = np.expand_dims(waveform, axis=0)
            
        # Conversion en Tensor PyTorch et ajout de la dimension batch: attendu (1, channels, time)
        waveform_tensor = torch.tensor(waveform).float().unsqueeze(0).to(self.device)
        return waveform_tensor

    def separate(self, waveform_tensor: torch.Tensor) -> torch.Tensor:
        """
        Effectue la séparation des stems.
        
        Args:
            waveform_tensor (torch.Tensor): Le tensor audio chargé.
            
        Returns:
            torch.Tensor: Tensor contenant les stems séparés de forme (stems, channels, time).
        """
        with torch.no_grad():
            # apply_model retourne une liste de batchs, on prend le premier élément [0]
            sources = apply_model(self.model, waveform_tensor)[0]
        return sources

    def save_stems(self, sources: torch.Tensor, original_file_name: str, output_dir: Union[str, Path]) -> List[Path]:
        """
        Sauvegarde les stems séparés dans un répertoire donné.
        
        Args:
            sources (torch.Tensor): Tensor des sources séparées.
            original_file_name (str): Nom du fichier d'origine (sans extension) pour prefixer les stems.
            output_dir (str | Path): Répertoire de sortie.
            
        Returns:
            List[Path]: Liste des chemins complets vers les fichiers wav sauvegardés.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        
        for source_tensor, name in zip(sources, self.model.sources):
            out_path = out_dir / f"{original_file_name}_{name}.wav"
            
            # soundfile attend un array (time, channels), d'où le .T
            sf.write(
                str(out_path),
                source_tensor.cpu().numpy().T,
                self.sample_rate,
            )
            saved_paths.append(out_path)
            
        return saved_paths

    def process_file(self, input_path: Union[str, Path], output_dir: Union[str, Path]) -> List[Path]:
        """
        Méthode principale de traitement d'un fichier audio (lecture -> séparation -> sauvegarde).
        
        Args:
            input_path (str | Path): Chemin complet vers le fichier audio.
            output_dir (str | Path): Répertoire où sauvegarder les stems extraits.
            
        Returns:
            List[Path]: Chemins des fichiers générés.
        """
        input_file = Path(input_path)
        
        # 1. Chargement de l'audio
        waveform_tensor = self.load_audio(input_file)
        
        # 2. Séparation
        sources = self.separate(waveform_tensor)
        
        # 3. Sauvegarde physique
        saved_paths = self.save_stems(sources, input_file.stem, output_dir)
        
        return saved_paths

    def process_batch(self, input_paths: List[Union[str, Path]], base_output_dir: Union[str, Path]) -> Dict[str, List[Path]]:
        """
        Traite une liste complète de fichiers. Utile dans un notebook pour calculer des métriques
        sur une boucle de 10 chansons, par exemple.
        Un sous-dossier sera créé pour chaque chanson dans le base_output_dir.
        
        Args:
            input_paths (List[str | Path]): Liste des fichiers audio d'entrée.
            base_output_dir (str | Path): Répertoire racine pour les sorties. 
            
        Returns:
            Dict[str, List[Path]]: Dictionnaire associant chaque fichier d'entrée à la liste de ses stems générés.
        """
        results = {}
        for idx, input_path in enumerate(input_paths):
            file_path = Path(input_path)
            print(f"[{idx+1}/{len(input_paths)}] Traitement en cours : {file_path.name}")
            
            # On stocke les stems de "titre.mp3" dans "base_output_dir/titre/"
            file_output_dir = Path(base_output_dir) / file_path.stem
            
            saved_paths = self.process_file(file_path, file_output_dir)
            results[str(file_path)] = saved_paths
            
        return results

    def process_file_in_memory(self, input_path: Union[str, Path]) -> List[Tuple[str, bytes, str]]:
        """
        Méthode spécifiquement pensée pour une application UI (type Streamlit).
        Elle exécute tout le flux mais renvoie les fichiers directement sous forme de bytes,
        évitant d'encombrer le stockage et permettant de manipuler les fichiers en RAM.
        
        Args:
            input_path (str | Path): Fichier audio à traiter.
            
        Returns:
            List[Tuple[str, bytes, str]]: Renvoie une liste contenant (nom_du_stem, bytes_audio, nom_de_fichier).
        """
        input_file = Path(input_path)
        waveform_tensor = self.load_audio(input_file)
        sources = self.separate(waveform_tensor)
        
        results = []
        # On utilise un dossier temporaire pour l'écriture avec soundfile avant la relecture en bytes
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            for source_tensor, name in zip(sources, self.model.sources):
                file_name = f"{input_file.stem}_{name}.wav"
                out_path = out_dir / file_name
                
                sf.write(
                    str(out_path),
                    source_tensor.cpu().numpy().T,
                    self.sample_rate,
                )
                
                with open(out_path, "rb") as f:
                    audio_bytes = f.read()
                    
                results.append((name, audio_bytes, file_name))
                
        return results

# Exemple d'usage si on lance directement ce script
if __name__ == "__main__":
    separator = AudioSeparator(model_name="htdemucs")
    print("Modèle chargé, prêt pour la séparation !")
    
    # --- Exemple pour tester un ou plusieurs fichiers ---
    # test_files = [Path("song1.mp3"), Path("song2.mp3")]
    # output_dir = Path("metrics_output")
    # results = separator.process_batch(test_files, output_dir)
    # print("Séparation par batch terminée.")
