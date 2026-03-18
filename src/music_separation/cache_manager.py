import tempfile
import shutil
from pathlib import Path
from typing import Union, List

class CacheManager:
    """Gestion propre des fichiers temporaires et des caches en cours d'exécution."""
    
    def __init__(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self._temp_dir.name)
        
    def get_temp_path(self, filename: str) -> Path:
        """Retourne un chemin pour un fichier temporaire."""
        return self.base_path / filename

    def create_output_dir(self, dirname: str = "output") -> Path:
        """Crée et retourne un sous-dossier de sortie."""
        out_dir = self.base_path / dirname
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir
        
    def write_uploaded_file(self, uploaded_file) -> Path:
        """Enregistre un fichier uploadé (ex. depuis Streamlit) et retourne son chemin."""
        file_path = self.get_temp_path(uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        return file_path
        
    def create_zip_archive(self, source_dir: Union[str, Path], archive_name: str = "stems_archive") -> Path:
        """Créé une archive ZIP du dossier source."""
        zip_base = self.base_path / archive_name
        archive_path = shutil.make_archive(str(zip_base), "zip", str(source_dir))
        return Path(archive_path)

    def cleanup(self):
        """Supprime le dossier temporaire."""
        self._temp_dir.cleanup()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
