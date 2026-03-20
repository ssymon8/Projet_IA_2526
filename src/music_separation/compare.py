from pathlib import Path
from typing import List, Dict

from .separate import AudioSeparator
from .evaluate import AudioEvaluator
from .config import EXPERIMENTS_DIR

def compare_models_on_track(
    track_path: Path,
    gt_stems_dir: Path,
    models: List[str] = ["htdemucs", "mdx_extra"],
    output_dir: Path = EXPERIMENTS_DIR / "comparison"
) -> Dict[str, Dict]:
    """
    Compare plusieurs modèles sur une unique piste.
    
    Args:
        track_path: Le chemin du son à tester.
        gt_stems_dir: Le dossier contenant la vérité terrain du modèle
        models: Liste des noms des modèles à comparer
        output_dir: Où enregistrer les résultats
    """
    track_path = Path(track_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = AudioEvaluator()
    comparison_results = {}
    
    for model_name in models:
        print(f"--- Comparaison sur le modèle : {model_name} ---")
        separator = AudioSeparator(model_name=model_name)
        
        # Séparation
        model_out_dir = output_dir / model_name / track_path.stem
        predicted_paths = separator.process_file(track_path, model_out_dir)
        
        # Vérité terrain
        track_gt_dir = gt_stems_dir / track_path.stem
        gt_paths = [track_gt_dir / f"{source}.wav" for source in separator.model.sources]
        
        # Métriques
        try:
            metrics = evaluator.compute_bss_metrics(gt_paths, predicted_paths)
            comparison_results[model_name] = metrics
            print(f"Modèle {model_name} - Metrics: {metrics}")
        except FileNotFoundError as e:
            print(f"Impossible d'évaluer {model_name} : erreur de lecture de la vérité terrain. {e}")
            
    return comparison_results
