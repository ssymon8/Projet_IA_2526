from pathlib import Path
from typing import List, Dict

from .separate import AudioSeparator
from .evaluate import AudioEvaluator
from .config import DEFAULT_MODEL, EXPERIMENTS_DIR

def run_benchmark(
    model_name: str, 
    tracks: List[Path], 
    gt_stems_dir: Path, 
    output_dir: Path = EXPERIMENTS_DIR / "benchmark"
) -> Dict[str, Dict]:
    """
    Lance une évaluation du modèle sur un grand nombre de morceaux.
    
    Args:
        model_name: Modèle (ex: "htdemucs")
        tracks: Liste des pistes audios 'mix' à séparer
        gt_stems_dir: Répertoire contenant les GT (Ground Truth)
        output_dir: Répertoire pour sauvegarder les prédictions
    """
    separator = AudioSeparator(model_name=model_name)
    evaluator = AudioEvaluator()
    results = {}
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for track in tracks:
        print(f"Benchmarking {track.name} avec {model_name}...")
        
        # Séparation
        track_out_dir = output_dir / track.stem
        predicted_paths = separator.process_file(track, track_out_dir)
        
        # Récupération de la vérité terrain (simplifié, dépend du nommage de votre dataset)
        track_gt_dir = gt_stems_dir / track.stem
        gt_paths = [track_gt_dir / f"{source}.wav" for source in separator.model.sources]
        
        # Évaluation (Ne marchera que si toutes les GT existent)
        try:
            metrics = evaluator.compute_bss_metrics(gt_paths, predicted_paths)
            results[track.stem] = metrics
            print(f"Metrics: {metrics}")
        except FileNotFoundError:
            print(f"Vérité terrain absente pour {track.stem}, métriques ignorées.")
            
    return results
