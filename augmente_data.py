import os
from pydub import AudioSegment
from itertools import combinations
import static_ffmpeg

# Initialisation FFmpeg
static_ffmpeg.add_paths()

# Configuration
input_dir = "./Bach10_v1.1" 
output_dir = "./Bach10_Augmented"

# On définit les instruments tels qu'ils apparaissent REELLEMENT dans tes fichiers
target_mapping = {
    "violin": ["violin", "Violin"],
    "clarinet": ["clarinet", "Clarinet"],
    "saxophone": ["saxphone", "saxophone", "Saxophone"], 
    "bassoon": ["bassoon", "Bassoon"]
}

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

pieces = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f)) and not f.startswith('.') and f != 'Code']
print(f"Pièces détectées : {pieces}")

for piece in pieces:
    piece_path = os.path.join(input_dir, piece)
    available_tracks = {}
    
    print(f"\n--- Analyse de : {piece} ---")
    all_files = os.listdir(piece_path)
    
    for file_name in all_files:
        # On ignore ce qui n'est pas du son ou qui est le mixage complet
        if not file_name.lower().endswith(".wav"): continue
        # Si le fichier contient juste le nom du morceau sans instrument (ex: 01-AchGottundHerr.wav)
        # on considère que c'est l'ensemble 
        if file_name.lower() == f"{piece.lower()}.wav": continue

        for clean_name, keywords in target_mapping.items():
            if any(key in file_name for key in keywords):
                print(f"  [OK] Trouvé {clean_name} -> {file_name}")
                available_tracks[clean_name] = AudioSegment.from_wav(os.path.join(piece_path, file_name))
                break
    
    if len(available_tracks) < 2:
        print(f"  [!] Pas assez de pistes pour {piece}. Trouvées : {list(available_tracks.keys())}")
        continue

    # Génération des combinaisons (Duos, Trios, Quatuors) [cite: 55, 56, 79]
    for r in range(2, 5):
        for combo in combinations(available_tracks.keys(), r):
            combo_name = "-".join(combo)
            current_target = os.path.join(output_dir, f"{piece}_{combo_name}")
            os.makedirs(current_target, exist_ok=True)
            
            mixture = None
            for inst_name in combo:
                track = available_tracks[inst_name]
                # On sauve avec le nom propre pour Demucs
                track.export(os.path.join(current_target, f"{inst_name}.wav"), format="wav")
                mixture = track if mixture is None else mixture.overlay(track)
            
            mixture.export(os.path.join(current_target, "mixture.wav"), format="wav")
            print(f"    -> Créé : {piece}_{combo_name}")

print("\nTerminé ! Tu peux maintenant vérifier Bach10_Augmented.")