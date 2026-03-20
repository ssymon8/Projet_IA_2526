from separate import AudioSeparator
from evaluate import AudioEvaluator

separator = AudioSeparator()
evaluator = AudioEvaluator()

# --- 1. Tu lances la Séparation ---
# On sépare "ma_chanson_mix.wav" -> génère 4 stems dans "dossier_output/"
separator.process_file("dataset/musdb18hq/test/Arise - Run Run Run/mixture.wav", "output/")

# --- 2. Tu Visualises les Fréquences ! ---
# On lui donne un dictionnaire { 'Titre': 'chemin/vers/audio.wav' }
fichiers_a_tracer = {
    "Mix Original": "dataset/musdb18hq/test/Arise - Run Run Run/mixture.wav",
    "Voix Estimée": "dataset/musdb18hq/test/Arise - Run Run Run/vocals.wav",
    "Batterie Estimée": "dataset/musdb18hq/test/Arise - Run Run Run/drums.wav"
}
evaluator.plot_spectrograms(fichiers_a_tracer, output_path="comparatif_frequences.png")


# --- 3. Tu calcules les Métriques Officielles ! ---
# (Il faut que tu aies les vraies pistes isolées sous la main pour comparer)
vraies_pistes = ["dataset/musdb18hq/test/Arise - Run Run Run/vocals.wav", 
                "dataset/musdb18hq/test/Arise - Run Run Run/drums.wav", 
                "dataset/musdb18hq/test/Arise - Run Run Run/bass.wav", 
                "dataset/musdb18hq/test/Arise - Run Run Run/other.wav"]
 
pistes_generees_par_demucs = [
    "output/Arise - Run Run Run/mixture_vocals.wav",
    "output/Arise - Run Run Run/mixture_drums.wav",
    "output/Arise - Run Run Run/mixture_bass.wav",
    "output/Arise - Run Run Run/mixture_other.wav"
]

metrics = evaluator.compute_bss_metrics(vraies_pistes, pistes_generees_par_demucs)
print(f"SDR moyen : {metrics['SDR']}")
print(f"ISR moyen : {metrics['ISR']}")
