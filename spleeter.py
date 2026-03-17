import argparse
from pathlib import Path

from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter


def parse_args():
    parser = argparse.ArgumentParser(description="Séparation de sources avec Spleeter.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="data/input/good for the ghost - Alge.mp3",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="data/output_spleeter",
    )
    parser.add_argument(
        "--stems",
        type=int,
        default=2,
        choices=[2, 4, 5],
        help="Nombre de sources à séparer (2, 4 ou 5).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Fichier d'entrée introuvable : {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Charger le modèle Spleeter (ex: 2stems : voix + accompagnement)
    model_name = f"spleeter:{args.stems}stems"
    print(f"Chargement du modèle {model_name} ...")
    separator = Separator(model_name)

    # 2) Charger l'audio
    print(f"Chargement de l'audio depuis {input_path} ...")
    audio_loader = AudioAdapter.default()
    waveform, sample_rate = audio_loader.load(
        str(input_path),
        sample_rate=44100  # ou None pour le sample rate original
    )

    # 3) Séparation
    print("Séparation des sources ...")
    prediction = separator.separate(waveform)

    # 4) Sauvegarde de chaque source au format .wav
    print(f"Sauvegarde des sources dans {output_dir} ...")
    for source_name, source_waveform in prediction.items():
        # ex: vocals, accompaniment, drums, bass, other
        out_file = output_dir / f"{input_path.stem}_{source_name}.wav"
        audio_loader.save(str(out_file), source_waveform, sample_rate)
        print(f"  -> {source_name} : {out_file}")

  


if __name__ == "__main__":
    main()