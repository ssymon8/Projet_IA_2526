import streamlit as st
from pathlib import Path
import tempfile
import shutil
import numpy as np
import torch
import librosa
import soundfile as sf

from separate import AudioSeparator


st.set_page_config(
    page_title="Séparation audio avec Demucs",
    layout="wide",
)
st.title("🎵 Séparation audio en stems avec Demucs")



@st.cache_resource
def get_audio_separator(model_name: str = "htdemucs"):
    return AudioSeparator(model_name=model_name)


def separate_audio(
    input_file_path: Path,
    output_dir: Path,
    model_name: str,
) -> list[tuple[str, bytes, str]]:
    """
    Retourne une liste de (stem_name, audio_bytes, nom_fichier) au lieu de chemins fichiers.
    Utilise la classe AudioSeparator pour faciliter l'intégration et factoriser le code.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Chargement du modèle
    status_text.info("⏳ Chargement du modèle Demucs...")
    progress_bar.progress(10)
    separator = get_audio_separator(model_name)


    status_text.info("⏳ Chargement et resampling de l'audio...")
    waveform_tensor = separator.load_audio(input_file_path)

    progress_bar.progress(35)
    
    # Séparation
    status_text.info("🎧 Séparation en stems en cours...")
    sources = separator.separate(waveform_tensor)
    
    progress_bar.progress(80)

    # Sauvegarde + lecture en mémoire immédiate
    status_text.info("💾 Sauvegarde et préparation des stems...")
    saved_paths = separator.save_stems(sources, input_file_path.stem, output_dir)
    
    output_files = []
    for name, out_path in zip(separator.model.sources, saved_paths):
        with open(out_path, "rb") as f:
            audio_bytes = f.read()
        output_files.append((name, audio_bytes, out_path.name))

    progress_bar.progress(100)
    status_text.success("✅ Séparation terminée !")

    return output_files


# --- UI ---
model_choice = st.selectbox(
    "Modèle Demucs",
    options=["htdemucs", "htdemucs_ft", "mdx_extra", "mdx_extra_q"],
    help="htdemucs = rapide. htdemucs_ft / mdx_extra = meilleure qualité, plus lent.",
)

uploaded_file = st.file_uploader(
    "Dépose un fichier audio",
    type=["mp3", "wav", "flac", "ogg", "m4a"],
)

if uploaded_file is not None:
    st.subheader("Fichier d'entrée")
    st.audio(uploaded_file)

    if st.button("Lancer la séparation"):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / uploaded_file.name
            output_dir = tmp_path / "output"

            with open(input_path, "wb") as f:
                f.write(uploaded_file.read())

            try:
                output_files = separate_audio(input_path, output_dir, model_choice)

                # CORRECTION 3 (suite) : le tmpdir est encore ouvert ici,
                # mais on a déjà tout lu en mémoire → pas de risque.

                st.subheader("🎼 Résultats")
                cols = st.columns(2)

                # Construction du zip en mémoire avant de quitter le with
                zip_base = tmp_path / "stems_archive"
                archive_path = shutil.make_archive(str(zip_base), "zip", output_dir)
                with open(archive_path, "rb") as f:
                    zip_bytes = f.read()

            except Exception as e:
                st.error(f"Erreur pendant la séparation : {e}")
                #logger.exception("Erreur de séparation")
                output_files = []
                zip_bytes = None

        # Hors du with tmpdir : on utilise uniquement les bytes déjà lus
        if output_files:
            cols = st.columns(2)
            for i, (stem_name, audio_bytes, filename) in enumerate(output_files):
                with cols[i % 2]:
                    st.markdown(f"### {stem_name.capitalize()}")
                    st.audio(audio_bytes, format="audio/wav")
                    st.download_button(
                        label=f"Télécharger {stem_name}",
                        data=audio_bytes,
                        file_name=filename,
                        mime="audio/wav",
                    )

            if zip_bytes:
                st.download_button(
                    label="📦 Télécharger tous les stems (.zip)",
                    data=zip_bytes,
                    file_name="stems_demucs.zip",
                    mime="application/zip",
                )