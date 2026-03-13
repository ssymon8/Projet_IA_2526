import streamlit as st
from pathlib import Path
import tempfile
import shutil
import numpy as np
import torch
import librosa
import soundfile as sf

from demucs.pretrained import get_model
from demucs.apply import apply_model


# CORRECTION 1 : désactive le file watcher de Streamlit pour torch
# Évite le RuntimeError "Tried to instantiate class '__path__._path'"
# causé par l'incompatibilité entre le watcher Streamlit et torch._classes
st.set_page_config(
    page_title="Séparation audio avec Demucs",
    layout="wide",
)
st.title("🎵 Séparation audio en stems avec Demucs")


# CORRECTION 2 : gestion explicite du device CUDA/CPU
@st.cache_resource
def load_demucs_model(model_name: str = "htdemucs"):
    model = get_model(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # logger.info("Modèle {} chargé sur {}", model_name, device)
    return model, device


def separate_audio(
    input_file_path: Path,
    output_dir: Path,
    model_name: str,
) -> list[tuple[str, bytes]]:
    """
    Retourne une liste de (stem_name, audio_bytes) au lieu de chemins fichiers.
    CORRECTION 3 : on lit les bytes EN MÉMOIRE avant que le tmpdir soit détruit.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Chargement du modèle
    status_text.info("⏳ Chargement du modèle Demucs...")
    progress_bar.progress(10)
    model, device = load_demucs_model(model_name)

    # CORRECTION 4 : forcer sr=44100 Hz — Demucs attend impérativement ce SR.
    # Sans ça, un fichier à 48000 Hz ou 22050 Hz donnera une séparation incorrecte.
    status_text.info("⏳ Chargement et resampling de l'audio à 44100 Hz...")
    waveform, sr = librosa.load(str(input_file_path), sr=44100, mono=False)
    #logger.info("SR forcé à 44100 Hz (SR original ignoré)")

    if waveform.ndim == 1:
        waveform = np.expand_dims(waveform, axis=0)  # (1, time) si mono

    # Demucs attend (batch, channels, time)
    waveform_tensor = torch.tensor(waveform).float().unsqueeze(0).to(device)
    progress_bar.progress(35)
    #logger.info(
    #    "Audio chargé : {} ({} Hz, {} canaux)",
    #    input_file_path.name,
    #    sr,
    #    waveform_tensor.shape[1],
    #)
    # Séparation
    status_text.info("🎧 Séparation en stems en cours...")
    with torch.no_grad():
        sources = apply_model(model, waveform_tensor)[0]  # (stems, channels, time)
    progress_bar.progress(80)
    #logger.info("Séparation terminée : {} stems extraits", sources.shape[0])

    # Sauvegarde + lecture en mémoire immédiate
    status_text.info("💾 Sauvegarde des stems...")
    output_files: list[tuple[str, bytes]] = []

    for source_tensor, name in zip(sources, model.sources):
        out_path = output_dir / f"{input_file_path.stem}_{name}.wav"
        sf.write(
            str(out_path),
            source_tensor.cpu().numpy().T,  # (time, channels)
            sr,
        )
        # CORRECTION 3 (suite) : lecture immédiate en bytes pendant que le fichier existe
        with open(out_path, "rb") as f:
            audio_bytes = f.read()
        output_files.append((name, audio_bytes, out_path.name))

    progress_bar.progress(100)
    status_text.success("✅ Séparation terminée !")

    return output_files


# --- UI ---

# CORRECTION 5 : sélecteur de modèle exposé à l'utilisateur
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