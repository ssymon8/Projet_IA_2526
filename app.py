import streamlit as st
from pathlib import Path

# Import depuis le package structuré
from src.music_separation import AudioSeparator, CacheManager

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
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.info("⏳ Chargement du modèle Demucs...")
    progress_bar.progress(10)
    separator = get_audio_separator(model_name)

    status_text.info("⏳ Chargement et resampling de l'audio...")
    waveform_tensor = separator.load_audio(input_file_path)
    progress_bar.progress(35)
    
    status_text.info("🎧 Séparation en stems en cours...")
    sources = separator.separate(waveform_tensor)
    progress_bar.progress(80)

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
        # Utilisation du CacheManager pour isoler la complexité de tempfile / shutil
        with CacheManager() as cm:
            input_path = cm.write_uploaded_file(uploaded_file)
            output_dir = cm.create_output_dir("output")

            try:
                output_files = separate_audio(input_path, output_dir, model_choice)

                st.subheader("🎼 Résultats")
                cols = st.columns(2)

                # Création de l'archive ZIP propement par le CacheManager
                archive_path = cm.create_zip_archive(output_dir, "stems_archive")
                with open(archive_path, "rb") as f:
                    zip_bytes = f.read()

            except Exception as e:
                st.error(f"Erreur pendant la séparation : {e}")
                output_files = []
                zip_bytes = None

        # Hors du bloc CacheManager : les fichiers tempos sont supprimés
        # mais on exploite l'audio entièrement lu en mémoire
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