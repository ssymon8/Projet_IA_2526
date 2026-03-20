import streamlit as st
from pathlib import Path

# Import depuis le package structuré
from src.music_separation import AudioSeparator, CacheManager, Visualizer

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
    progress_bar.progress(30)

    status_text.info("🎧 Séparation en stems en cours...")
    sources = separator.separate(waveform_tensor)
    progress_bar.progress(70)

    status_text.info("💾 Sauvegarde et préparation des stems...")
    saved_paths = separator.save_stems(sources, input_file_path.stem, output_dir)

    output_files = []
    for name, out_path in zip(separator.model.sources, saved_paths):
        with open(out_path, "rb") as f:
            audio_bytes = f.read()
        output_files.append((name, audio_bytes, out_path.name))

    progress_bar.progress(80)

    # --- Génération des spectrogrammes (stockés en session_state) ---
    status_text.info("📊 Génération des spectrogrammes...")
    viz = Visualizer()
    spectrograms = {}

    spectrograms["Original"] = viz.spectrogram_from_path(input_file_path, "Original")
    for name, audio_bytes, _ in output_files:
        spectrograms[name.capitalize()] = viz.spectrogram_from_bytes(audio_bytes, name.capitalize())

    st.session_state["spectrograms"] = spectrograms

    progress_bar.progress(100)
    status_text.success("✅ Séparation et spectrogrammes terminés !")

    return output_files


# --- UI ---
model_choice = st.selectbox(
    "Modèle Demucs",
    options=["htdemucs", "htdemucs_ft", "mdx_extra", "mdx_extra_q"],
    help="htdemucs = rapide. htdemucs_ft / mdx_extra = meilleure qualité, plus lent.",
)

uploaded_file = st.file_uploader(
    "Dépose un fichier audio",
    type=["mp3", "wav", "flac", "ogg", "m4a", "mp4"],
)

if uploaded_file is not None:
    st.subheader("Fichier d'entrée")

    st.audio(uploaded_file)

    # Spectrogramme de l'input (expander, visible seulement si généré)
    if "spectrograms" in st.session_state and "Original" in st.session_state["spectrograms"]:
        with st.expander("📊 Voir le spectrogramme — Original"):
            st.image(st.session_state["spectrograms"]["Original"], use_container_width=True)

    if st.button("🚀 Lancer la séparation"):
        with CacheManager() as cm:
            uploaded_file.seek(0)
            input_path = cm.write_uploaded_file(uploaded_file)
            output_dir = cm.create_output_dir("output")

            try:
                output_files = separate_audio(input_path, output_dir, model_choice)

                archive_path = cm.create_zip_archive(output_dir, "stems_archive")
                with open(archive_path, "rb") as f:
                    zip_bytes = f.read()

                st.session_state["output_files"] = output_files
                st.session_state["zip_bytes"] = zip_bytes

            except Exception as e:
                st.error(f"Erreur pendant la séparation : {e}")
                st.session_state["output_files"] = []
                st.session_state["zip_bytes"] = None

        st.rerun()

    if "output_files" in st.session_state and st.session_state["output_files"]:
        st.subheader("🎼 Résultats")
        cols = st.columns(2)

        for i, (stem_name, audio_bytes, filename) in enumerate(st.session_state["output_files"]):
            with cols[i % 2]:
                st.markdown(f"### {stem_name.capitalize()}")
                st.audio(audio_bytes, format="audio/wav")

                st.download_button(
                    label="⬇️ Télécharger",
                    data=audio_bytes,
                    file_name=filename,
                    mime="audio/wav",
                    key=f"dl_{stem_name}"
                )

                spec_key = stem_name.capitalize()
                if (
                    "spectrograms" in st.session_state
                    and spec_key in st.session_state["spectrograms"]
                ):
                    with st.expander(f"📊 Voir le spectrogramme — {spec_key}"):
                        st.image(
                            st.session_state["spectrograms"][spec_key],
                            use_container_width=True,
                        )

        if st.session_state.get("zip_bytes"):
            st.write("---")
            st.download_button(
                label="📦 Télécharger tous les stems (.zip)",
                data=st.session_state["zip_bytes"],
                file_name="stems_demucs.zip",
                mime="application/zip",
            )