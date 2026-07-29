import hashlib
import os
import tempfile

import streamlit as st

from app.graph.workflow import graph
from app.voice.asr import transcribe_audio
from app.voice.tts import text_to_speech


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Research Platform",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 AI Research Platform")

st.caption(
    "Multi-agent research across the web, GitHub, academic papers, "
    "and long-term memory — with voice input and output."
)


# =========================================================
# SESSION STATE
# =========================================================

DEFAULT_STATE = {
    "voice_query": "",
    "last_report": None,
    "report_audio": None,
    "last_audio_hash": None,
    "last_query": "",
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# RESEARCH QUESTION
# =========================================================

st.subheader("Research Question")

typed_query = st.text_input(
    "Type your research question",
    placeholder="e.g. How do patches work in Vision Transformers?",
)

st.markdown("**OR**")


# =========================================================
# MICROPHONE
# =========================================================

audio = st.audio_input(
    "🎙️ Record your research question"
)


# =========================================================
# SPEECH TO TEXT
# =========================================================

if audio is not None:

    audio_bytes = audio.getvalue()

    # Prevent Streamlit reruns from transcribing
    # the same recording repeatedly.
    audio_hash = hashlib.sha256(
        audio_bytes
    ).hexdigest()

    if audio_hash != st.session_state.last_audio_hash:

        temp_path = None

        try:

            # st.audio_input returns WAV audio.
            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
            ) as temp:

                temp.write(audio_bytes)
                temp_path = temp.name

            with st.spinner(
                "🎧 Transcribing..."
            ):

                transcript = transcribe_audio(
                    temp_path
                )

            transcript = transcript.strip()

            if not transcript:

                st.warning(
                    "No speech was detected."
                )

            else:

                st.session_state.voice_query = (
                    transcript
                )

                st.session_state.last_audio_hash = (
                    audio_hash
                )

                # Previous research belongs to the
                # previous question.
                st.session_state.last_report = None
                st.session_state.report_audio = None

        except Exception as exc:

            st.error(
                f"Transcription failed: {exc}"
            )

            # Useful while developing.
            st.exception(exc)

        finally:

            if (
                temp_path
                and os.path.exists(temp_path)
            ):
                os.remove(temp_path)


# =========================================================
# TRANSCRIPT
# =========================================================

if st.session_state.voice_query:

    st.write("**Voice Transcript**")

    st.info(
        st.session_state.voice_query
    )


# =========================================================
# QUERY SELECTION
# =========================================================

# Typed question has priority when both exist.

if typed_query.strip():

    query = typed_query.strip()

else:

    query = st.session_state.voice_query.strip()


# =========================================================
# BUTTONS
# =========================================================

research_col, clear_col = st.columns(
    [4, 1]
)


with research_col:

    research_clicked = st.button(
        "🚀 Research",
        type="primary",
        use_container_width=True,
    )


with clear_col:

    clear_clicked = st.button(
        "🗑️ Clear",
        use_container_width=True,
    )


# =========================================================
# CLEAR
# =========================================================

if clear_clicked:

    st.session_state.voice_query = ""
    st.session_state.last_report = None
    st.session_state.report_audio = None
    st.session_state.last_audio_hash = None
    st.session_state.last_query = ""

    st.rerun()


# =========================================================
# RUN RESEARCH
# =========================================================

if research_clicked:

    if not query:

        st.warning(
            "Type a research question or record "
            "your question first."
        )

    else:

        st.session_state.last_report = None
        st.session_state.report_audio = None
        st.session_state.last_query = query

        try:

            with st.spinner(
                "🔎 Research agents are working..."
            ):

                result = graph.invoke(
                    {
                        "query": query,
                    }
                )

            # ---------------------------------------------
            # VALIDATE GRAPH RESULT
            # ---------------------------------------------

            if result is None:

                st.error(
                    "The research graph returned no result."
                )

            elif not isinstance(result, dict):

                st.error(
                    "The research graph returned an "
                    "unexpected result type."
                )

            else:

                report = result.get(
                    "report"
                )

                if not report:

                    st.error(
                        "Research completed, but no "
                        "report was generated."
                    )

                else:

                    st.session_state.last_report = (
                        report
                    )

                    st.success(
                        "Research completed."
                    )

        except Exception as exc:

            st.error(
                f"Research failed: {exc}"
            )

            # Keep this while developing.
            st.exception(exc)


# =========================================================
# REPORT
# =========================================================

if st.session_state.last_report:

    st.divider()

    st.subheader(
        "📄 Research Report"
    )

    if st.session_state.last_query:

        st.caption(
            f"Research question: "
            f"{st.session_state.last_query}"
        )

    st.markdown(
        st.session_state.last_report
    )


    # =====================================================
    # TTS
    # =====================================================

    st.divider()

    st.subheader(
        "🔊 Voice Response"
    )

    if st.session_state.report_audio is None:

        generate_voice = st.button(
            "🔊 Generate Voice Response",
            use_container_width=True,
        )

        if generate_voice:

            tts_path = None

            try:

                with st.spinner(
                    "Generating voice response..."
                ):

                    tts_path = text_to_speech(
                        st.session_state.last_report
                    )

                    if not tts_path:

                        raise RuntimeError(
                            "TTS returned no audio file."
                        )

                    if not os.path.exists(
                        tts_path
                    ):

                        raise FileNotFoundError(
                            "Generated TTS file "
                            "does not exist."
                        )

                    with open(
                        tts_path,
                        "rb",
                    ) as audio_file:

                        generated_audio = (
                            audio_file.read()
                        )

                    if not generated_audio:

                        raise RuntimeError(
                            "Generated audio is empty."
                        )

                    st.session_state.report_audio = (
                        generated_audio
                    )

                st.rerun()

            except Exception as exc:

                st.error(
                    f"TTS failed: {exc}"
                )

                st.exception(exc)

            finally:

                if (
                    tts_path
                    and os.path.exists(tts_path)
                ):
                    os.remove(tts_path)


# =========================================================
# AUDIO PLAYER
# =========================================================

if st.session_state.report_audio:

    st.audio(
        st.session_state.report_audio,
        format="audio/mp3",
    )

    if st.button(
        "🔄 Regenerate Voice",
        use_container_width=True,
    ):

        st.session_state.report_audio = None

        st.rerun()