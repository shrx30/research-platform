from functools import lru_cache

from faster_whisper import WhisperModel


@lru_cache(maxsize=1)
def get_model():
    return WhisperModel(
        "base.en",
        device="cpu",
        compute_type="int8",
    )


def transcribe_audio(audio_path: str) -> str:
    model = get_model()

    segments, _ = model.transcribe(
        audio_path,
        language="en",
        beam_size=5,
        vad_filter=False,
    )

    transcript = " ".join(
        segment.text.strip()
        for segment in segments
        if segment.text.strip()
    ).strip()

    if not transcript:
        raise ValueError("No speech detected.")

    return transcript