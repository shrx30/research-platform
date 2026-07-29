import asyncio
import tempfile

import edge_tts


VOICE = "en-US-AriaNeural"


async def _generate_audio(
    text: str,
    output_path: str,
) -> None:

    communicator = edge_tts.Communicate(
        text=text,
        voice=VOICE,
    )

    await communicator.save(
        output_path
    )


def text_to_speech(text: str) -> str:

    if not text or not text.strip():

        raise ValueError(
            "Cannot generate speech from empty text."
        )

    temp_file = tempfile.NamedTemporaryFile(
        suffix=".mp3",
        delete=False,
    )

    output_path = temp_file.name
    temp_file.close()

    asyncio.run(
        _generate_audio(
            text=text,
            output_path=output_path,
        )
    )

    return output_path