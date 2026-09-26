from __future__ import annotations

import json
import wave
from io import BytesIO

import pytest

from support_agent.voice.transcription import (
    MAX_AUDIO_BYTES,
    AudioValidationError,
    transcribe_pcm_wav,
    validate_audio_payload,
)


class FakeRecognizer:
    def __init__(self, *, final_text: str) -> None:
        self.final_text = final_text
        self.chunks: list[bytes] = []

    def AcceptWaveform(self, data: bytes) -> bool:  # noqa: N802
        self.chunks.append(data)
        return False

    def Result(self) -> str:  # noqa: N802
        return json.dumps({"text": ""})

    def FinalResult(self) -> str:  # noqa: N802
        return json.dumps({"text": self.final_text})


def wav_bytes(*, seconds: int = 1) -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 16000 * seconds)
    return output.getvalue()


@pytest.mark.parametrize(
    ("audio", "content_type", "code"),
    [
        (b"", "audio/webm", "empty_audio"),
        (b"a" * (MAX_AUDIO_BYTES + 1), "audio/webm", "audio_too_large"),
        (b"audio", "text/plain", "unsupported_audio_type"),
    ],
    ids=("empty", "oversized", "unsupported-type"),
)
def test_validate_audio_payload_rejects_invalid_inputs(
    audio: bytes, content_type: str, code: str
) -> None:
    with pytest.raises(AudioValidationError, match=".") as error:
        validate_audio_payload(audio, content_type)
    assert error.value.code == code


def test_transcribe_pcm_wav_returns_transcript_and_duration() -> None:
    recognizer = FakeRecognizer(final_text="need help with my order")
    result = transcribe_pcm_wav(
        wav_bytes(), recognizer=recognizer, asr_model_version="test-vosk"
    )
    assert result.transcript == "need help with my order"
    assert result.duration_ms == 1000
    assert result.asr_model_version == "test-vosk"
    assert recognizer.chunks


def test_transcribe_pcm_wav_rejects_empty_transcript() -> None:
    with pytest.raises(AudioValidationError, match="No speech") as error:
        transcribe_pcm_wav(
            wav_bytes(), recognizer=FakeRecognizer(final_text=""), asr_model_version="test-vosk"
        )
    assert error.value.code == "empty_transcript"


def test_transcribe_pcm_wav_rejects_long_audio() -> None:
    with pytest.raises(AudioValidationError, match="30 seconds") as error:
        transcribe_pcm_wav(
            wav_bytes(seconds=31),
            recognizer=FakeRecognizer(final_text="hello"),
            asr_model_version="test-vosk",
        )
    assert error.value.code == "audio_too_long"
