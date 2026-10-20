from __future__ import annotations

import json
import wave
from io import BytesIO
from types import SimpleNamespace

import pytest

from support_agent.voice import transcription
from support_agent.voice.transcription import (
    MAX_AUDIO_BYTES,
    AudioValidationError,
    convert_to_pcm_wav,
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


def wav_bytes(*, seconds: int = 1, sample: int = 1000) -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        frame = sample.to_bytes(2, "little", signed=True)
        wav_file.writeframes(frame * 16000 * seconds)
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


def test_transcribe_pcm_wav_rejects_silence_before_recognition() -> None:
    recognizer = FakeRecognizer(final_text="hallucinated speech")
    with pytest.raises(AudioValidationError, match="No speech") as error:
        transcribe_pcm_wav(
            wav_bytes(sample=0), recognizer=recognizer, asr_model_version="test-vosk"
        )
    assert error.value.code == "empty_transcript"
    assert not recognizer.chunks


def test_voice_evaluation_retains_unrecognized_speech_as_empty_text() -> None:
    result = transcribe_pcm_wav(
        wav_bytes(), recognizer=FakeRecognizer(final_text=""), asr_model_version="fixture",
        allow_empty_transcript=True,
    )
    assert result.transcript == ""
    assert result.warnings == ("empty_transcript",)


def test_transcribe_pcm_wav_rejects_long_audio() -> None:
    with pytest.raises(AudioValidationError, match="30 seconds") as error:
        transcribe_pcm_wav(
            wav_bytes(seconds=31),
            recognizer=FakeRecognizer(final_text="hello"),
            asr_model_version="test-vosk",
        )
    assert error.value.code == "audio_too_long"


def test_duration_uses_pcm_size_when_streamed_wav_header_is_unbounded() -> None:
    audio = bytearray(wav_bytes())
    audio[40:44] = (0xFFFFFFFF).to_bytes(4, "little")
    result = transcribe_pcm_wav(
        bytes(audio), recognizer=FakeRecognizer(final_text="hello"), asr_model_version="test-vosk"
    )
    assert result.duration_ms == 1000


def test_conversion_uses_bundled_ffmpeg_when_system_binary_is_absent(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["input"] = kwargs["input"]
        return SimpleNamespace(returncode=0, stdout=b"converted", stderr=b"")

    monkeypatch.setattr(transcription.shutil, "which", lambda _: None)
    monkeypatch.setattr(transcription.subprocess, "run", fake_run)
    monkeypatch.setattr("imageio_ffmpeg.get_ffmpeg_exe", lambda: "bundled-ffmpeg")

    assert convert_to_pcm_wav(b"browser-audio") == b"converted"
    assert captured["command"][0] == "bundled-ffmpeg"
    assert captured["input"] == b"browser-audio"
