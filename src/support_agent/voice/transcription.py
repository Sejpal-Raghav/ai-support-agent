"""Bounded, non-persistent browser-audio transcription with Vosk."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import wave
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Protocol

import numpy as np

MAX_AUDIO_BYTES = 10 * 1024 * 1024
MAX_AUDIO_SECONDS = 30
ENCODING_DURATION_TOLERANCE_MS = 500
SUPPORTED_CONTENT_TYPES = frozenset(
    {
        "audio/webm",
        "audio/ogg",
        "audio/wav",
        "audio/x-wav",
    }
)


class AudioValidationError(ValueError):
    """A safe client-facing audio rejection with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class TranscriptionResult:
    """Recognized text and bounded processing metadata."""

    transcript: str
    duration_ms: int
    asr_model_version: str
    warnings: tuple[str, ...]
    latency_ms: int


class _Recognizer(Protocol):
    def AcceptWaveform(self, data: bytes) -> bool:  # noqa: N802
        """Accept a PCM chunk."""

    def Result(self) -> str:  # noqa: N802
        """Return a JSON partial result."""

    def FinalResult(self) -> str:  # noqa: N802
        """Return a JSON final result."""


def validate_audio_payload(audio: bytes, content_type: str | None) -> None:
    """Reject absent, oversized, or unsupported browser audio before conversion."""
    normalized_type = (content_type or "").split(";", maxsplit=1)[0].strip().lower()
    if not audio:
        raise AudioValidationError("empty_audio", "No audio was received.")
    if len(audio) > MAX_AUDIO_BYTES:
        raise AudioValidationError(
            "audio_too_large", f"Audio must be at most {MAX_AUDIO_BYTES // (1024 * 1024)} MB."
        )
    if normalized_type not in SUPPORTED_CONTENT_TYPES:
        raise AudioValidationError(
            "unsupported_audio_type", "Use WebM, OGG, or WAV audio recorded from the browser."
        )


def convert_to_pcm_wav(audio: bytes, *, ffmpeg_binary: str = "ffmpeg") -> bytes:
    """Convert untrusted audio in-memory to mono 16-kHz signed 16-bit WAV."""
    resolved_ffmpeg = ffmpeg_binary
    if ffmpeg_binary == "ffmpeg" and shutil.which(ffmpeg_binary) is None:
        try:
            import imageio_ffmpeg
        except ImportError as error:
            raise RuntimeError("FFmpeg is required for browser-audio transcription.") from error
        resolved_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = (
        resolved_ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        "pipe:1",
    )
    try:
        completed = subprocess.run(
            command,
            input=audio,
            capture_output=True,
            check=False,
            timeout=35,
        )
    except FileNotFoundError as error:
        raise RuntimeError("FFmpeg is required for browser-audio transcription.") from error
    except subprocess.TimeoutExpired as error:
        raise AudioValidationError(
            "audio_conversion_timeout", "Audio conversion took too long."
        ) from error
    if completed.returncode != 0 or not completed.stdout:
        raise AudioValidationError("invalid_audio", "The recording could not be decoded.")
    return completed.stdout


def _wav_duration_and_frames(wav_bytes: bytes) -> tuple[int, bytes]:
    try:
        with wave.open(BytesIO(wav_bytes), "rb") as wav_file:
            if wav_file.getnchannels() != 1 or wav_file.getframerate() != 16000:
                raise AudioValidationError(
                    "invalid_pcm_audio", "Audio conversion produced invalid PCM."
                )
            if wav_file.getsampwidth() != 2:
                raise AudioValidationError(
                    "invalid_pcm_audio", "Audio conversion produced invalid PCM."
                )
            frame_count = wav_file.getnframes()
            pcm_frames = wav_file.readframes(frame_count)
            bytes_per_second = (
                wav_file.getframerate() * wav_file.getnchannels() * wav_file.getsampwidth()
            )
            duration_ms = round(len(pcm_frames) * 1000 / bytes_per_second)
            if duration_ms > MAX_AUDIO_SECONDS * 1000 + ENCODING_DURATION_TOLERANCE_MS:
                raise AudioValidationError(
                    "audio_too_long", f"Audio must be at most {MAX_AUDIO_SECONDS} seconds."
                )
            return duration_ms, pcm_frames
    except wave.Error as error:
        raise AudioValidationError(
            "invalid_pcm_audio", "Audio conversion produced invalid PCM."
        ) from error


def transcribe_pcm_wav(
    wav_bytes: bytes,
    *,
    recognizer: _Recognizer,
    asr_model_version: str,
) -> TranscriptionResult:
    """Transcribe a validated PCM WAV stream without retaining audio or transcript state."""
    started = time.perf_counter()
    duration_ms, pcm_frames = _wav_duration_and_frames(wav_bytes)
    samples = np.frombuffer(pcm_frames, dtype="<i2").astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    if rms < 0.002:
        raise AudioValidationError("empty_transcript", "No speech was detected. Please try again.")
    text_parts: list[str] = []
    for offset in range(0, len(pcm_frames), 4000):
        chunk = pcm_frames[offset : offset + 4000]
        if recognizer.AcceptWaveform(chunk):
            text = str(json.loads(recognizer.Result()).get("text", "")).strip()
            if text:
                text_parts.append(text)
    final_text = str(json.loads(recognizer.FinalResult()).get("text", "")).strip()
    if final_text:
        text_parts.append(final_text)
    transcript = " ".join(text_parts).strip()
    if not transcript:
        raise AudioValidationError("empty_transcript", "No speech was detected. Please try again.")
    return TranscriptionResult(
        transcript=transcript,
        duration_ms=duration_ms,
        asr_model_version=asr_model_version,
        warnings=(),
        latency_ms=round((time.perf_counter() - started) * 1000),
    )


def transcribe_webm_audio(
    audio: bytes,
    *,
    content_type: str | None,
    model_path: Path,
    asr_model_version: str = "vosk-model-en-us-0.22-lgraph",
    ffmpeg_binary: str = "ffmpeg",
) -> TranscriptionResult:
    """Convert and transcribe one browser recording using a local Vosk model."""
    validate_audio_payload(audio, content_type)
    if not model_path.is_dir():
        raise RuntimeError("The configured Vosk model directory is unavailable.")
    wav_bytes = convert_to_pcm_wav(audio, ffmpeg_binary=ffmpeg_binary)
    try:
        from vosk import KaldiRecognizer
    except ImportError as error:
        raise RuntimeError(
            "Install the api dependency group to enable Vosk transcription."
        ) from error
    model = load_vosk_model(str(model_path))
    recognizer = KaldiRecognizer(model, 16000)
    return transcribe_pcm_wav(
        wav_bytes,
        recognizer=recognizer,
        asr_model_version=asr_model_version,
    )


@lru_cache(maxsize=2)
def load_vosk_model(model_path: str) -> object:
    """Load each immutable Vosk model once per server process."""
    from vosk import Model

    return Model(model_path)
