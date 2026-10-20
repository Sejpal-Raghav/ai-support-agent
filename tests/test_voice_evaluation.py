import pytest

from support_agent.evaluation.voice import VoiceEvaluationRecord, evaluate_voice, word_errors


def record(index, reference="where is my order", transcript="where is my order"):
    return VoiceEvaluationRecord(
        recording_id=str(index),
        speaker_id=str(index % 5),
        noise="quiet" if index < 25 else "moderate" if index < 40 else "noisy",
        split="held_out",
        reference=reference,
        transcript=transcript,
        asr_model_version="fixture",
        duration_ms=1000,
        latency_ms=20,
    )


def test_word_errors_count_insertions_deletions_and_substitutions() -> None:
    assert word_errors("Where is my order?", "where is the order please") == (2, 4)
    assert word_errors("my order", "") == (2, 2)
    assert word_errors("I’m waiting!", "i'm waiting") == (0, 2)


def test_voice_wer_is_corpus_weighted_and_acceptance_is_explicit() -> None:
    result = evaluate_voice([record(0, "one", "wrong"), record(1, "one two three", "one two")])
    assert result["wer"] == 0.5
    assert result["accepted"] is False
    assert evaluate_voice([record(index) for index in range(50)])["accepted"] is True


def test_voice_evaluation_rejects_duplicate_recordings_and_empty_references() -> None:
    with pytest.raises(ValueError, match="unique"):
        evaluate_voice([record(0), record(0)])
    with pytest.raises(ValueError, match="English word"):
        word_errors("...", "hello")
