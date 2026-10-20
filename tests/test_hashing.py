import hashlib
from pathlib import Path

import pytest

from support_agent.hashing import verify_jsonl_sha256


@pytest.mark.parametrize("source_newline", ["\n", "\r\n"])
@pytest.mark.parametrize("transport_newline", ["\n", "\r\n"])
@pytest.mark.parametrize("final_newline", [False, True])
def test_jsonl_hash_tolerates_only_transport_line_endings(
    tmp_path: Path, source_newline: str, transport_newline: str, final_newline: bool
) -> None:
    lines = ['{"case_id":"one","reply_text":"Hello world"}', '{"case_id":"two"}']
    original = (source_newline.join(lines) + source_newline).encode()
    expected = hashlib.sha256(original).hexdigest()
    path = tmp_path / "cases.jsonl"
    transported = transport_newline.join(lines) + (transport_newline if final_newline else "")
    path.write_bytes(transported.encode())
    assert verify_jsonl_sha256(path, expected) == expected


def test_jsonl_hash_still_rejects_changed_evidence(tmp_path: Path) -> None:
    original = b'{"reply_text":"Please check tracking"}\r\n'
    path = tmp_path / "cases.jsonl"
    path.write_bytes(b'{"reply_text":"Please share passwords"}\n')
    with pytest.raises(ValueError, match="Replace both Render Secret Files"):
        verify_jsonl_sha256(path, hashlib.sha256(original).hexdigest())
