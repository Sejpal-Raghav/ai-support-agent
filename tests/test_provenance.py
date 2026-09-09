from pathlib import Path

from support_agent.hashing import canonical_sha256
from support_agent.provenance import build_artifact_manifest, verify_manifest


def test_canonical_hash_is_key_order_independent() -> None:
    assert canonical_sha256({"a": 1, "b": 2}) == canonical_sha256({"b": 2, "a": 1})


def test_manifest_verifies_outputs(tmp_path: Path) -> None:
    output = tmp_path / "output.txt"
    output.write_text("verified", encoding="utf-8")
    manifest = build_artifact_manifest(
        inputs=(), outputs=(output,), command=("phase0",)
    )
    assert verify_manifest(manifest, tmp_path).valid


def test_manifest_reports_changed_file(tmp_path: Path) -> None:
    output = tmp_path / "output.txt"
    output.write_text("before", encoding="utf-8")
    manifest = build_artifact_manifest(inputs=(), outputs=(output,), command=("phase0",))
    output.write_text("after", encoding="utf-8")
    verification = verify_manifest(manifest, tmp_path)
    assert not verification.valid
    assert verification.mismatched_paths == (output.as_posix(),)
