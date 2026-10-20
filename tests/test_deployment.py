import base64
import json
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

from support_agent.deployment import (
    BUNDLE_NAME,
    container_memory_limit,
    package_evidence,
    unpack_evidence,
    validate_evidence,
    validate_memory_budget,
)
from support_agent.hashing import file_sha256


@pytest.fixture
def evidence(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    corpus = source / "demo_cases.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "case_id": "case-1",
                "source_thread_id": "thread-1",
                "split": "train",
                "customer_text": "Where is my package?",
                "reply_text": "Check carrier tracking.",
                "intent_id": "delivery_tracking",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    corpus.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "corpus_sha256": file_sha256(corpus),
                "case_count": 1,
                "thread_partitions": {"thread-1": "train", "held-out-thread": "test"},
            }
        ),
        encoding="utf-8",
    )
    return corpus


def test_bundle_roundtrip_preserves_exact_evidence_and_removes_unused_membership(
    evidence, tmp_path
):
    bundle = tmp_path / BUNDLE_NAME
    assert package_evidence(evidence, bundle) == 1
    # Secret editors may wrap ASCII text and change its line endings.
    bundle.write_bytes(
        b"\r\n".join(
            bundle.read_bytes()[index : index + 72] for index in range(0, bundle.stat().st_size, 72)
        )
    )
    unpacked = unpack_evidence(bundle, tmp_path / "deployed")
    assert unpacked.read_bytes() == evidence.read_bytes()
    _, manifest = validate_evidence(unpacked)
    assert manifest["thread_partitions"] == {"thread-1": "train"}


def test_bundle_rejects_unexpected_archive_members(tmp_path):
    data = BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("../outside.txt", "untrusted")
        archive.writestr("demo_cases.jsonl", "{}")
    bundle = tmp_path / BUNDLE_NAME
    bundle.write_bytes(base64.b64encode(data.getvalue()))
    with pytest.raises(ValueError, match="exactly"):
        unpack_evidence(bundle, tmp_path / "deployed")
    assert not (tmp_path / "outside.txt").exists()


def test_bundle_rejects_incomplete_paste(tmp_path):
    bundle = tmp_path / BUNDLE_NAME
    bundle.write_text("not a complete bundle!", encoding="ascii")
    with pytest.raises(ValueError, match="complete exported"):
        unpack_evidence(bundle, tmp_path / "deployed")


def test_export_rejects_a_mismatched_pair(evidence, tmp_path):
    evidence.write_text(
        evidence.read_text(encoding="utf-8").replace("tracking", "password"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="does not match"):
        package_evidence(evidence, tmp_path / BUNDLE_NAME)


def test_export_rejects_held_out_evidence(evidence, tmp_path):
    path = evidence.with_suffix(".manifest.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["thread_partitions"]["thread-1"] = "test"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="non-train"):
        package_evidence(evidence, tmp_path / BUNDLE_NAME)


def test_memory_preflight_explains_a_512_mib_instance(tmp_path):
    (tmp_path / "memory.max").write_text(str(512 * 1024**2), encoding="ascii")
    with pytest.raises(RuntimeError, match="requires at least 1 GiB"):
        validate_memory_budget(container_memory_limit(tmp_path))
    validate_memory_budget(1024**3)


def test_memory_preflight_accepts_unlimited_or_non_container_hosts(tmp_path):
    assert container_memory_limit(tmp_path) is None
    (tmp_path / "memory.max").write_text("max", encoding="ascii")
    assert container_memory_limit(tmp_path) is None
    validate_memory_budget(None)


def test_compact_backend_accepts_512_mib_but_rejects_less():
    validate_memory_budget(512 * 1024**2, backend="onnx")
    with pytest.raises(RuntimeError, match="at least 512 MiB"):
        validate_memory_budget(256 * 1024**2, backend="onnx")


def test_memory_preflight_reads_cgroup_v1(tmp_path):
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory/memory.limit_in_bytes").write_text(str(2 * 1024**3), encoding="ascii")
    assert container_memory_limit(tmp_path) == 2 * 1024**3
