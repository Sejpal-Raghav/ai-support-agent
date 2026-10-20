"""Verified, single-file transport for private inference evidence on hosted services."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

from support_agent.hashing import verify_jsonl_sha256

CORPUS_NAME = "demo_cases.jsonl"
MANIFEST_NAME = "demo_cases.manifest.json"
BUNDLE_NAME = "support_agent_bundle.b64"
MAX_BUNDLE_BYTES = 1_000_000
MAX_EXPANDED_BYTES = 8_000_000


def container_memory_limit(cgroup_root: Path = Path("/sys/fs/cgroup")) -> int | None:
    """Read cgroup v2/v1 container limits; a host without a limit returns None."""
    for path in (cgroup_root / "memory.max", cgroup_root / "memory/memory.limit_in_bytes"):
        if path.is_file():
            value = path.read_text(encoding="ascii").strip()
            if value.isdecimal() and 0 < int(value) < 2**60:
                return int(value)
    return None


def validate_memory_budget(limit: int | None, *, backend: str = "pytorch") -> None:
    """Explain an insufficient instance limit before the model loader is killed by the OS."""
    minimum = 512 * 1024**2 if backend == "onnx" else 1024**3
    if limit is not None and limit < minimum:
        raise RuntimeError(
            f"Container memory limit is {limit // 1024**2} MiB. "
            + (
                "The compact ONNX + small Vosk service requires at least 512 MiB."
                if backend == "onnx"
                else "The PyTorch service requires at least 1 GiB. "
                "Use the compact ONNX Docker image for a free 512 MiB instance."
            )
        )


def validate_evidence(cases_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify content, unique IDs, and train membership before loading any ML libraries."""
    manifest = json.loads(cases_path.with_suffix(".manifest.json").read_text(encoding="utf-8-sig"))
    verify_jsonl_sha256(cases_path, manifest["corpus_sha256"])
    records = [
        json.loads(line)
        for line in cases_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("The inference corpus is empty.")
    identifiers = [row["case_id"] for row in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("The inference corpus contains duplicate case IDs.")
    for row in records:
        if row["split"] != "train" or not row["source_thread_id"]:
            raise ValueError("The inference corpus must contain only traceable train cases.")
        if manifest["thread_partitions"].get(row["source_thread_id"]) != "train":
            raise ValueError("Inference evidence includes a non-train source thread.")
    if "case_count" in manifest and manifest["case_count"] != len(records):
        raise ValueError("The corpus record count does not match its manifest.")
    return records, manifest


def package_evidence(cases_path: Path, output_path: Path) -> int:
    """Create one ASCII secret file so JSON and its manifest cannot be pasted out of sync."""
    records, manifest = validate_evidence(cases_path)
    compact = dict(manifest)
    compact["thread_partitions"] = {row["source_thread_id"]: "train" for row in records}
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in (
            (CORPUS_NAME, cases_path.read_bytes()),
            (MANIFEST_NAME, json.dumps(compact, sort_keys=True).encode("utf-8")),
        ):
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
    encoded = base64.b64encode(output.getvalue()) + b"\n"
    if len(encoded) > MAX_BUNDLE_BYTES:
        raise ValueError("Inference bundle exceeds the secret-file transport limit.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(encoded)
    return len(records)


def unpack_evidence(bundle_path: Path, destination_root: Path) -> Path:
    """Read only the two allowlisted members and reject corrupt, oversized, or edited bundles."""
    if bundle_path.stat().st_size > MAX_BUNDLE_BYTES:
        raise ValueError("Inference bundle exceeds the secret-file transport limit.")
    encoded = b"".join(bundle_path.read_bytes().split())
    try:
        packed = base64.b64decode(encoded, validate=True)
        with zipfile.ZipFile(BytesIO(packed)) as archive:
            members = archive.infolist()
            if len(members) != 2 or {info.filename for info in members} != {
                CORPUS_NAME,
                MANIFEST_NAME,
            }:
                raise ValueError("Inference bundle must contain exactly the corpus and manifest.")
            if sum(info.file_size for info in members) > MAX_EXPANDED_BYTES:
                raise ValueError("Expanded inference bundle is too large.")
            contents = {info.filename: archive.read(info) for info in members}
    except (binascii.Error, zipfile.BadZipFile) as error:
        raise ValueError(
            "Invalid inference bundle. Paste the complete exported .b64 file."
        ) from error
    destination = destination_root / hashlib.sha256(packed).hexdigest()[:16]
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in contents.items():
        (destination / name).write_bytes(content)
    cases_path = destination / CORPUS_NAME
    validate_evidence(cases_path)
    return cases_path
