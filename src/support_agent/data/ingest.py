"""Streaming ingestion for the Kaggle Customer Support on Twitter CSV."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field

from support_agent.schemas import RawMessage

REQUIRED_SOURCE_COLUMNS = frozenset(
    {
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id",
    }
)
INGEST_ARROW_SCHEMA = pa.schema(
    [
        pa.field("tweet_id", pa.string(), nullable=False),
        pa.field("author_id", pa.string(), nullable=False),
        pa.field("inbound", pa.bool_(), nullable=False),
        pa.field("created_at", pa.string(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
        pa.field("response_tweet_ids", pa.list_(pa.string()), nullable=False),
        pa.field("in_response_to_tweet_id", pa.string(), nullable=True),
    ]
)


class IngestStats(BaseModel):
    """Stable ingestion counts for source validation and provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_path: str
    output_path: str
    rows_read: int = Field(ge=0)
    rows_written: int = Field(ge=0)
    rows_rejected: int = Field(ge=0)


def validate_source_columns(csv_path: Path, required: frozenset[str]) -> None:
    """Reject a CSV that does not contain the documented source columns."""
    header = pd.read_csv(csv_path, nrows=0, dtype="string")
    missing = sorted(required.difference(header.columns))
    if missing:
        raise ValueError(f"Missing required source columns: {', '.join(missing)}")


def _parse_optional_id(value: object) -> str | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    candidate = str(value).strip()
    return candidate or None


def _parse_response_ids(value: object) -> tuple[str, ...]:
    raw = _parse_optional_id(value)
    if raw is None:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _parse_inbound(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"inbound must be true or false, got {value!r}")


def parse_raw_row(row: Mapping[str, Any]) -> RawMessage:
    """Convert one untrusted CSV row into the immutable raw-message schema."""
    created_at = pd.to_datetime(row["created_at"], utc=True, errors="raise").to_pydatetime()
    tweet_id = _parse_optional_id(row["tweet_id"])
    author_id = _parse_optional_id(row["author_id"])
    text = _parse_optional_id(row["text"])
    if tweet_id is None or author_id is None or text is None:
        raise ValueError("tweet_id, author_id, and text must be non-empty")
    return RawMessage(
        tweet_id=tweet_id,
        author_id=author_id,
        inbound=_parse_inbound(row["inbound"]),
        created_at=created_at,
        text=text,
        response_tweet_ids=_parse_response_ids(row["response_tweet_id"]),
        in_response_to_tweet_id=_parse_optional_id(row["in_response_to_tweet_id"]),
    )


def _normalized_frame(chunk: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    records: list[dict[str, object]] = []
    rejected = 0
    for untyped_row in chunk.to_dict(orient="records"):
        row = {str(key): value for key, value in untyped_row.items()}
        try:
            records.append(parse_raw_row(row).model_dump(mode="json"))
        except (TypeError, ValueError):
            rejected += 1
    return pd.DataFrame.from_records(records), rejected


def _to_arrow_table(normalized: pd.DataFrame) -> pa.Table:
    """Materialize a fixed schema so all-null chunks cannot alter Parquet types."""
    return pa.Table.from_pydict(
        {
            "tweet_id": normalized["tweet_id"].tolist(),
            "author_id": normalized["author_id"].tolist(),
            "inbound": normalized["inbound"].tolist(),
            "created_at": normalized["created_at"].tolist(),
            "text": normalized["text"].tolist(),
            "response_tweet_ids": normalized["response_tweet_ids"].tolist(),
            "in_response_to_tweet_id": normalized["in_response_to_tweet_id"].tolist(),
        },
        schema=INGEST_ARROW_SCHEMA,
    )


def ingest_csv(csv_path: Path, output_path: Path, chunksize: int = 250_000) -> IngestStats:
    """Validate and atomically convert source CSV rows to a normalized Parquet table."""
    validate_source_columns(csv_path, REQUIRED_SOURCE_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.part")
    temporary_path.unlink(missing_ok=True)
    rows_read = 0
    rows_written = 0
    rows_rejected = 0
    parquet_writer: Any = None
    try:
        source_chunks = pd.read_csv(
            csv_path,
            dtype="string",
            chunksize=chunksize,
            keep_default_na=True,
        )
        for chunk in source_chunks:
            rows_read += len(chunk)
            normalized, rejected = _normalized_frame(chunk)
            rows_rejected += rejected
            if normalized.empty:
                continue
            table = _to_arrow_table(normalized)
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(temporary_path, table.schema)
            parquet_writer.write_table(table)
            rows_written += len(normalized)
    finally:
        if parquet_writer is not None:
            parquet_writer.close()
    if rows_written == 0:
        temporary_path.unlink(missing_ok=True)
        raise ValueError("Ingestion produced zero valid rows")
    temporary_path.replace(output_path)
    return IngestStats(
        source_path=csv_path.as_posix(),
        output_path=output_path.as_posix(),
        rows_read=rows_read,
        rows_written=rows_written,
        rows_rejected=rows_rejected,
    )


def iter_ingested_messages(parquet_path: Path) -> Iterator[RawMessage]:
    """Yield validated normalized messages from a Phase-1 Parquet artifact."""
    frame = pd.read_parquet(parquet_path)
    for row in frame.to_dict(orient="records"):
        row["response_tweet_ids"] = tuple(row["response_tweet_ids"])
        yield RawMessage.model_validate(row)
