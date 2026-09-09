"""Deterministic cohort sampling and chronological leakage-safe splits."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from support_agent.hashing import canonical_sha256
from support_agent.schemas import ReconstructedThread


class ThreadSplits(BaseModel):
    """Frozen chronological train/validation/test partition by full thread."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    train: tuple[ReconstructedThread, ...]
    validation: tuple[ReconstructedThread, ...]
    test: tuple[ReconstructedThread, ...]


class LeakageAudit(BaseModel):
    """Explicit checks required before taxonomy, retrieval, or annotation work."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    train_thread_count: int = Field(ge=0)
    validation_thread_count: int = Field(ge=0)
    test_thread_count: int = Field(ge=0)
    train_validation_overlap: tuple[str, ...]
    train_test_overlap: tuple[str, ...]
    validation_test_overlap: tuple[str, ...]
    train_end: datetime | None
    validation_start: datetime | None
    validation_end: datetime | None
    test_start: datetime | None


def select_cohort(
    threads: Sequence[ReconstructedThread], size: int, seed: int
) -> list[ReconstructedThread]:
    """Take the fixed cohort by hash rank, independent of timestamp or label."""
    if len(threads) < size:
        raise ValueError(f"Need at least {size} threads after deduplication; found {len(threads)}")
    unique_ids = {thread.thread_id for thread in threads}
    if len(unique_ids) != len(threads):
        raise ValueError("Cohort source contains duplicate thread IDs")
    return sorted(
        threads,
        key=lambda thread: canonical_sha256(f"cohort:{seed}:{thread.thread_id}"),
    )[:size]


def chronological_split(
    threads: Sequence[ReconstructedThread], ratios: tuple[float, float, float]
) -> ThreadSplits:
    """Sort selected threads by time and split without splitting any component."""
    if ratios != (0.70, 0.15, 0.15):
        raise ValueError("ratios must be exactly (0.70, 0.15, 0.15)")
    ordered = tuple(sorted(threads, key=lambda item: (item.started_at, item.thread_id)))
    train_count = int(len(ordered) * ratios[0])
    validation_count = int(len(ordered) * ratios[1])
    return ThreadSplits(
        train=ordered[:train_count],
        validation=ordered[train_count : train_count + validation_count],
        test=ordered[train_count + validation_count :],
    )


def assert_no_split_leakage(splits: ThreadSplits) -> LeakageAudit:
    """Raise on ID or chronological leakage and otherwise return auditable bounds."""
    train_ids = {thread.thread_id for thread in splits.train}
    validation_ids = {thread.thread_id for thread in splits.validation}
    test_ids = {thread.thread_id for thread in splits.test}
    train_validation = tuple(sorted(train_ids.intersection(validation_ids)))
    train_test = tuple(sorted(train_ids.intersection(test_ids)))
    validation_test = tuple(sorted(validation_ids.intersection(test_ids)))
    if train_validation or train_test or validation_test:
        raise ValueError("Thread ID overlap detected across splits")
    train_end = splits.train[-1].started_at if splits.train else None
    validation_start = splits.validation[0].started_at if splits.validation else None
    validation_end = splits.validation[-1].started_at if splits.validation else None
    test_start = splits.test[0].started_at if splits.test else None
    if train_end is not None and validation_start is not None and train_end > validation_start:
        raise ValueError("Train split is not chronologically before validation")
    if validation_end is not None and test_start is not None and validation_end > test_start:
        raise ValueError("Validation split is not chronologically before test")
    return LeakageAudit(
        train_thread_count=len(splits.train),
        validation_thread_count=len(splits.validation),
        test_thread_count=len(splits.test),
        train_validation_overlap=train_validation,
        train_test_overlap=train_test,
        validation_test_overlap=validation_test,
        train_end=train_end,
        validation_start=validation_start,
        validation_end=validation_end,
        test_start=test_start,
    )
