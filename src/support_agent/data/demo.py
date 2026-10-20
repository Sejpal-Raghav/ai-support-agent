"""Conversation-level partitioning for the local demonstration corpus."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import pandas as pd


def partition_messages(frame: pd.DataFrame, brand_id: str) -> dict[str, tuple[str, str]]:
    """Split complete connected conversations chronologically, using both reference directions."""
    parents = {str(value): str(value) for value in frame.tweet_id}

    def root(value: str) -> str:
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    for row in frame.itertuples(index=False):
        references = list(cast(Sequence[str], row.response_tweet_ids))
        if row.in_response_to_tweet_id is not None:
            references.append(str(row.in_response_to_tweet_id))
        for reference in references:
            if str(reference) in parents:
                left, right = root(str(row.tweet_id)), root(str(reference))
                parents[max(left, right)] = min(left, right)
    groups = {value: root(value) for value in parents}
    brand_groups = {
        groups[str(row.tweet_id)]
        for row in frame.itertuples(index=False)
        if not row.inbound and row.author_id == brand_id
    }
    starts: dict[str, str] = {}
    for row in frame.itertuples(index=False):
        group = groups[str(row.tweet_id)]
        if group in brand_groups:
            starts[group] = min(starts.get(group, str(row.created_at)), str(row.created_at))
    ordered = sorted(brand_groups, key=lambda group: (starts[group], group))
    train_end, validation_end = int(len(ordered) * 0.70), int(len(ordered) * 0.85)
    partitions = {
        group: "train" if index < train_end else "validation" if index < validation_end else "test"
        for index, group in enumerate(ordered)
    }
    return {
        value: (group, partitions[group]) for value, group in groups.items() if group in partitions
    }
