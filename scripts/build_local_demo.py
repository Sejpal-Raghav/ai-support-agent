"""Build a compact AmazonHelp evidence corpus from the local ingested subset."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import typer

from support_agent.data.demo import partition_messages
from support_agent.demo_runtime import (
    INTENT_SEEDS,
    is_relevant_historical_reply,
    is_safe_historical_reply,
    is_support_exchange,
    sanitize_historical_reply,
)
from support_agent.hashing import file_sha256
from support_agent.text import normalize_customer_text

app = typer.Typer(add_completion=False)

PATTERNS = {
    "delivery_tracking": re.compile(
        r"\b(deliver|delivery|track|package|parcel|shipment|arriv)\w*\b", re.I
    ),
    "return_refund": re.compile(r"\b(return|refund|money back|reimburse)\w*\b", re.I),
    "damaged_or_wrong_item": re.compile(
        r"\b(damag|broken|defect|wrong item|missing item)\w*\b", re.I
    ),
    "account_or_payment": re.compile(
        r"\b(account|login|payment|charge|charged|card|prime)\w*\b", re.I
    ),
    "order_change_or_cancel": re.compile(r"\b(cancel|change|modify|address)\w*\b", re.I),
    "product_or_stock": re.compile(
        r"\b(stock|available|product|item|size|colour|color)\w*\b", re.I
    ),
}


def intent_for(text: str) -> str | None:
    matches = [intent for intent, pattern in PATTERNS.items() if pattern.search(text)]
    return matches[0] if len(matches) == 1 else None


@app.command()
def main(
    parquet_path: Path = typer.Option(Path("data/interim/messages.parquet"), exists=True),
    output_path: Path = typer.Option(Path("artifacts/demo_cases.jsonl")),
    brand_id: str = typer.Option("AmazonHelp"),
    per_intent: int = typer.Option(150, min=20, max=500),
) -> None:
    frame = pd.read_parquet(parquet_path)
    partitions = partition_messages(frame, brand_id)
    held_out_texts = {
        normalize_customer_text(str(row.text))
        for row in frame.loc[frame.inbound].itertuples(index=False)
        if str(row.tweet_id) in partitions and partitions[str(row.tweet_id)][1] != "train"
    }
    inbound = {
        str(row.tweet_id): str(row.text)
        for row in frame.loc[frame.inbound, ["tweet_id", "text"]].itertuples(index=False)
    }
    selected: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen: set[str] = set()
    outbound = frame.loc[~frame.inbound & (frame.author_id == brand_id)]
    for row in outbound.sort_values("tweet_id").itertuples(index=False):
        parent_id = row.in_response_to_tweet_id
        customer_text = inbound.get(str(parent_id)) if parent_id is not None else None
        if not customer_text:
            continue
        thread_id, split = partitions[str(row.tweet_id)]
        normalized = normalize_customer_text(customer_text)
        if split != "train" or normalized in held_out_texts or normalized in seen:
            continue
        intent = intent_for(customer_text)
        if intent is None or len(selected[intent]) >= per_intent:
            continue
        raw_reply = str(row.text)
        if not is_support_exchange(customer_text, raw_reply) or not is_safe_historical_reply(
            raw_reply
        ):
            continue
        reply = sanitize_historical_reply(raw_reply)
        if not is_relevant_historical_reply(reply, intent, customer_text):
            continue
        if len(reply) < 20 or reply.rstrip().endswith((":", "here", "at")):
            continue
        selected[intent].append(
            {
                "case_id": f"{brand_id}:{row.tweet_id}",
                "customer_text": customer_text,
                "reply_text": reply,
                "intent_id": intent,
                "source_thread_id": thread_id,
                "split": "train",
            }
        )
        seen.add(normalized)
    missing = sorted(set(INTENT_SEEDS) - {intent for intent, rows in selected.items() if rows})
    if missing:
        raise ValueError(f"no historical cases found for intents: {missing}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for intent in INTENT_SEEDS:
            for record in selected[intent]:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    groups = {group: split for group, split in partitions.values()}
    manifest = {
        "schema_version": "2",
        "source_sha256": file_sha256(parquet_path),
        "corpus_sha256": file_sha256(output_path),
        "split_method": "connected_conversations_chronological_70_15_15",
        "thread_partitions": groups,
        "case_count": sum(map(len, selected.values())),
        "intent_counts": {intent: len(rows) for intent, rows in selected.items()},
        "cross_split_exact_duplicates_excluded": True,
        "near_duplicate_review": "pending",
        "labels": "heuristic_demo_only",
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    typer.echo(f"Wrote {sum(map(len, selected.values()))} {brand_id} cases to {output_path}")


if __name__ == "__main__":
    app()
