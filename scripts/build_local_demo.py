"""Build a compact AmazonHelp evidence corpus from the local ingested subset."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import typer

from support_agent.demo_runtime import INTENT_SEEDS, sanitize_historical_reply

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
    inbound = {
        str(row.tweet_id): str(row.text)
        for row in frame.loc[frame.inbound, ["tweet_id", "text"]].itertuples(index=False)
    }
    selected: dict[str, list[dict[str, str]]] = defaultdict(list)
    outbound = frame.loc[~frame.inbound & (frame.author_id == brand_id)]
    for row in outbound.sort_values("tweet_id").itertuples(index=False):
        parent_id = row.in_response_to_tweet_id
        customer_text = inbound.get(str(parent_id)) if parent_id is not None else None
        if not customer_text:
            continue
        intent = intent_for(customer_text)
        if intent is None or len(selected[intent]) >= per_intent:
            continue
        raw_reply = str(row.text)
        if re.search(r"https?://", raw_reply, flags=re.IGNORECASE):
            continue
        reply = sanitize_historical_reply(raw_reply)
        if len(reply) < 12:
            continue
        selected[intent].append(
            {
                "case_id": f"{brand_id}:{row.tweet_id}",
                "customer_text": customer_text,
                "reply_text": reply,
                "intent_id": intent,
            }
        )
    missing = sorted(set(INTENT_SEEDS) - {intent for intent, rows in selected.items() if rows})
    if missing:
        raise ValueError(f"no historical cases found for intents: {missing}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for intent in INTENT_SEEDS:
            for record in selected[intent]:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    typer.echo(f"Wrote {sum(map(len, selected.values()))} {brand_id} cases to {output_path}")


if __name__ == "__main__":
    app()
