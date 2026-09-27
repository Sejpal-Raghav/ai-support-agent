from pathlib import Path

import pandas as pd
import pytest

from support_agent.data.ingest import (
    REQUIRED_SOURCE_COLUMNS,
    ingest_csv,
    iter_ingested_messages,
    parse_raw_row,
    validate_source_columns,
)


def test_validate_source_columns_rejects_missing_column(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("tweet_id,author_id\n1,customer\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing required source columns"):
        validate_source_columns(path, REQUIRED_SOURCE_COLUMNS)


def test_parse_raw_row_parses_response_ids() -> None:
    message = parse_raw_row(
        {
            "tweet_id": "1",
            "author_id": "customer",
            "inbound": "True",
            "created_at": "Mon Jan 01 00:00:00 +0000 2018",
            "text": "Need help",
            "response_tweet_id": "2, 3",
            "in_response_to_tweet_id": pd.NA,
        }
    )
    assert message.response_tweet_ids == ("2", "3")
    assert message.in_response_to_tweet_id is None


def test_ingest_round_trip_fixture(tmp_path: Path) -> None:
    output = tmp_path / "messages.parquet"
    stats = ingest_csv(Path("data/fixtures/tiny_tweets.csv"), output, chunksize=1)
    messages = list(iter_ingested_messages(output))
    assert stats.rows_read == 2
    assert stats.rows_written == 2
    assert stats.rows_rejected == 0
    assert [message.tweet_id for message in messages] == ["1", "2"]
    assert not output.with_suffix(".parquet.part").exists()


def test_ingest_honours_max_rows(tmp_path: Path) -> None:
    output = tmp_path / "limited.parquet"
    stats = ingest_csv(Path("data/fixtures/tiny_tweets.csv"), output, chunksize=2, max_rows=1)
    assert stats.rows_read == stats.rows_written == 1
