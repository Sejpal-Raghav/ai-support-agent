from pathlib import Path

import pandas as pd
import pytest

from support_agent.data.download import (
    DATASET_HANDLE,
    download_dataset,
    find_twcs_csv,
    place_downloaded_dataset,
)
from support_agent.data.ingest import REQUIRED_SOURCE_COLUMNS


def _write_valid_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{column: "" for column in REQUIRED_SOURCE_COLUMNS}]).to_csv(path, index=False)


def test_find_twcs_csv_supports_nested_download_directory(tmp_path: Path) -> None:
    expected = tmp_path / "nested" / "twcs.csv"
    _write_valid_csv(expected)
    assert find_twcs_csv(tmp_path) == expected


def test_place_downloaded_dataset_copies_and_validates_csv(tmp_path: Path) -> None:
    downloaded = tmp_path / "downloaded"
    _write_valid_csv(downloaded / "twcs.csv")
    destination = tmp_path / "raw" / "twcs.csv"

    result = place_downloaded_dataset(downloaded, destination)

    assert result == destination
    assert destination.is_file()


def test_place_downloaded_dataset_does_not_overwrite_by_default(tmp_path: Path) -> None:
    downloaded = tmp_path / "downloaded"
    _write_valid_csv(downloaded / "twcs.csv")
    destination = tmp_path / "twcs.csv"
    destination.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        place_downloaded_dataset(downloaded, destination)


def test_download_dataset_uses_expected_handle(tmp_path: Path) -> None:
    downloaded = tmp_path / "downloaded"
    _write_valid_csv(downloaded / "twcs.csv")
    requested_handles: list[str] = []

    def fake_downloader(handle: str) -> str:
        requested_handles.append(handle)
        return str(downloaded)

    destination = tmp_path / "raw" / "twcs.csv"
    assert download_dataset(destination, downloader=fake_downloader) == destination
    assert requested_handles == [DATASET_HANDLE]
