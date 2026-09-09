"""Download and place the public Kaggle source CSV used by the pipeline."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

from support_agent.data.ingest import REQUIRED_SOURCE_COLUMNS, validate_source_columns

DATASET_HANDLE = "thoughtvector/customer-support-on-twitter"
DEFAULT_DESTINATION = Path("data/raw/twcs.csv")


def find_twcs_csv(dataset_directory: Path) -> Path:
    """Return the single ``twcs.csv`` found in a Kaggle download directory."""
    dataset_directory = dataset_directory.resolve()
    if not dataset_directory.is_dir():
        raise NotADirectoryError(f"Kaggle download path is not a directory: {dataset_directory}")

    matches = sorted(path for path in dataset_directory.rglob("twcs.csv") if path.is_file())
    if not matches:
        raise FileNotFoundError(f"twcs.csv was not found under {dataset_directory}")
    if len(matches) > 1:
        rendered = ", ".join(path.as_posix() for path in matches)
        raise RuntimeError(f"Expected one twcs.csv under {dataset_directory}; found: {rendered}")
    return matches[0]


def place_downloaded_dataset(
    dataset_directory: Path,
    destination: Path = DEFAULT_DESTINATION,
    *,
    overwrite: bool = False,
) -> Path:
    """Validate and copy Kaggle's ``twcs.csv`` to the pipeline's source-data path."""
    source = find_twcs_csv(dataset_directory)
    destination = destination.resolve()

    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Destination already exists: {destination}. Pass --overwrite to replace it."
        )

    validate_source_columns(source, REQUIRED_SOURCE_COLUMNS)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source != destination:
        shutil.copy2(source, destination)
    validate_source_columns(destination, REQUIRED_SOURCE_COLUMNS)
    return destination


def download_dataset(
    destination: Path = DEFAULT_DESTINATION,
    *,
    overwrite: bool = False,
    downloader: Callable[[str], str] | None = None,
) -> Path:
    """Download the pinned Kaggle dataset and place its CSV at ``destination``."""
    if downloader is None:
        import kagglehub

        downloader = kagglehub.dataset_download

    downloaded_directory = Path(downloader(DATASET_HANDLE))
    return place_downloaded_dataset(downloaded_directory, destination, overwrite=overwrite)
