"""HTTP downloader with caching and progress bars.

Each download is idempotent: if the destination file already exists and is
non-empty, the download is skipped. Use force=True to re-download.

Downloads are streamed and written to a .part file, then renamed on success,
so partial files never look like completed downloads.
"""

from __future__ import annotations

from pathlib import Path

import httpx
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from . import __version__
from .config import ITA_CASE_DETAIL_URLS, ITA_SUMMARY_URLS, RAW_DIR

CHUNK_SIZE = 1 << 16  # 64 KiB
# Say who is downloading. httpx's default "python-httpx/x.y" is a common reason
# for a government CDN to refuse a request.
HEADERS = {"User-Agent": f"nextrecordable/{__version__} (+https://github.com/ASM-21/Nextrecordable)"}


def download_file(url: str, dest: Path, *, force: bool = False) -> Path:
    """Stream `url` to `dest`. Skip if dest exists and is non-empty unless force=True."""
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    with httpx.stream("GET", url, follow_redirects=True, timeout=60.0, headers=HEADERS) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))

        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            transient=False,
        ) as progress:
            task = progress.add_task(f"[cyan]{dest.name}", total=total or None)
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes(chunk_size=CHUNK_SIZE):
                    f.write(chunk)
                    progress.update(task, advance=len(chunk))

    tmp.rename(dest)
    return dest


def download_ita_summary(years: list[int] | None = None, *, force: bool = False) -> list[Path]:
    """Download ITA Form 300A summary zips for the requested years."""
    years = years or sorted(ITA_SUMMARY_URLS)
    paths: list[Path] = []
    for year in years:
        if year not in ITA_SUMMARY_URLS:
            raise ValueError(f"No summary URL configured for year {year}")
        url = ITA_SUMMARY_URLS[year]
        dest = RAW_DIR / "ita_summary" / f"ita_summary_{year}.zip"
        paths.append(download_file(url, dest, force=force))
    return paths


def download_ita_case_detail(
    years: list[int] | None = None, *, force: bool = False
) -> list[Path]:
    """Download ITA Form 300/301 case detail zips for the requested years."""
    years = years or sorted(ITA_CASE_DETAIL_URLS)
    paths: list[Path] = []
    for year in years:
        if year not in ITA_CASE_DETAIL_URLS:
            raise ValueError(f"No case detail URL configured for year {year}")
        url = ITA_CASE_DETAIL_URLS[year]
        dest = RAW_DIR / "ita_case_detail" / f"ita_case_detail_{year}.zip"
        paths.append(download_file(url, dest, force=force))
    return paths
