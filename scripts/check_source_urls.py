"""Print the HTTP status of every configured OSHA download URL.

Exits 1 if any URL fails, so the scheduled CI job shows a warning. Run it
locally before `run-all` if downloads start failing.
"""

from __future__ import annotations

import sys

import httpx

from nextrecordable.config import ITA_CASE_DETAIL_URLS, ITA_SUMMARY_URLS
from nextrecordable.download import HEADERS


def main() -> int:
    failed = []
    urls = [("summary", y, u) for y, u in ITA_SUMMARY_URLS.items()]
    urls += [("case-detail", y, u) for y, u in ITA_CASE_DETAIL_URLS.items()]
    for kind, year, url in urls:
        try:
            status = httpx.head(url, follow_redirects=True, timeout=20.0, headers=HEADERS).status_code
        except httpx.HTTPError as exc:
            status = f"error: {exc.__class__.__name__}"
        print(f"{kind:12s} {year}  {status}  {url}")
        if not (isinstance(status, int) and status < 400):
            failed.append((kind, year))
    if failed:
        print(f"Stale or unreachable: {failed}. Update nextrecordable/config.py.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
