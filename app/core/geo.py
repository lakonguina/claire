"""Country of an IP address, from the DB-IP Lite database (CC BY 4.0).

The database is downloaded into the image at build time
(``python -m app.core.geo``). Without it, lookups return ``None``.
"""

from __future__ import annotations

import datetime
import functools
import gzip
import urllib.request
from pathlib import Path

import maxminddb

DB = Path(__file__).resolve().parents[2] / "geo" / "country.mmdb"
URL = "https://download.db-ip.com/free/dbip-country-lite-{month}.mmdb.gz"


@functools.cache
def _reader() -> maxminddb.Reader | None:
    return maxminddb.open_database(DB) if DB.exists() else None


def country(ip: str | None) -> str | None:
    """Return the ISO country code for ``ip``, or ``None`` if unknown."""
    reader = _reader()
    if not ip or reader is None:
        return None
    try:
        record = reader.get(ip)
    except ValueError:
        return None
    if not isinstance(record, dict):
        return None
    return record.get("country", {}).get("iso_code")


def download() -> None:
    """Fetch this month's database (last month's early in the month)."""
    first = datetime.date.today().replace(day=1)
    previous = (first - datetime.timedelta(days=1)).replace(day=1)
    for month in (first, previous):
        url = URL.format(month=month.strftime("%Y-%m"))
        # DB-IP refuses urllib's default User-Agent.
        request = urllib.request.Request(url, headers={"User-Agent": "claire"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = gzip.decompress(response.read())
        except OSError:
            continue
        DB.parent.mkdir(parents=True, exist_ok=True)
        DB.write_bytes(data)
        return
    print("geo: no DB-IP database downloaded; countries stay unknown")


if __name__ == "__main__":
    download()
