"""Trusted verifier client for MAST's Gaia DR3 TAP service.

This runs on the control box, never inside the agent sandbox. The model under
evaluation never sees or influences this code path; it only produces a CSV that
we check against the live catalog.

Endpoint shape (verified 2026-11-08):

    GET {base}/sync?REQUEST=doQuery&LANG=ADQL&FORMAT=json&QUERY=...
    -> {"info": [{"name": ..., "type": ...}, ...], "data": [[...], ...]}

``source_id`` comes back as a JSON *string*, which is what we want: a 64-bit
Gaia id does not survive a round trip through a double.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import httpx

log = logging.getLogger(__name__)

MAST_GAIADR3_TAP = "https://mast.stsci.edu/vo-tap/api/v0.1/gaiadr3"
GAIA_SOURCE_TABLE = "dbo.gaia_source"
USER_AGENT = "astro-llm-eval-verifier/0.1 (+https://github.com/jwfraustro/astro-llm-eval)"

# HTTP statuses worth retrying. 4xx (other than 429) means *our* query is wrong,
# which is an infra/verifier bug, not something a retry fixes.
_RETRY_STATUSES = {429, 500, 502, 503, 504}


class TAPError(RuntimeError):
    """The verifier could not get an answer from the service.

    This is an infrastructure failure, not a model failure. Scorers must not
    convert it into INCORRECT.
    """


class QueryRejected(TAPError):
    """The service answered, and said the ADQL was wrong (HTTP 4xx).

    For the verifier's own queries this is still an infra bug, hence the
    subclassing. A task that runs *the model's* query catches this first and
    scores it INCORRECT.
    """


@dataclass(frozen=True)
class GaiaSource:
    source_id: int
    ra: float
    dec: float


class GaiaVerifier:
    """Minimal synchronous TAP client with retries and a polite request spacing.

    Args:
        base_url: TAP service root (no trailing ``/sync``).
        table: Fully qualified table name holding ``source_id, ra, dec``.
        timeout: Per-request timeout in seconds.
        retries: Attempts per query before raising :class:`TAPError`.
        chunk_size: Max ids per ``IN (...)`` clause.
        min_interval: Minimum seconds between requests to the service.
    """

    def __init__(
        self,
        base_url: str = MAST_GAIADR3_TAP,
        table: str = GAIA_SOURCE_TABLE,
        *,
        timeout: float = 60.0,
        retries: int = 3,
        chunk_size: int = 100,
        min_interval: float = 0.5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.table = table
        self.timeout = timeout
        self.retries = retries
        self.chunk_size = chunk_size
        self.min_interval = min_interval
        self._last_request = 0.0
        self._client = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        )

    # ------------------------------------------------------------------ core

    def sync_query(self, adql: str) -> list[list]:
        """Run a synchronous ADQL query and return the ``data`` rows."""
        return self.sync_query_full(adql)[1]

    def sync_query_full(self, adql: str) -> tuple[list[str], list[list]]:
        """Run a synchronous ADQL query; return ``(column_names, rows)``."""
        params = {
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "json",
            "QUERY": adql,
        }
        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 1):
            self._pace()
            try:
                resp = self._client.get(f"{self.base_url}/sync", params=params)
            except httpx.HTTPError as exc:  # timeouts, connection resets, ...
                last_exc = exc
                log.warning("TAP request failed (attempt %d/%d): %s", attempt, self.retries, exc)
                self._backoff(attempt)
                continue

            if resp.status_code in _RETRY_STATUSES:
                last_exc = TAPError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                log.warning("TAP returned %s (attempt %d/%d)", resp.status_code, attempt, self.retries)
                self._backoff(attempt)
                continue
            if resp.status_code != 200:
                # Bad ADQL or a changed service: retrying will not help.
                detail = f"HTTP {resp.status_code} for query {adql!r}: {_error_text(resp.text)}"
                raise QueryRejected(detail) if 400 <= resp.status_code < 500 else TAPError(detail)

            try:
                payload = resp.json()
            except ValueError as exc:
                raise TAPError(f"non-JSON response: {resp.text[:200]}") from exc
            if not isinstance(payload, dict) or "data" not in payload:
                raise TAPError(f"unexpected response shape: {str(payload)[:200]}")
            columns = [str(c.get("name", "")) for c in payload.get("info", []) if isinstance(c, dict)]
            return columns, payload["data"]

        raise TAPError(f"TAP query failed after {self.retries} attempts: {last_exc}")

    # --------------------------------------------------------------- queries

    def lookup(self, ids: Iterable[int]) -> dict[int, GaiaSource]:
        """Return the catalog ``(ra, dec)`` for each id that actually exists.

        Ids that are not in the catalog are simply absent from the result.
        """
        unique = sorted({int(i) for i in ids})
        found: dict[int, GaiaSource] = {}
        for chunk in _chunks(unique, self.chunk_size):
            in_list = ", ".join(str(i) for i in chunk)
            adql = f"SELECT source_id, ra, dec FROM {self.table} WHERE source_id IN ({in_list})"
            for sid, ra, dec in self.sync_query(adql):
                sid_int = int(sid)
                found[sid_int] = GaiaSource(sid_int, float(ra), float(dec))
        return found

    def count_in_cone(self, ra: float, dec: float, radius: float, *, limit: int) -> int:
        """Number of sources in the cone, capped at ``limit``.

        Used when generating task variants to make sure a cone is satisfiable
        without pulling a full count over a dense field.
        """
        adql = (
            f"SELECT TOP {int(limit)} source_id FROM {self.table} "
            f"WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {radius}))=1"
        )
        return len(self.sync_query(adql))

    def healthcheck(self) -> bool:
        """True if the service answers a trivial query."""
        try:
            rows = self.sync_query(f"SELECT TOP 1 source_id FROM {self.table}")
        except TAPError as exc:
            log.error("TAP healthcheck failed: %s", exc)
            return False
        return len(rows) == 1

    # --------------------------------------------------------------- helpers

    def _pace(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    @staticmethod
    def _backoff(attempt: int) -> None:
        time.sleep(min(2.0**attempt, 30.0))

    def close(self) -> None:
        self._client.close()


_VOTABLE_INFO_RE = re.compile(r'<INFO[^>]*name="QUERY_STATUS"[^>]*>(.*?)</INFO>', re.DOTALL)


def _error_text(body: str, limit: int = 500) -> str:
    """TAP errors come back as a VOTable; pull out the human-readable part."""
    m = _VOTABLE_INFO_RE.search(body)
    text = " ".join((m.group(1) if m else body).split())
    return text[:limit]


def _chunks(seq: Sequence[int], n: int) -> Iterable[Sequence[int]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]
