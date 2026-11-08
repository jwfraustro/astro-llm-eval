"""Trusted verifier client for an ObjObsSAP service (IVOA Working Draft).

Ground truth for the observability task is the service's own answer: the
protocol is deterministic for a given (POS, TIME, MIN_OBS, FACILITY), so the
verifier simply asks the same question and compares intervals.

Service shape (verified 2026-11-08 against https://vo-labs.dev/objobssap/api):

    GET {base}/query?POS=ra,dec&TIME=mjd0/mjd1&MIN_OBS=s[&FACILITY=...]
    -> VOTable with FIELDs t_start, t_stop (MJD, days), t_observability (s),
       facility, plus t_validity / validity_* bookkeeping.

Intervals that overlap the requested TIME range are returned whole, not clipped.
Omitting FACILITY selects the service's default facility. An unsupported
FACILITY yields an empty table with QUERY_STATUS=OK, not an error.
"""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)

VO_LABS_OBJOBSSAP = "https://vo-labs.dev/objobssap/api"
VO_LABS_DOCS_URL = "https://vo-labs.dev/objobssap"
USER_AGENT = "astro-llm-eval-verifier/0.1 (+https://github.com/jwfraustro/astro-llm-eval)"

_RETRY_STATUSES = {429, 500, 502, 503, 504}


class ObjObsSAPError(RuntimeError):
    """Infrastructure failure: the verifier could not get an answer."""


@dataclass(frozen=True)
class Interval:
    t_start: float  # MJD
    t_stop: float  # MJD
    t_observability: float | None = None  # seconds, as reported by the service
    facility: str | None = None

    @property
    def duration_s(self) -> float:
        return (self.t_stop - self.t_start) * 86400.0


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_votable_intervals(xml_text: str) -> list[Interval]:
    """Extract intervals from an ObjObsSAP VOTable. Raises on an error INFO."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ObjObsSAPError(f"response is not XML: {xml_text[:200]!r}") from exc

    for info in root.iter():
        if _local(info.tag) == "INFO":
            name = (info.get("name") or info.get("ID") or "").lower()
            value = info.get("value") or ""
            if name == "error" or (name == "query_status" and value.upper() == "ERROR"):
                raise ObjObsSAPError(f"service error: {value or info.text}")

    fields = [f.get("name") or f.get("ID") for f in root.iter() if _local(f.tag) == "FIELD"]
    if not fields:
        raise ObjObsSAPError("VOTable has no FIELDs")
    try:
        i_start, i_stop = fields.index("t_start"), fields.index("t_stop")
    except ValueError as exc:
        raise ObjObsSAPError(f"VOTable lacks t_start/t_stop; fields={fields}") from exc
    i_obs = fields.index("t_observability") if "t_observability" in fields else None
    i_fac = fields.index("facility") if "facility" in fields else None

    out: list[Interval] = []
    for tr in root.iter():
        if _local(tr.tag) != "TR":
            continue
        cells = [(td.text or "").strip() for td in tr if _local(td.tag) == "TD"]
        out.append(
            Interval(
                t_start=float(cells[i_start]),
                t_stop=float(cells[i_stop]),
                t_observability=float(cells[i_obs]) if i_obs is not None and cells[i_obs] else None,
                facility=cells[i_fac] if i_fac is not None else None,
            )
        )
    return sorted(out, key=lambda iv: iv.t_start)


class ObjObsSAPClient:
    def __init__(
        self,
        base_url: str = VO_LABS_OBJOBSSAP,
        *,
        timeout: float = 60.0,
        retries: int = 3,
        min_interval: float = 0.5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.retries = retries
        self.min_interval = min_interval
        self._last_request = 0.0
        self._client = httpx.Client(timeout=timeout, headers={"User-Agent": USER_AGENT})

    def query(
        self,
        ra: float,
        dec: float,
        t0: float,
        t1: float,
        *,
        min_obs: int = 30,
        facility: str | None = None,
    ) -> list[Interval]:
        params: dict[str, str] = {"POS": f"{ra},{dec}", "TIME": f"{t0}/{t1}", "MIN_OBS": str(int(min_obs))}
        if facility is not None:
            params["FACILITY"] = facility
        return parse_votable_intervals(self._get("/query", params))

    def healthcheck(self) -> bool:
        try:
            # A position that always rises from a mid-northern site, one day.
            self.query(90.0, 45.0, 61000.0, 61001.0)
            return True
        except ObjObsSAPError as exc:
            log.error("ObjObsSAP healthcheck failed: %s", exc)
            return False

    def _get(self, path: str, params: dict[str, str]) -> str:
        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 1):
            self._pace()
            try:
                resp = self._client.get(f"{self.base_url}{path}", params=params)
            except httpx.HTTPError as exc:
                last_exc = exc
                log.warning("ObjObsSAP request failed (attempt %d/%d): %s", attempt, self.retries, exc)
                time.sleep(min(2.0**attempt, 30.0))
                continue
            if resp.status_code in _RETRY_STATUSES:
                last_exc = ObjObsSAPError(f"HTTP {resp.status_code}")
                time.sleep(min(2.0**attempt, 30.0))
                continue
            if resp.status_code != 200:
                raise ObjObsSAPError(f"HTTP {resp.status_code} for {params}: {resp.text[:300]}")
            return resp.text
        raise ObjObsSAPError(f"request failed after {self.retries} attempts: {last_exc}")

    def _pace(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def close(self) -> None:
        self._client.close()
