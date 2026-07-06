"""Client for the IEEE vTools OU List API.

The OU List API returns, for a single Organizational Unit (OU) identified by
its SPOID, both its parent SPOIDs and its child SPOIDs in one call, along with
type/status metadata and precomputed region/section/society/division ancestry.
That makes it well suited to lazy, one-call-per-node graph exploration.

    GET https://vtools.vtools.ieee.org/api/public/v1/ous/list?spoid=<SPOID>

The API is fully public (no authentication required). See the OU List API entry
in the vtools-api-docs README for details.
"""

import json
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import urllib3

OU_LIST_URL = "https://vtools.vtools.ieee.org/api/public/v1/ous/list"
# WebInABox Unit Details feed: unit name/type/url and a list of officers.
OFFICERS_URL = "https://webinabox.vtools.ieee.org/wibp_officers/feed/"
URL_REQUEST_TIMEOUT = 30  # seconds


@dataclass
class OU:
    """A single Organizational Unit as returned by the OU List API."""

    spoid: str
    name: str = ""
    type_code: str = ""
    type_desc: str = ""
    status_code: str = ""
    status_desc: str = ""
    url: str = ""
    parents: list = field(default_factory=list)
    children: list = field(default_factory=list)
    region_spoids: list = field(default_factory=list)
    section_spoids: list = field(default_factory=list)
    society_spoids: list = field(default_factory=list)
    division_spoids: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)


def _split_spoids(value):
    """Split a comma-separated SPOID string into a clean list.

    The API returns either null or a string like "R60007,C016". Duplicate and
    empty entries are dropped while preserving first-seen order.
    """
    if not value:
        return []
    seen = []
    for token in str(value).split(","):
        token = token.strip()
        if token and token not in seen:
            seen.append(token)
    return seen


def parse_ou(attributes):
    """Build an OU from the 'attributes' dict of an OU List API record."""
    return OU(
        spoid=attributes.get("spoid", ""),
        name=attributes.get("name", "") or "",
        type_code=attributes.get("type-code", "") or "",
        type_desc=attributes.get("type-description", "") or "",
        status_code=attributes.get("status-code", "") or "",
        status_desc=attributes.get("status-description", "") or "",
        url=attributes.get("url", "") or "",
        parents=_split_spoids(attributes.get("parent-spoids")),
        children=_split_spoids(attributes.get("child-spoids")),
        region_spoids=_split_spoids(attributes.get("region-spoids")),
        section_spoids=_split_spoids(attributes.get("section-spoids")),
        society_spoids=_split_spoids(attributes.get("society-spoids")),
        division_spoids=_split_spoids(attributes.get("division-spoids")),
        raw=attributes,
    )


def get_ou(spoid, http=None):
    """Fetch and parse a single OU by SPOID.

    Returns an OU on success, or None if the SPOID is unknown, the response is
    empty, or the response can't be parsed. Network/parse problems are logged
    and swallowed so a bad SPOID can't crash an interactive session.
    """
    spoid = (spoid or "").strip()
    if not spoid:
        return None

    if http is None:
        http = urllib3.PoolManager()

    try:
        resp = http.request(
            "GET",
            OU_LIST_URL,
            fields={"spoid": spoid},
            timeout=URL_REQUEST_TIMEOUT,
        )
    except Exception as exc:  # network error, DNS, timeout, ...
        logging.error("OU List request for %s failed: %s", spoid, exc)
        return None

    if resp.status != 200:
        logging.warning("OU List request for %s returned HTTP %s",
                        spoid, resp.status)
        return None

    try:
        payload = json.loads(resp.data.decode("utf-8"))
    except Exception as exc:
        logging.error("Could not parse OU List response for %s: %s",
                      spoid, exc)
        return None

    records = payload.get("data") or []
    if not records:
        logging.info("OU List returned no data for %s", spoid)
        return None

    attributes = records[0].get("attributes") or {}
    if not attributes.get("spoid"):
        attributes.setdefault("spoid", spoid)
    return parse_ou(attributes)


def get_officers(spoid, http=None):
    """Fetch a unit's officers from the WebInABox Unit Details feed.

    Returns a list of {"position": ..., "name": ...} dicts, in feed order.
    Returns an empty list if the unit has no published officer list (e.g. the
    admin restricted it) or on any network/parse error.
    """
    spoid = (spoid or "").strip()
    if not spoid:
        return []

    if http is None:
        http = urllib3.PoolManager()

    try:
        resp = http.request("GET", OFFICERS_URL + spoid,
                            timeout=URL_REQUEST_TIMEOUT)
    except Exception as exc:
        logging.error("Officers request for %s failed: %s", spoid, exc)
        return []

    if resp.status != 200 or not resp.data:
        return []

    try:
        root = ET.fromstring(resp.data)
    except Exception as exc:
        logging.error("Could not parse officers XML for %s: %s", spoid, exc)
        return []

    officers = []
    for officer in root.findall(".//officer"):
        position = (officer.findtext("position") or "").strip()
        name = (officer.findtext("name") or "").strip()
        if position or name:
            officers.append({"position": position, "name": name})
    return officers
