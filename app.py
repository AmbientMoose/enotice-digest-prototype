"""eNotice Digest -- prototype.

Pick an eNotice data file, an IEEE Section, and one or more units related to
that Section (its ancestors and descendants). The main panel then summarizes
every "sent" eNotice in the chosen file that was addressed to one of the
selected units (matched on the ``recipient_SPOIDs`` column).

Related units are found by walking the IEEE OU List API graph outward from the
Section -- up through parents and down through children -- exactly the way the
IEEE OU Explorer app determines a unit's parents and children (API edges plus
the reciprocity supplement). Units of type Academic, Grouping, or Other are
neither shown nor traversed through.

Run locally:  streamlit run app.py
"""

import concurrent.futures
import csv
import hmac
import html
import json
import os
import re
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import streamlit as st
import urllib3
from streamlit_searchbox import st_searchbox

import ouclient
import outype
from ouclient import OU
from outype import UnitType

# --------------------------------------------------------------------------- #
# Configuration

_HERE = Path(__file__).parent
_DATA_DIR = _HERE / "eNotice_data"
_INDEX_PATH = _HERE / "units.csv"
_RECIP_PATH = _HERE / "reciprocity_violations.csv"
_LOGO_PATH = _HERE / "assets" / "ieee-logo.png"
_TAG_CATEGORIES_PATH = _HERE / "tag_categories.csv"
_TAXONOMY_PATH = _HERE / "taxonomy" / "ieee_taxonomy.csv"

# Digest view shows this many eNotice tiles per page (newest first); the user
# pages through the rest with the Newer/Older controls.
_DIGEST_MAX_TILES = 6

# AI summary agent (Anthropic). A light model keeps per-digest cost down.
_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
_SUMMARY_MAX_TOKENS = 220
_TAG_MAX_TOKENS = 320
_FETCH_TIMEOUT = 20  # seconds, per source page fetch

# AI tag agent: at most this many *generated* tags per eNotice. Event tags
# (rule 4) and taxonomy ancestors (rule 5) are added on top and do not count
# against this limit.
_DIGEST_MAX_TAGS = 6

# Rule 5 (IEEE Taxonomy): at most this many taxonomy terms count toward the tag
# limit (lean toward one higher-level term covering multiple topics). A term at
# this taxonomy depth or deeper is treated as "lower-level" and is only used
# when named verbatim in the content; otherwise a broader ancestor is used.
_TAXONOMY_MAX_TAGS = 2
_TAX_DEEP_LEVEL = 2

# vTools Events API: structured event data (tags, category, location, content)
# by event id. Preferred over scraping the event_url HTML page.
_EVENTS_API = "https://events.vtools.ieee.org/api/public/v5/events/list?id={}"

# Digest search: scan the full text of the newest this-many eNotices in the
# current digest (selected units + sidebar date range) and keep those whose
# public page matches every search term.
_SEARCH_SCAN_LIMIT = 200
_SEARCH_TEXT_CAP = 20000  # chars of page text kept for term matching

# Tile image: OpenAI generation (fallback when no suitable page image exists).
_IMAGE_MODEL = "gpt-image-1"
_IMAGE_SIZE = "1024x1024"
_IMAGE_QUALITY = "low"

# Substrings that mark an <img> as chrome (logos, banners, icons, maps) rather
# than an eNotice's own content picture.
_CHROME_IMG = (
    "ieee-logo", "logo_ieee_vtools", "vtools/logo", "favicon",
    "enotice_header", "enotice_footer", "fb_logo", "twitter", "facebook",
    "ieee_logo_share", "add_to_calendar", "ical_icon", "google_cal",
    "staticmap", "maps.googleapis", "content/dam/ieee-org",
    "meeting_registration_link_image", "/assets/",
)

_SEARCH_MIN_CHARS = 3
_SEARCH_LIMIT = 50

# Statistic columns that are excluded from the digest summary table.
_STAT_COLUMNS = ["Sent", "Delivered", "Bounced", "Opened"]

# Other columns hidden from the table (status is constant "sent" after
# filtering, so it carries no information).
_HIDDEN_COLUMNS = ["status"]

# Unit types that are never shown in, nor traversed through, the related-units
# list.
_EXCLUDED_TYPES = {UnitType.ACADEMIC, UnitType.GROUPING, UnitType.UNKNOWN}

_MAX_WORKERS = 8
_HTTP = urllib3.PoolManager(maxsize=_MAX_WORKERS)

st.set_page_config(page_title="eNotice Digest", page_icon="📬", layout="wide")


# --------------------------------------------------------------------------- #
# Access gate (modeled on the IEEE Section Operations Assistant)

def _load_secrets_into_env():
    """Copy API keys from Streamlit secrets into the environment so the Anthropic
    SDK (which reads os.getenv) works on Streamlit Cloud. No-op if no secrets."""
    try:
        secrets = st.secrets
    except Exception:
        return
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        if key in secrets and not os.getenv(key):
            os.environ[key] = str(secrets[key])


def check_password():
    """Gate the app behind a shared password in st.secrets['app_password'] (or
    the APP_PASSWORD env var). If none is configured, the app is open."""
    configured = ""
    try:
        if hasattr(st, "secrets"):
            configured = str(st.secrets.get("app_password", ""))
    except Exception:
        configured = ""
    if not configured:
        configured = os.getenv("APP_PASSWORD", "")
    if not configured:
        return  # open deployment
    if st.session_state.get("auth_ok"):
        return

    def _check():
        ok = hmac.compare_digest(st.session_state.get("pw", ""), configured)
        st.session_state["auth_ok"] = ok
        if ok:
            st.session_state.pop("pw", None)

    st.text_input("Enter access password", type="password", key="pw",
                  on_change=_check)
    if st.session_state.get("auth_ok") is False:
        st.error("Incorrect password. Ask Chris for the access password.")
    st.stop()


_load_secrets_into_env()
check_password()


# --------------------------------------------------------------------------- #
# Shared helpers (mirroring IEEE OU Explorer)

def resolve_spoid(spoid):
    """Region 10's OU List data lives under 'R0'; look that up for 'R10'."""
    return "R0" if spoid == "R10" else spoid


@st.cache_data(ttl=3600, show_spinner=False)
def load_ou(spoid):
    """Fetch a full OU by SPOID, cached (Streamlit Cloud disk is ephemeral)."""
    return ouclient.get_ou(spoid, http=_HTTP)


@st.cache_data(show_spinner=False)
def load_supplements():
    """Load reciprocity_violations.csv into edge-supplement maps.

    Returns (extra_children, extra_parents):
      extra_children[parent_spoid] -> [child spoids the API omits]
      extra_parents[child_spoid]   -> [parent spoids the API omits]
    SPOIDs are normalized for the R0/R10 alias, matching the OU Explorer.
    """
    extra_children, extra_parents = {}, {}
    if not _RECIP_PATH.exists():
        return extra_children, extra_parents
    with open(_RECIP_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            unit = resolve_spoid((r.get("unit_spoid") or "").strip())
            related = resolve_spoid((r.get("related_spoid") or "").strip())
            issue = r.get("issue") or ""
            if not unit or not related:
                continue
            if issue.startswith("parent"):      # parent 'related' omits 'unit'
                extra_children.setdefault(related, [])
                if unit not in extra_children[related]:
                    extra_children[related].append(unit)
            elif issue.startswith("child"):      # child 'related' omits 'unit'
                extra_parents.setdefault(related, [])
                if unit not in extra_parents[related]:
                    extra_parents[related].append(unit)
    return extra_children, extra_parents


# --------------------------------------------------------------------------- #
# Section search index (units.csv, restricted to Sections)

@st.cache_data(show_spinner=False)
def load_section_index():
    """Load Sections from units.csv as searchbox rows.

    Only units that classify as a Section are kept, so the picker offers
    Sections exclusively.
    """
    rows = []
    if not _INDEX_PATH.exists():
        return rows
    with open(_INDEX_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            spoid = (r.get("spoid") or "").strip()
            name = (r.get("name") or "").strip()
            if not spoid or not name:
                continue
            type_desc = (r.get("type") or "").strip()
            if outype.classify_ou(OU(spoid=spoid, type_desc=type_desc)) \
                    is not UnitType.SECTION:
                continue
            _c, _s, emoji, _z = outype.style_for(UnitType.SECTION)
            rows.append({"spoid": spoid, "name": name,
                         "name_lower": name.lower(),
                         "label": f"{emoji} {name} ({spoid})"})
    return rows


def search_sections(query):
    """st_searchbox callback: Sections whose name contains the query."""
    q = (query or "").strip().lower()
    if len(q) < _SEARCH_MIN_CHARS:
        return []
    matches = [r for r in load_section_index() if q in r["name_lower"]]
    matches.sort(key=lambda r: (not r["name_lower"].startswith(q),
                                r["name_lower"]))
    return [(r["label"], r["spoid"]) for r in matches[:_SEARCH_LIMIT]]


# --------------------------------------------------------------------------- #
# Related-units traversal

def _fetch_many(spoids):
    """Fetch several OUs concurrently; return {spoid: OU} (None dropped)."""
    out = {}
    if not spoids:
        return out
    with concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        for spoid, ou in zip(spoids,
                             pool.map(lambda s: load_ou(resolve_spoid(s)),
                                      spoids)):
            if ou is not None:
                out[spoid] = ou
    return out


def _walk(start_ou, edges_of, seen, collected, progress=None):
    """Breadth-first walk in one direction (parents or children).

    ``edges_of(ou)`` returns the next SPOIDs to consider. A node is added to
    ``collected`` and expanded only if it fetches successfully and is not an
    excluded type; excluded nodes are neither shown nor traversed through.
    ``progress(len(collected))``, when given, is called after each BFS level so
    callers can drive a progress indicator as units are discovered.
    """
    frontier = [start_ou]
    while frontier:
        candidates = []
        for ou in frontier:
            for sp in edges_of(ou):
                if resolve_spoid(sp) not in seen:
                    seen.add(resolve_spoid(sp))
                    candidates.append(sp)
        fetched = _fetch_many(candidates)
        frontier = []
        for sp in candidates:
            ou = fetched.get(sp)
            if ou is None:
                continue
            if outype.classify_ou(ou) in _EXCLUDED_TYPES:
                continue
            collected[ou.spoid] = ou
            frontier.append(ou)
        if progress is not None:
            progress(len(collected))


def related_units(section_spoid, progress=None):
    """Return {spoid: OU} for the Section plus its ancestors and descendants.

    ``progress`` is an optional ``callable(units_found_so_far)`` called after
    each BFS level so the caller can drive a progress bar. This function is not
    ``@st.cache_data``-cached (so ``progress`` may safely update a Streamlit
    element); the sidebar memoizes the result per Section in ``st.session_state``
    instead.

    Parents/children are the OU List API edges supplemented by the reciprocity
    data (as in the OU Explorer). Academic/Grouping/Other units are excluded
    and not traversed through. The Section itself is always included.
    """
    extra_children, extra_parents = load_supplements()
    root = load_ou(resolve_spoid(section_spoid))
    if root is None:
        return {}

    collected = {root.spoid: root}
    seen = {resolve_spoid(root.spoid)}

    def parents_of(ou):
        return list(ou.parents) + extra_parents.get(resolve_spoid(ou.spoid), [])

    def children_of(ou):
        return list(ou.children) + \
            extra_children.get(resolve_spoid(ou.spoid), [])

    _walk(root, parents_of, seen, collected, progress)   # ancestors
    _walk(root, children_of, seen, collected, progress)  # descendants
    return collected


def recipient_pool(related, filename):
    """Related units that received at least one eNotice in the file -- the
    selectable list in the Related-units picker (kept only if a recipient in the
    chosen file). Derived from an already-computed related-units map."""
    recipients = load_recipient_spoids(filename)
    return {u: ou for u, ou in related.items()
            if resolve_spoid(u) in recipients}


@st.cache_data(show_spinner=False)
def section_ancestors(section_spoid):
    """{spoid: OU} for the Section's ancestors (parents, grandparents, ...).

    Parents-only walk via API + reciprocity, excluding and not traversing
    through Academic/Grouping/Other, the same model as related_units.
    """
    _ec, extra_parents = load_supplements()
    root = load_ou(resolve_spoid(section_spoid))
    if root is None:
        return {}
    collected = {}
    seen = {resolve_spoid(root.spoid)}

    def parents_of(ou):
        return list(ou.parents) + extra_parents.get(resolve_spoid(ou.spoid), [])

    _walk(root, parents_of, seen, collected)
    return collected


def society_links(related):
    """Map the Section's chapters to the selectable Societies that parent them,
    from an already-computed related-units map.

    Returns (chapters_by_soc, sbc_links):
      chapters_by_soc[society_spoid] -> [Chapter / Joint Chapter spoids] whose
          direct parent is that Society.
      sbc_links -> list of (sbc_spoid, student_branch_spoid, society_spoid) for
          Student Branch Chapters, which are added only once their Student
          Branch is also selected.
    society_spoids are normalized and limited to the Societies in units.csv.
    """
    societies = {resolve_spoid(r["spoid"]) for r in load_society_index()}
    _ec, extra_parents = load_supplements()
    chapters_by_soc, sbc_links = {}, []
    for sp, ou in related.items():
        kind = outype.classify_ou(ou)
        raw_parents = list(ou.parents) + extra_parents.get(resolve_spoid(sp), [])
        soc_parents = [p for p in {resolve_spoid(x) for x in raw_parents}
                       if p in societies]
        if kind is UnitType.CHAPTER:
            for soc in soc_parents:
                chapters_by_soc.setdefault(soc, []).append(sp)
        elif kind is UnitType.STUDENT_BRANCH_CHAPTER:
            sb = None
            for p in raw_parents:
                po = related.get(p)
                if po is not None and \
                        outype.classify_ou(po) is UnitType.STUDENT_BRANCH:
                    sb = po.spoid
                    break
            for soc in soc_parents:
                sbc_links.append((sp, sb, soc))
    return chapters_by_soc, sbc_links


@st.cache_data(show_spinner=False)
def load_society_index():
    """Load Societies from units.csv as picker rows ({spoid, name, label})."""
    rows = []
    if not _INDEX_PATH.exists():
        return rows
    _c, _s, emoji, _z = outype.style_for(UnitType.SOCIETY)
    with open(_INDEX_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            spoid = (r.get("spoid") or "").strip()
            name = (r.get("name") or "").strip()
            if not spoid or not name:
                continue
            type_desc = (r.get("type") or "").strip()
            if outype.classify_ou(OU(spoid=spoid, type_desc=type_desc)) \
                    is not UnitType.SOCIETY:
                continue
            rows.append({"spoid": spoid, "name": name,
                         "label": f"{emoji} {name} ({spoid})"})
    rows.sort(key=lambda r: r["name"].lower())
    return rows


def unit_label(ou):
    _c, _s, emoji, _z = outype.style_for(outype.classify_ou(ou))
    name = ou.name or ""
    return f"{emoji} {name} ({ou.spoid})" if name else f"{emoji} ({ou.spoid})"


# --------------------------------------------------------------------------- #
# eNotice data

def list_data_files():
    if not _DATA_DIR.exists():
        return []
    return sorted(p.name for p in _DATA_DIR.glob("*.csv"))


@st.cache_data(show_spinner=False)
def load_enotices(filename):
    """Load one eNotice CSV as strings, keeping only 'sent' notices.

    Adds a hidden ``_sent_dt`` datetime column (parsed from ``sent_at``) used
    for the date-range filter; it is dropped before the table is displayed.
    """
    df = pd.read_csv(_DATA_DIR / filename, dtype=str,
                     keep_default_na=False, encoding="utf-8")
    # Drop empty/unnamed trailing columns (a stray comma in the CSV header
    # produces an "Unnamed: N" column).
    df = df[[c for c in df.columns
             if c and not str(c).startswith("Unnamed:")]]
    if "status" in df.columns:
        df = df[df["status"].str.strip().str.lower() == "sent"]
    df = df.reset_index(drop=True)
    if "sent_at" in df.columns:
        df["_sent_dt"] = pd.to_datetime(df["sent_at"], errors="coerce")
    else:
        df["_sent_dt"] = pd.NaT
    return df


def parse_recipient_spoids(cell):
    """Normalized SPOID set from one recipient_SPOIDs cell.

    The eNotice CSV separates multiple recipients with '|' (pipe); commas are
    tolerated too. SPOIDs are normalized with resolve_spoid so they match the
    traversal SPOIDs.
    """
    if not cell:
        return set()
    return {resolve_spoid(t.strip())
            for t in str(cell).replace("|", ",").split(",") if t.strip()}


@st.cache_data(show_spinner=False)
def load_recipient_spoids(filename):
    """Set of SPOIDs that received at least one sent eNotice in the file."""
    df = load_enotices(filename)
    recips = set()
    if "recipient_SPOIDs" in df.columns:
        for cell in df["recipient_SPOIDs"]:
            recips |= parse_recipient_spoids(cell)
    return recips


def filter_dataframe(df):
    """Render an expander of per-column filters and return the filtered frame.

    Each chosen column gets a widget matched to its cardinality: a multiselect
    of distinct values for low-cardinality columns, or a case-insensitive
    substring text input otherwise. Filters combine with AND.
    """
    filtered = df
    with st.expander("🔎 Column filters"):
        to_filter = st.multiselect("Filter by columns", list(df.columns))
        for col in to_filter:
            left, right = st.columns((1, 20))
            left.markdown("↳")
            series = filtered[col].astype(str)
            if series.nunique() <= 50:
                choices = right.multiselect(
                    f"Values for “{col}”", sorted(series.unique()),
                    key=f"filt_ms_{col}")
                if choices:
                    filtered = filtered[series.isin(choices)]
            else:
                text = right.text_input(f"Substring in “{col}”",
                                        key=f"filt_tx_{col}")
                if text:
                    filtered = filtered[
                        series.str.contains(text, case=False, na=False)]
    return filtered


def matches_recipients(cell, selected):
    """True if any selected SPOID appears in a recipient_SPOIDs cell."""
    return bool(parse_recipient_spoids(cell) & selected)


# --------------------------------------------------------------------------- #
# AI summary agent (Anthropic; modeled on the IEEE Section Operations Assistant)

_SUMMARY_SYSTEM = (
    "You summarize IEEE eNotices for a member digest. Given the text of an "
    "eNotice (the primary source) and optionally a related event page (a "
    "secondary source), write a neutral, factual 2-3 sentence summary of what "
    "the notice is about and its most important details -- the event or action, "
    "any key dates or deadlines, and what the reader is asked to do. Base the "
    "summary only on the provided content; do not invent details. No marketing "
    "language, greeting, or closing. Output only the summary sentences."
)


@st.cache_data(show_spinner=False, ttl=86400)
def _fetch_raw(url):
    """Fetch a URL and return its raw HTML, cached. '' on error/non-200."""
    url = (url or "").strip()
    if not url:
        return ""
    try:
        resp = _HTTP.request("GET", url, timeout=_FETCH_TIMEOUT)
    except Exception:
        return ""
    if resp.status != 200 or not resp.data:
        return ""
    return resp.data.decode("utf-8", errors="replace")


def _strip_html(doc, cap=6000):
    """Visible text of an HTML fragment (scripts/markup stripped, capped)."""
    doc = doc or ""
    doc = re.sub(r"(?is)<script.*?</script>", " ", doc)
    doc = re.sub(r"(?is)<style.*?</style>", " ", doc)
    text = html.unescape(re.sub(r"(?s)<[^>]+>", " ", doc))
    return re.sub(r"\s+", " ", text).strip()[:cap]


def _fetch_text(url, cap=6000):
    """Visible text of a page (scripts/markup stripped, capped)."""
    return _strip_html(_fetch_raw(url), cap=cap)


def _event_id_from_url(event_url):
    """The numeric event id embedded in a vTools event URL (…/m/<id>), or ''."""
    m = re.search(r"/m/(\d+)", str(event_url or ""))
    return m.group(1) if m else ""


@st.cache_data(show_spinner=False, ttl=86400)
def event_api(event_id):
    """Structured event data from the vTools Events API, keyed by event id.

    Returns a normalized dict (empty on any failure) with the fields the digest
    needs: title, text (description + agenda, markup stripped), tags, category,
    city/state/country, location_type, and image URL. Preferred over scraping
    the event_url HTML page. Cached per id so each event is fetched once.
    """
    event_id = str(event_id or "").strip()
    if not event_id:
        return {}
    raw = _fetch_raw(_EVENTS_API.format(event_id))
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
        rec = (payload.get("data") or [None])[0]
        if not rec:
            return {}
        a = rec.get("attributes", {}) or {}
        included = payload.get("included", []) or []
    except Exception:
        return {}

    # Resolve related records (category name, state/country names) from the
    # JSON:API "included" list.
    by_ref = {(i.get("type"), str(i.get("id"))): (i.get("attributes") or {})
              for i in included}

    rels = rec.get("relationships", {}) or {}

    def _related(rel, *name_keys):
        data = ((rels.get(rel) or {}).get("data")) or None
        if not data:
            return ""
        attrs = by_ref.get((data.get("type"), str(data.get("id"))), {})
        for k in name_keys:
            if attrs.get(k):
                return str(attrs[k])
        return ""

    tags = a.get("tags")
    if not isinstance(tags, list):
        tags = str(a.get("keywords") or "").split()

    text = " ".join(t for t in (_strip_html(a.get("description") or "", 4000),
                                _strip_html(a.get("agenda") or "", 1000)) if t)

    return {
        "title": str(a.get("title") or "").strip(),
        "text": text,
        "tags": [str(t) for t in tags if str(t).strip()],
        "category": _related("category", "name"),
        "city": str(a.get("city") or "").strip(),
        "state": _related("state", "name"),
        "country": _related("country", "name"),
        "location_type": str(a.get("location-type") or "").strip().lower(),
        "image": str(a.get("image") or "").strip(),
    }


def _event_content_block(ev):
    """A compact text block describing an event, for the summary/tag agents."""
    if not ev:
        return ""
    parts = []
    if ev.get("title"):
        parts.append(f"Event title: {ev['title']}")
    loc_bits = [b for b in (ev.get("city"), ev.get("state"), ev.get("country"))
                if b]
    if ev.get("location_type"):
        loc = ev["location_type"]
        if loc_bits:
            loc += " -- " + ", ".join(loc_bits)
        parts.append(f"Location: {loc}")
    if ev.get("category"):
        parts.append(f"Event category: {ev['category']}")
    if ev.get("tags"):
        parts.append("Event tags: " + " ".join(ev["tags"]))
    if ev.get("text"):
        parts.append(f"Event details: {ev['text']}")
    return "\n".join(parts)


@st.cache_data(show_spinner=False, ttl=86400)
def enotice_summary(public_url, event_id):
    """A 2-3 sentence AI summary of an eNotice from its public page (primary)
    and its event's structured API record (secondary). Returns None when
    unavailable (no key, fetch or LLM failure) so the caller can fall back to
    placeholder text.

    Cached per (public_url, event_id) so each notice is summarized once.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    primary = _fetch_text(public_url)
    if not primary:
        return None
    content = f"eNotice page ({public_url}):\n{primary}"
    secondary = _event_content_block(event_api(event_id)) if event_id else ""
    if secondary:
        content += f"\n\nRelated event:\n{secondary}"
    try:
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=_ANTHROPIC_MODEL,
            max_tokens=_SUMMARY_MAX_TOKENS,
            system=_SUMMARY_SYSTEM,
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(b.text for b in msg.content
                       if getattr(b, "type", "") == "text").strip()
        return text or None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# AI tag agent: topical hashtags for an eNotice tile (rules 1-6)

_TAG_SYSTEM = (
    "You analyze an IEEE eNotice shown in a member digest to help tag it. You "
    "are given the eNotice's subject line, its short summary, and (when present) "
    "the associated event's own tags. Respond with a single JSON object and "
    "nothing else, with exactly these keys:\n"
    '- "category": the ONE best-matching category from the allowed list below, '
    "copied verbatim, if one clearly applies -- otherwise null.\n"
    '- "relevant_event_tags": an array (empty if none, never null) containing '
    "the subset of the event's own tags listed in the content that are closely "
    "related to the eNotice's subject matter, copied verbatim. Exclude organizer "
    "or person names, host/section/chapter codes, and anything not about the "
    "topic. If there is no related event, use an empty array.\n"
    '- "geography": the single most specific place the notice is tied to (for '
    'example just the city of an in-person event, as "Santiago"), or null if it '
    "is not tied to a physical place (for example an online-only notice).\n"
    '- "technical_topics": an array of the specific technical, engineering, or '
    'scientific subject areas the notice is about (for example "machine '
    'learning", "electric vehicles", "power systems"), or an empty array if the '
    "notice is not technical.\n"
    '- "conference_short": if the notice is about a specific conference or '
    "symposium that has a short name or acronym, that short name -- excluding "
    'any location/city suffix but keeping a year if present (for example '
    '"ISIE2026" from "ISIE2026-Nagoya"); otherwise null.\n'
    '- "conference_long": if the notice is about a specific conference or '
    "symposium but only a full name is available, that full name with any "
    'leading "IEEE" and any leading ordinal such as "35th" removed (for example '
    '"International Symposium on Industrial Electronics"); otherwise null.\n'
    "Base everything only on the provided content; do not invent details."
)


def _camel_tag(text):
    """Normalize a phrase to a camel-case tag body (no leading '#').

    Splits on any run of non-alphanumerics, upper-cases each token's first
    letter while preserving all-caps acronyms, and joins with no separators:
    'greenhouse-gas' -> 'GreenhouseGas', 'STEM' -> 'STEM', 'Aerospace control'
    -> 'AerospaceControl'. Returns '' when nothing usable remains.
    """
    tokens = re.findall(r"[0-9A-Za-z]+", str(text or ""))
    return "".join(t if t.isupper() else t[:1].upper() + t[1:] for t in tokens)


def _dedup_ci(items):
    """De-duplicate tag bodies case-insensitively, preserving first-seen order."""
    out, seen = [], set()
    for it in items:
        k = it.lower()
        if it and k not in seen:
            seen.add(k)
            out.append(it)
    return out


@st.cache_data(show_spinner=False)
def load_tag_categories():
    """The controlled list of eNotice categories (rule 3), from CSV. A missing
    file yields [] (rule 3 simply never fires)."""
    try:
        with open(_TAG_CATEGORIES_PATH, newline="", encoding="utf-8") as fh:
            return [c.strip() for r in csv.DictReader(fh)
                    if (c := (r.get("category") or "").strip())]
    except OSError:
        return []


@st.cache_data(show_spinner=False)
def load_taxonomy():
    """IEEE taxonomy terms for rule 5, as a list of (term_lower, path_terms)
    sorted longest-term-first so the most specific match wins. path_terms is the
    term's full_path (root..leaf) of display terms. Empty if the file is
    absent."""
    try:
        with open(_TAXONOMY_PATH, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return []
    out, seen = [], set()
    for r in rows:
        term = (r.get("term") or "").strip()
        if not term or term.lower() in seen:
            continue
        seen.add(term.lower())
        segs = [s.strip() for s in (r.get("full_path") or "").split(">")
                if s.strip()] or [term]
        out.append((term.lower(), segs))
    out.sort(key=lambda ts: len(ts[0]), reverse=True)
    return out


def _match_taxonomy(topics_text, content_text, limit=_TAXONOMY_MAX_TAGS):
    """Match IEEE taxonomy terms for an eNotice's technical topics (rule 5),
    leaning toward higher-level terms.

    Lexically finds taxonomy terms in `topics_text` (the model's technical
    topics), then for each taxonomy family it (a) consolidates two or more
    matches into their deepest shared ancestor -- one higher-level term that
    covers them -- and (b) generalizes any lower-level term (taxonomy depth
    >= `_TAX_DEEP_LEVEL`) that is not named verbatim in `content_text` up to a
    broader ancestor. Returns (leaf_terms, ancestor_terms): up to `limit`
    representative terms that count toward the tag limit, plus the broader terms
    on their paths (exempt)."""
    topics_low = f" {topics_text.lower()} "
    content_low = f" {content_text.lower()} "

    # 1. Lexical hits, longest-first, dropping terms contained in a longer match.
    hits = []  # (term_lower, segs)
    for term_lower, segs in load_taxonomy():
        if len(term_lower) < 4:
            continue
        if any(term_lower in k for k, _ in hits):
            continue
        if re.search(r"\b" + re.escape(term_lower) + r"\b", topics_low):
            hits.append((term_lower, segs))
            if len(hits) >= 12:
                break
    if not hits:
        return [], []

    def _mentioned(term):
        return bool(re.search(r"\b" + re.escape(term.lower()) + r"\b", content_low))

    def _generalize(segs):
        # Climb toward the root while the term is lower-level and unmentioned.
        segs = list(segs)
        while len(segs) - 1 >= _TAX_DEEP_LEVEL and not _mentioned(segs[-1]):
            segs = segs[:-1]
        return segs

    # 2. One representative term per taxonomy family.
    families = {}  # family term -> list of segs
    for _, segs in hits:
        families.setdefault(segs[0], []).append(segs)

    reps = []  # (rep_segs, group_size, most_specific_match_len)
    for group in families.values():
        if len(group) >= 2:
            prefix = group[0]  # deepest common ancestor = longest common prefix
            for segs in group[1:]:
                n = 0
                while n < len(prefix) and n < len(segs) and prefix[n] == segs[n]:
                    n += 1
                prefix = prefix[:n]
            rep = _generalize(prefix)
        else:
            rep = _generalize(group[0])
        reps.append((rep, len(group), max(len(s[-1]) for s in group)))

    # 3. Prefer families with more matches (then more specific), then cap.
    reps.sort(key=lambda r: (r[1], r[2]), reverse=True)
    leaves, ancestors = [], []
    for rep_segs, _, _ in reps[:limit]:
        if not rep_segs:
            continue
        if rep_segs[-1] not in leaves:
            leaves.append(rep_segs[-1])
            ancestors.extend(a for a in rep_segs[:-1] if a not in ancestors)
    ancestors = [a for a in ancestors if a not in leaves]
    return leaves, ancestors


def _tag_llm(content, categories):
    """Ask the Claude agent for a category, relevant event tags, geography,
    technical topics, and conference name(s) as a JSON object. {} on error."""
    try:
        import anthropic
        client = anthropic.Anthropic()
        allowed = ", ".join(categories) if categories else "(none provided)"
        msg = client.messages.create(
            model=_ANTHROPIC_MODEL,
            max_tokens=_TAG_MAX_TOKENS,
            system=_TAG_SYSTEM + "\n\nAllowed categories: " + allowed,
            messages=[{"role": "user", "content": content}],
        )
        raw = "".join(b.text for b in msg.content
                      if getattr(b, "type", "") == "text")
        m = re.search(r"\{.*\}", raw, re.S)
        return json.loads(m.group(0)) if m else {}
    except Exception:
        return {}


# Provenance descriptions shown when hovering a tag. Each explains whether the
# tag was AI-generated (and by which rule) or imported from the associated event.
_TAG_SRC_CATEGORY = "AI-generated: matched eNotice category (rule 3)"
_TAG_SRC_EVENT = "Imported from the associated event's tags (rule 4)"
_TAG_SRC_TAXONOMY = "AI-generated: matched IEEE Taxonomy term (rule 5)"
_TAG_SRC_TAXONOMY_ANCESTOR = ("AI-generated: broader IEEE Taxonomy category of a "
                              "matched term (rule 5)")
_TAG_SRC_GEO_AI = "AI-generated: location associated with the eNotice (rule 6)"
_TAG_SRC_GEO_EVENT = "From the associated event's location (rule 6)"
_TAG_SRC_CONFERENCE = "AI-generated: conference or symposium name (rule 7)"


def _conference_tag(short, long):
    """A tag body for a conference/symposium the eNotice is about (rule 7).

    Prefers a short name/acronym, dropping a trailing location suffix but keeping
    a year (``ISIE2026-Nagoya`` -> ``ISIE2026``). Otherwise shortens the long
    name by stripping a leading ``IEEE`` and an ordinal such as ``35th``
    (``IEEE 35th International Symposium on Industrial Electronics`` ->
    ``InternationalSymposiumOnIndustrialElectronics``). '' if neither applies.
    """
    short = (short or "").strip()
    if short:
        # Drop a trailing location word (a separator + purely alphabetic token);
        # a year stays because it contains digits.
        short = re.sub(r"[-–\s]+[A-Za-z]+$", "", short).strip()
        return _camel_tag(short)
    long = (long or "").strip()
    if not long:
        return ""
    prev = None
    while prev != long:  # strip a leading "IEEE" and/or ordinal, in any order
        prev = long
        long = re.sub(r"^\s*IEEE\b\.?\s*", "", long, flags=re.I)
        long = re.sub(r"^\s*\d+\s*(?:st|nd|rd|th)\b\.?\s*", "", long, flags=re.I)
    return _camel_tag(long)


def _dedup_tags(pairs):
    """De-duplicate (tag_body, source) pairs case-insensitively, treating an
    ``IEEE``-prefixed variant as a duplicate of the bare tag and keeping the
    shorter body (rule 7: ``IEEESoutheastCon2025`` / ``SoutheastCon2025`` ->
    ``SoutheastCon2025``). Exact duplicates keep the first (highest-priority)
    occurrence and its source."""
    out, pos = [], {}  # out: [body, source]; pos: dedup key -> index in out
    for body, source in pairs:
        if not body:
            continue
        low = body.lower()
        key = re.sub(r"^ieee", "", low) or low
        if key in pos:
            i = pos[key]
            if len(body) < len(out[i][0]):  # prefer the shorter variant
                out[i] = [body, source]
            continue
        pos[key] = len(out)
        out.append([body, source])
    return [(b, s) for b, s in out]


def _filter_event_tags(event_tags, relevant):
    """Keep only the event's own tags the model judged closely related to the
    content (rule 1's relevance requirement applied to rule 4). `relevant` is the
    model's verbatim subset; None (field absent / parse failure) keeps all tags
    as a fallback, while a provided list keeps the intersection -- matched by
    normalized form so the original tag text is preserved."""
    if not event_tags:
        return []
    if relevant is None:
        return list(event_tags)
    keep = {_camel_tag(t).lower() for t in relevant if str(t).strip()}
    return [t for t in event_tags if _camel_tag(t).lower() in keep]


@st.cache_data(show_spinner=False, ttl=86400)
def enotice_tags(subject, summary, event_id):
    """Topical tags for an eNotice tile, or None to fall back to placeholders.

    Tags are generated from what the reader actually sees -- the eNotice
    `subject`, its 2-3 sentence `summary`, and the associated event's own tags --
    rather than the full page text, so they track the summary and subject.

    Returns a list of (tag_body, provenance) pairs, where provenance explains on
    hover whether the tag was AI-generated (and by which rule) or imported from
    the associated event. Every tag comes from one of rules 3-7 and must be
    closely related to the content (rule 1): the matched category (rule 3), the
    event's own relevant tags (rule 4), IEEE taxonomy terms + ancestors (rule 5),
    a geography (rule 6), and a conference/symposium name (rule 7). Category,
    taxonomy leaves, geography, and the conference name count toward
    `_DIGEST_MAX_TAGS`; event tags and taxonomy ancestors do not.

    Cached per (subject, summary, event_id) so each notice is tagged once.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    subject = (subject or "").strip()
    summary = (summary or "").strip()
    ev = event_api(event_id) if event_id else {}
    if not subject and not summary and not ev:
        return None

    categories = load_tag_categories()
    parts = []
    if subject:
        parts.append(f"eNotice subject: {subject}")
    if summary:
        parts.append(f"eNotice summary: {summary}")
    if ev.get("tags"):
        parts.append("Associated event's own tags: " + " ".join(ev["tags"]))
    data = _tag_llm("\n\n".join(parts), categories)

    # Every tag must satisfy one of rules 3-7 (rule 1). Counted tags (category,
    # conference, geography, taxonomy leaves) fill up to the six-tag cap.
    counted = []  # (body, source)

    cat = (data.get("category") or "").strip()
    if cat and any(cat.lower() == c.lower() for c in categories):
        counted.append((_camel_tag(cat), _TAG_SRC_CATEGORY))

    conf = _conference_tag(data.get("conference_short"),
                           data.get("conference_long"))
    if conf:
        counted.append((conf, _TAG_SRC_CONFERENCE))

    geo = ""
    geo_source = _TAG_SRC_GEO_AI
    if ev.get("location_type") in ("physical", "hybrid"):
        geo = ev.get("city") or ev.get("state") or ev.get("country") or ""
        if geo:
            geo_source = _TAG_SRC_GEO_EVENT
    geo = geo or (data.get("geography") or "").strip()
    geo = geo.split(",")[0].strip()  # keep the most specific part (the city)
    if geo:
        counted.append((_camel_tag(geo), geo_source))

    # Rule 5: match taxonomy terms against the model's technical topics (anchored
    # there rather than free text to avoid matching generic words). If the model
    # found no topics but the event is categorized Technical, fall back to the
    # subject. The "mentioned in content" check uses only the subject, summary,
    # and event tags.
    topics = [str(t) for t in (data.get("technical_topics") or []) if str(t)]
    scan = " ".join(topics)
    if not scan and ev.get("category", "").lower() == "technical":
        scan = subject
    ancestors = []
    if scan:
        content_text = " ".join(x for x in (subject, summary,
                                            " ".join(ev.get("tags", []))) if x)
        leaves, ancestors = _match_taxonomy(scan, content_text)
        counted.extend((_camel_tag(t), _TAG_SRC_TAXONOMY) for t in leaves)

    counted = [(b, s) for b, s in counted if b]
    counted = _dedup_tags(counted)[:_DIGEST_MAX_TAGS]

    # Exempt tags: the event's own relevant tags (rule 4) and taxonomy ancestors
    # (rule 5).
    relevant = _filter_event_tags(ev.get("tags", []),
                                  data.get("relevant_event_tags"))
    exempt = [(_camel_tag(t), _TAG_SRC_EVENT) for t in relevant]
    exempt += [(_camel_tag(t), _TAG_SRC_TAXONOMY_ANCESTOR) for t in ancestors]

    return _dedup_tags(counted + [(b, s) for b, s in exempt if b]) or None


# --------------------------------------------------------------------------- #
# Tile image agent: use a suitable picture from the page, else generate one

def _content_image_candidates(html_doc, base_url):
    """Absolute URLs of an eNotice/event page's content images (chrome removed).

    Uploaded content images (…/vtools_ui/media/display/…) are preferred first.
    """
    out, seen = [], set()
    for src in re.findall(r'<img[^>]+src=["\']([^"\']+)', html_doc, re.I):
        absu = urljoin(base_url, src)
        low = absu.lower()
        if any(p in low for p in _CHROME_IMG) or absu in seen:
            continue
        seen.add(absu)
        out.append(absu)
    out.sort(key=lambda u: 0 if "media/display" in u.lower() else 1)
    return out


def _image_is_suitable(img_url, subject):
    """Ask the Claude agent (vision) whether an image is a relevant content
    picture (not a logo/banner/icon/map). Returns False on any error."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=_ANTHROPIC_MODEL,
            max_tokens=5,
            system=("You choose a thumbnail for an IEEE eNotice. Decide whether "
                    "the given image is a suitable content picture -- a relevant "
                    "photo, event flyer, or topical graphic -- and NOT a logo, "
                    "generic banner or email template, icon, map, QR code, "
                    "social button, or purely decorative element. Answer with "
                    "only YES or NO."),
            messages=[{"role": "user", "content": [
                {"type": "image",
                 "source": {"type": "url", "url": img_url}},
                {"type": "text",
                 "text": f"eNotice subject: {subject}\nIs this a suitable "
                         "content picture? Answer YES or NO."},
            ]}],
        )
        ans = "".join(b.text for b in msg.content
                      if getattr(b, "type", "") == "text").strip().upper()
        return ans.startswith("Y")
    except Exception:
        return False


def _generate_image(subject, context):
    """Generate a realistic thumbnail with OpenAI when no page image fits.

    The Claude agent writes the prompt; OpenAI renders it. Returns a data-URI or
    None (no OpenAI key, or any failure)."""
    if not os.getenv("OPENAI_API_KEY"):
        return None
    prompt = None
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model=_ANTHROPIC_MODEL,
                max_tokens=120,
                system=("Write ONE concise image-generation prompt (a single "
                        "sentence) for a realistic, professional thumbnail that "
                        "visually represents this IEEE eNotice's topic or event. "
                        "No text, words, letters, logos, watermarks, or "
                        "identifiable real people. Output only the prompt."),
                messages=[{"role": "user",
                           "content": f"Subject: {subject}\n\n{context}"}],
            )
            prompt = "".join(b.text for b in msg.content
                             if getattr(b, "type", "") == "text").strip()
        except Exception:
            prompt = None
    if not prompt:
        prompt = (f"A realistic, professional photo representing the topic of an "
                  f"IEEE event titled '{subject}'. No text or logos.")
    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.images.generate(
            model=_IMAGE_MODEL, prompt=prompt,
            size=_IMAGE_SIZE, quality=_IMAGE_QUALITY, n=1)
        b64 = resp.data[0].b64_json
        return f"data:image/png;base64,{b64}" if b64 else None
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=86400)
def enotice_image(public_url, event_id, subject):
    """An image src for a digest tile, or None.

    Prefers a suitable content picture from the public page's images, then the
    event's own image from the Events API (both confirmed by the vision agent);
    otherwise generates one. Cached per notice so pages aren't re-scanned and
    images aren't re-generated on rerun.
    """
    raw = _fetch_raw(public_url)
    if raw:
        for cand in _content_image_candidates(raw, public_url)[:2]:
            if _image_is_suitable(cand, subject):
                return cand
    ev = event_api(event_id) if event_id else {}
    if ev.get("image") and _image_is_suitable(ev["image"], subject):
        return ev["image"]
    return _generate_image(subject, _fetch_text(public_url)[:1500])


# --------------------------------------------------------------------------- #
# Digest view helpers

_URL_RE = re.compile(r"https?://[^\s<]+")
# Trailing sentence punctuation / stray HTML entities to keep OUT of a link
# (but not &amp;, which is a real URL query separator).
_URL_TRAIL_RE = re.compile(r"(&(?:quot|gt|lt|#x27|#39);|[.,;:!?)\]}])$")


def _linkify(text):
    """HTML-escape text, then turn bare http(s) URLs into clickable links."""
    def repl(m):
        url, trail = m.group(0), ""
        while True:
            t = _URL_TRAIL_RE.search(url)
            if not t:
                break
            trail = t.group(0) + trail
            url = url[:-len(t.group(0))]
        if not url:
            return m.group(0)
        return f'<a href="{url}" target="_blank">{url}</a>{trail}'
    return _URL_RE.sub(repl, html.escape(text))


def _fmt_full_date(d):
    """'Thursday, November 20, 2025' (no leading zero on the day)."""
    return f"{d.strftime('%A, %B')} {d.day}, {d.year}"


def _fmt_date(d):
    """'November 20, 2025'."""
    return f"{d.strftime('%B')} {d.day}, {d.year}"


def _unit_name(spoid, unit_info):
    ou = unit_info.get(resolve_spoid(spoid))
    return ou.name if ou is not None and ou.name else spoid


def _unit_link(spoid, unit_info):
    """A unit name linked to its website, or plain text if no URL is known."""
    name = html.escape(_unit_name(spoid, unit_info))
    ou = unit_info.get(resolve_spoid(spoid))
    url = (ou.url or "").strip() if ou is not None else ""
    if url:
        return f'<a href="{html.escape(url)}" target="_blank">{name}</a>'
    return name


def _join_units(spoids, unit_info):
    """Comma-separated unit links with an 'and' before the last, name-sorted."""
    ordered = sorted(set(spoids), key=lambda s: _unit_name(s, unit_info).lower())
    links = [_unit_link(s, unit_info) for s in ordered]
    if len(links) <= 1:
        return "".join(links)
    if len(links) == 2:
        return " and ".join(links)
    return ", ".join(links[:-1]) + ", and " + links[-1]


def _go_digest():
    st.session_state.view = "digest"


def _go_prefs():
    st.session_state.view = "preferences"


# View picker: one control (upper-left of every page) replacing the old
# cross-links. Labels map to the internal `view` session-state values. The
# Daily/Weekly/Monthly digest views appear only when the member has assigned
# that frequency to at least one included unit (see _active_digest_freqs).
_VIEW_LABELS = {
    "archive": "Table View",
    "digest": "eNotice Archive",
    "daily": "Daily Digest",
    "weekly": "Weekly Digest",
    "monthly": "Monthly Digest",
    "preferences": "Preferences Center",
}
_VIEW_BY_LABEL = {label: view for view, label in _VIEW_LABELS.items()}
# Frequency label (as stored in preferences) -> its view key, in picker order.
_FREQ_VIEW = {"Daily Digest": "daily", "Weekly Digest": "weekly",
              "Monthly Digest": "monthly"}
_VIEW_FREQ = {view: freq for freq, view in _FREQ_VIEW.items()}


def _active_digest_freqs(selected_norm):
    """{frequency_label: {units}} for each digest frequency that has at least
    one *included* unit assigned to it in the saved preferences. Immediate units
    (individual emails) and fully-excluded units are not counted."""
    saved = _saved_prefs(selected_norm)
    freqs = saved.get("frequencies", {})
    included = saved.get("units")
    out = {}
    for u in selected_norm:
        if included is not None and u not in included:
            continue
        f = freqs.get(u)
        if f in _FREQ_VIEW:
            out.setdefault(f, set()).add(u)
    return out


def _on_view_pick():
    label = st.session_state.get("view_picker")
    if label in _VIEW_BY_LABEL:
        st.session_state.view = _VIEW_BY_LABEL[label]


def _render_view_picker(selected_norm):
    """A segmented view switcher shown at the upper-left of every page. The
    Daily/Weekly/Monthly options are included only when active."""
    active = _active_digest_freqs(selected_norm)
    options = ["eNotice Archive", "Preferences Center"]
    options += [f for f in _FREQ_VIEW if f in active]  # Daily, Weekly, Monthly
    options.append("Table View")
    cur = _VIEW_LABELS.get(st.session_state.get("view", "digest"),
                           "eNotice Archive")
    if cur not in options:  # e.g. a digest view whose frequency is now inactive
        cur = "eNotice Archive"
    # Keep the widget in sync when the view changes by other means (e.g. Save).
    if st.session_state.get("view_picker") != cur:
        st.session_state["view_picker"] = cur
    c, _ = st.columns([5, 2])
    with c:
        st.segmented_control(
            "View", options, key="view_picker",
            on_change=_on_view_pick, label_visibility="collapsed")


def _digest_prev():
    st.session_state["digest_page"] = max(
        0, st.session_state.get("digest_page", 0) - 1)


def _digest_next():
    st.session_state["digest_page"] = st.session_state.get("digest_page", 0) + 1


# --------------------------------------------------------------------------- #
# Preferences center (member-facing "Manage your digest settings" page)

_FREQ_OPTIONS = ["Immediate", "Daily Digest", "Weekly Digest", "Monthly Digest"]


def _pref_sig(selected_norm):
    """Stable key for the current sidebar unit selection, so saved preferences
    reset automatically when the selection changes."""
    return "|".join(sorted(selected_norm))


def _saved_prefs(selected_norm):
    """The saved preferences dict for the current selection ({} if none)."""
    return st.session_state.get("saved_prefs", {}).get(
        _pref_sig(selected_norm), {})


def _seen_tags(selected_norm):
    """The set of tag bodies seen so far in the current digest display, as a
    {lowercased: display} dict. It accumulates as digest tiles are rendered
    (see `_render_tile_page`) and resets whenever the sidebar selection changes,
    so the Tags-to-Exclude picker offers exactly the tags present in the
    member's current digest."""
    sig = _pref_sig(selected_norm)
    if st.session_state.get("seen_tags_sig") != sig:
        st.session_state["seen_tags_sig"] = sig
        st.session_state["seen_tags"] = {}
    return st.session_state["seen_tags"]


def _monthly_rates(df, selected_norm):
    """Expected eNotices per month per selected unit, from the whole data file:
    (that unit's received count across the file) / (months the file spans).
    Returns ({spoid: emails_per_month_float}, months)."""
    dates = df["_sent_dt"].dropna()
    if dates.empty:
        months = 1
    else:
        months = max(1, round((dates.max() - dates.min()).days / 30.44))
    counts = Counter()
    for cell in df["recipient_SPOIDs"]:
        for u in parse_recipient_spoids(cell) & selected_norm:
            counts[u] += 1
    return {u: counts.get(u, 0) / months for u in selected_norm}, months


def _fmt_rate(x):
    """Format an emails-per-month rate: 0, '<1' for a small nonzero, else int."""
    if x <= 0:
        return "0"
    if x < 0.5:
        return "<1"
    return str(round(x))


def _expected_panel_html(rates, included):
    """Right-column 'Expected Emails' panel for the included units.

    The total is the sum of the per-unit rounded rates (not a separate rounding
    of the raw sum) so the breakdown always adds up to the headline number."""
    total = sum(round(rates.get(u, 0)) for u in included)
    rows = "".join(
        f'<div class="pref-row"><span class="pref-name">{html.escape(name)}</span>'
        f'<span class="pref-count">{_fmt_rate(rates.get(u, 0))}</span></div>'
        for u, name in sorted(included.items(),
                              key=lambda kv: rates.get(kv[0], 0), reverse=True))
    rows = rows or '<div class="pref-sub">No units included.</div>'
    n = len(included)
    return (
        '<div class="pref-panel">'
        '<div class="pref-panel-title">Expected Emails</div>'
        '<div class="pref-sub">eNotices per month if you choose Immediate '
        'delivery</div>'
        f'<div class="pref-total">{_fmt_rate(total)}</div>'
        f'<div class="pref-sub">{n} Organizational Unit'
        f'{"" if n == 1 else "s"} included</div>'
        '<div class="pref-breakdown-title">Breakdown by Organizational Unit</div>'
        f'{rows}'
        '</div>')


# --------------------------------------------------------------------------- #
# IEEE Account portal chrome (header + footer), modeled on
# ieee.org/ieee-privacyportal, so the member-facing preferences page reads like
# the real IEEE communication-preferences experience.

# The header shows the signed-in member; per request it is a literal placeholder.
_MEMBER_NAME = "[MEMBER NAME]"

# Top utility bar items (left group). Rendered as plain text -- the header
# deliberately carries no hyperlinks. "Xplore" is italicized like the portal.
_IEEE_TOPNAV_LEFT = [
    '<span class="ieee-home">⌂</span>&nbsp;IEEE.org',
    'IEEE <em>Xplore</em> Digital Library',
    'IEEE Standards',
    'IEEE Spectrum',
    'More Sites',
]

_IEEE_ABOUT_BLURB = (
    "IEEE is the world's largest technical professional organization dedicated "
    "to advancing technology for the benefit of humanity.")

# Blue footer columns: (heading, blurb-or-None, [(label, href), ...]).
_IEEE_FOOTER_COLS = [
    ("About IEEE", _IEEE_ABOUT_BLURB, [
        ("Learn more about IEEE", "https://www.ieee.org/about/index.html"),
        ("IEEE Mission & Vision",
         "https://www.ieee.org/about/vision-mission.html"),
    ]),
    ("Membership", None, [
        ("Join", "https://www.ieee.org/membership/join/index.html"),
        ("Renew", "https://www.ieee.org/membership/renew.html"),
        ("Benefits", "https://www.ieee.org/membership/benefits.html"),
        ("IEEE Collabratec", "https://ieee-collabratec.ieee.org"),
    ]),
    ("Get involved", None, [
        ("Conferences", "https://www.ieee.org/conferences/index.html"),
        ("Local activities", "https://www.ieee.org/communities/index.html"),
        ("Publishing", "https://www.ieee.org/publications/index.html"),
        ("Societies", "https://www.ieee.org/communities/societies/index.html"),
        ("Councils", "https://www.ieee.org/communities/technical-councils.html"),
        ("Technical careers", "https://careers.ieee.org"),
        ("Volunteer", "https://www.ieee.org/volunteers/index.html"),
    ]),
]
# "Connect with IEEE" column: an outlined Contact & Support button, then links.
_IEEE_CONNECT = [
    ("IEEE Collabratec", "https://ieee-collabratec.ieee.org"),
    ("Careers at IEEE", "https://careers.ieee.org"),
    ("IEEE Newsroom", "https://www.ieee.org/about/news/index.html"),
    ("IEEE Media Kit", "https://www.ieee.org/about/news/media-kit.html"),
]
# Social icons (name, href, inline SVG path). White glyphs on the blue footer.
_IEEE_SOCIAL = [
    ("Facebook", "https://www.facebook.com/IEEE.org",
     "M13 22v-8h2.7l.4-3H13V9.1c0-.9.3-1.5 1.6-1.5H16V5c-.3 0-1.2-.1-2.2-.1-2.2 "
     "0-3.8 1.4-3.8 3.9V11H7.5v3H10v8h3z"),
    ("Twitter", "https://twitter.com/IEEEorg",
     "M23 4.9c-.8.4-1.7.6-2.6.8 1-.6 1.6-1.5 2-2.5-.9.5-1.9.9-2.9 1.1C18.6 3.4 "
     "17.4 3 16.1 3c-2.5 0-4.5 2-4.5 4.5 0 .4 0 .7.1 1C8 8.3 5 6.6 3 3.9c-.4.7-.6 "
     "1.5-.6 2.3 0 1.6.8 3 2 3.8-.7 0-1.4-.2-2-.5v.1c0 2.2 1.5 4 3.6 4.4-.4.1-.8."
     "2-1.2.2-.3 0-.6 0-.8-.1.6 1.8 2.3 3.1 4.3 3.2-1.6 1.2-3.5 2-5.7 2-.4 0-.7 "
     "0-1.1-.1C3.9 20.3 6.2 21 8.7 21c8.3 0 12.8-6.9 12.8-12.8v-.6c.9-.6 1.6-1.4 "
     "2.2-2.3z"),
    ("LinkedIn", "https://www.linkedin.com/company/ieee",
     "M6.9 8.5H3.9V21h3V8.5zM5.4 3.5C4.4 3.5 3.7 4.2 3.7 5.1c0 .9.7 1.6 1.6 1.6.9 "
     "0 1.6-.7 1.6-1.6 0-.9-.7-1.6-1.5-1.6zM20.3 21v-6.8c0-3.6-1.9-5.3-4.5-5.3-2.1 "
     "0-3 1.1-3.5 1.9V8.5H9.3V21h3v-6.9c0-.4 0-.7.1-1 .3-.7.9-1.4 1.9-1.4 1.4 0 "
     "1.9 1 1.9 2.6V21h3.1z"),
    ("YouTube", "https://www.youtube.com/ieee",
     "M23 12s0-3.2-.4-4.7c-.2-.8-.9-1.5-1.7-1.7C19.4 5.2 12 5.2 12 5.2s-7.4 0-8.9."
     "4c-.8.2-1.5.9-1.7 1.7C1 8.8 1 12 1 12s0 3.2.4 4.7c.2.8.9 1.5 1.7 1.7 1.5.4 "
     "8.9.4 8.9.4s7.4 0 8.9-.4c.8-.2 1.5-.9 1.7-1.7.4-1.5.4-4.7.4-4.7zM9.8 15.3V8.7"
     "l6 3.3-6 3.3z"),
    ("Instagram", "https://www.instagram.com/ieee_org",
     "M12 4.5c2.4 0 2.7 0 3.6.1 2.5.1 3.6 1.3 3.7 3.7 0 .9.1 1.2.1 3.6s0 2.7-.1 "
     "3.6c-.1 2.4-1.3 3.6-3.7 3.7-.9 0-1.2.1-3.6.1s-2.7 0-3.6-.1c-2.5-.1-3.6-1.3-"
     "3.7-3.7 0-.9-.1-1.2-.1-3.6s0-2.7.1-3.6C4.9 5.9 6 4.7 8.4 4.6c.9 0 1.2-.1 "
     "3.6-.1zM12 3c-2.4 0-2.8 0-3.7.1C5 3.2 3.2 5 3.1 8.3 3 9.2 3 9.6 3 12s0 2.8.1 "
     "3.7c.1 3.3 1.9 5.1 5.2 5.2.9 0 1.3.1 3.7.1s2.8 0 3.7-.1c3.3-.1 5.1-1.9 5.2-"
     "5.2 0-.9.1-1.3.1-3.7s0-2.8-.1-3.7c-.1-3.3-1.9-5.1-5.2-5.2C14.8 3 14.4 3 12 "
     "3zm0 4.4a4.6 4.6 0 100 9.2 4.6 4.6 0 000-9.2zm0 7.6a3 3 0 110-6 3 3 0 010 "
     "6zm4.8-7.8a1.1 1.1 0 100 2.2 1.1 1.1 0 000-2.2z"),
]
# Dark bottom legal strip: (label, href).
_IEEE_FOOTER_LEGAL = [
    ("Home", "https://www.ieee.org"),
    ("Sitemap", "https://www.ieee.org/sitemap.html"),
    ("Contact & Support", "https://supportcenter.ieee.org"),
    ("Accessibility", "https://www.ieee.org/accessibility.html"),
    ("NonDiscrimination Policy",
     "https://www.ieee.org/about/corporate/governance/p9-26.html"),
    ("Privacy & opting out of cookies",
     "https://www.ieee.org/security-privacy.html"),
    ("Feedback", "https://supportcenter.ieee.org"),
]


@st.cache_data(show_spinner=False)
def _logo_data_uri():
    """The IEEE logo as a data URI so it can be embedded in HTML chrome."""
    try:
        import base64
        data = base64.b64encode(_LOGO_PATH.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{data}"
    except OSError:
        return ""


def _render_ieee_header():
    """Dark utility bar, the white IEEE Account brand bar, and the breadcrumb.

    Modeled on the ieee.org/ieee-privacyportal header. Per request the header
    carries no hyperlinks, and the signed-in member is a literal placeholder."""
    bar = '<span class="ieee-tn-bar">|</span>'
    left = bar.join(f'<span>{s}</span>' for s in _IEEE_TOPNAV_LEFT)
    right = bar.join([
        '<span>Cart (0)</span>',
        f'<span class="ieee-tn-name">{html.escape(_MEMBER_NAME)}</span>',
        '<span>Sign Out</span>'])
    logo = _logo_data_uri()
    logo_html = (f'<img class="ieee-logo" src="{logo}" alt="IEEE">'
                 if logo else '<span class="ieee-account">IEEE</span>')
    crumbs = ["IEEE Account", "Profile", "Communication Preferences",
              "Local IEEE Communications"]
    crumb_html = '<span class="ieee-crumb-sep">›</span>'.join(
        f'<span class="ieee-crumb{" current" if i == len(crumbs) - 1 else ""}">'
        f'{html.escape(c)}</span>'
        for i, c in enumerate(crumbs))
    st.markdown(
        '<div class="ieee-topnav"><div class="ieee-chrome-inner ieee-tn-inner">'
        f'<div class="ieee-tn-left">{left}</div>'
        f'<div class="ieee-tn-right">{right}</div></div></div>'
        '<div class="ieee-brandbar"><div class="ieee-chrome-inner ieee-brand-inner">'
        f'<span class="ieee-account">IEEE Account</span>{logo_html}</div></div>'
        '<div class="ieee-breadcrumb"><div class="ieee-chrome-inner">'
        f'{crumb_html}</div></div>',
        unsafe_allow_html=True)


def _render_ieee_footer():
    """The blue IEEE footer (columns, Locations, socials) and the dark legal
    strip, modeled on the ieee.org/ieee-privacyportal footer."""
    def _links(items):
        return "".join(
            f'<li><a href="{html.escape(href)}" target="_blank">'
            f'{html.escape(label)}</a></li>' for label, href in items)

    cols = ""
    for head, blurb, items in _IEEE_FOOTER_COLS:
        blurb_html = (f'<p class="ieee-fblurb">{html.escape(blurb)}</p>'
                      if blurb else "")
        cols += (f'<div class="ieee-fcol"><h4>{html.escape(head)}</h4>'
                 f'{blurb_html}<ul>{_links(items)}</ul></div>')
    connect = (
        '<div class="ieee-fcol"><h4>Connect with IEEE</h4>'
        '<a class="ieee-contact-btn" href="https://supportcenter.ieee.org" '
        'target="_blank">Contact &amp; Support</a>'
        f'<ul>{_links(_IEEE_CONNECT)}</ul></div>')
    social = "".join(
        f'<a class="ieee-social" href="{html.escape(url)}" target="_blank" '
        f'aria-label="{html.escape(name)}"><svg viewBox="0 0 24 24" '
        f'fill="currentColor">{path}</svg></a>'
        for name, url, path in _IEEE_SOCIAL)
    locations = (
        '<div class="ieee-flocrow"><div class="ieee-floc">'
        '<h4>Locations</h4><p class="ieee-fblurb">IEEE has a global presence '
        'with seven offices internationally.</p>'
        '<ul><li><a href="https://www.ieee.org/about/contact-center.html" '
        'target="_blank">IEEE office locations</a></li></ul></div>'
        f'<div class="ieee-fsocials">{social}</div></div>')
    legal = '<span class="ieee-legal-bar">|</span>'.join(
        f'<a href="{html.escape(href)}" target="_blank">{html.escape(label)}</a>'
        for label, href in _IEEE_FOOTER_LEGAL)
    terms = ('<a href="https://www.ieee.org/about/help/site-terms-conditions.html" '
             'target="_blank">IEEE Terms and Conditions</a>')
    st.markdown(
        '<div class="ieee-footer"><div class="ieee-chrome-inner">'
        f'<div class="ieee-fcols">{cols}{connect}</div>{locations}'
        '</div></div>'
        '<div class="ieee-footer-legal"><div class="ieee-chrome-inner">'
        f'<div class="ieee-legal-links">{legal}</div>'
        f'<div class="ieee-copyright">© Copyright {date.today().year} IEEE — '
        f'All rights reserved. Use of this website signifies your agreement to '
        f'the {terms}.<br>A not-for-profit organization, IEEE is the world\'s '
        'largest technical professional organization dedicated to advancing '
        'technology for the benefit of humanity.</div>'
        '</div></div>',
        unsafe_allow_html=True)


def render_preferences_view(df, selected_norm, unit_info):
    """The member-facing preferences center, shown in the main pane with IEEE
    Account portal chrome. Live: toggling units updates the Expected Emails
    panel, and choosing Custom reveals per-unit frequency selectors. Saving
    applies the preferences and returns to the digest."""
    # View picker sits above the IEEE Account portal header.
    _render_view_picker(selected_norm)
    _render_ieee_header()

    saved = _saved_prefs(selected_norm)
    sig = _pref_sig(selected_norm)
    p = f"pref::{sig}::"  # common prefix for this selection's widget keys
    units_sorted = sorted(selected_norm,
                          key=lambda s: _unit_name(s, unit_info).lower())
    rates, _months = _monthly_rates(df, selected_norm)
    seen = _seen_tags(selected_norm)  # {lowercased: display} from the digest

    st.markdown('<div class="pref-page">', unsafe_allow_html=True)
    st.markdown('<h1 class="pref-h1">Local IEEE Communications</h1>',
                unsafe_allow_html=True)
    st.markdown('<p class="pref-intro">Configure your personalized local IEEE '
                'communications preferences. These choices control which '
                'eNotices you receive and how they are delivered.</p>',
                unsafe_allow_html=True)

    left, right = st.columns([3, 2], vertical_alignment="top")

    with left:
        st.markdown("#### Organizational Units")
        st.caption("Choose which organizational units' eNotices you want to "
                   "receive, and how often.")
        st.caption("**Immediate** sends each eNotice as its own email; "
                   "**Daily**, **Weekly**, or **Monthly** combine them into a "
                   "digest.")
        saved_units = saved.get("units")
        saved_freqs = saved.get("frequencies", {})

        # Bulk action: apply one delivery frequency to every unit at once
        # (replaces the old global frequency dropdown + hidden "Custom" mode).
        def _apply_all_freq():
            v = st.session_state[p + "setall"]
            for unit in units_sorted:
                st.session_state[p + "freq::" + unit] = v

        sa1, sa2 = st.columns([2, 1], vertical_alignment="bottom")
        sa1.selectbox("Set all units to", _FREQ_OPTIONS, key=p + "setall")
        sa2.button("Apply to all", on_click=_apply_all_freq,
                   use_container_width=True, key=p + "applyall")

        # One row per unit: an include checkbox (labelled with the unit name)
        # and that unit's own delivery frequency. Seeding the frequency in
        # session_state (rather than via index=) lets "Apply to all" and the
        # saved-value restore drive the selectbox without Streamlit warnings; an
        # excluded unit's frequency is disabled.
        st.markdown('<div class="pref-unit-head"><span>Include</span>'
                    '<span>Frequency</span></div>', unsafe_allow_html=True)
        unit_checked, unit_freq = {}, {}
        for u in units_sorted:
            fkey = p + "freq::" + u
            if fkey not in st.session_state:
                cur = saved_freqs.get(u, "Immediate")
                st.session_state[fkey] = cur if cur in _FREQ_OPTIONS \
                    else "Immediate"
            rc1, rc2 = st.columns([3, 2], vertical_alignment="center")
            with rc1:
                default_inc = True if saved_units is None else (u in saved_units)
                unit_checked[u] = st.checkbox(
                    _unit_name(u, unit_info), value=default_inc,
                    key=p + "unit::" + u)
            with rc2:
                unit_freq[u] = st.selectbox(
                    f"Frequency for {_unit_name(u, unit_info)}", _FREQ_OPTIONS,
                    key=fkey, label_visibility="collapsed",
                    disabled=not unit_checked[u])

        # Priority Tags and Tags to Exclude both pick from the tags in the
        # member's current archive. A tag can't be on both lists; a shared
        # on-change handler removes any overlap, with Priority winning.
        tag_options = sorted(seen.values(), key=str.lower)

        def _dedupe_tag_lists():
            prio_set = set(st.session_state.get(p + "priority", []))
            if prio_set:
                st.session_state[p + "excl"] = [
                    t for t in st.session_state.get(p + "excl", [])
                    if t not in prio_set]

        st.markdown("#### Priority Tags")
        st.caption("eNotices tagged with any of these hashtags are delivered "
                   "immediately — even from a unit set to a digest, and even if "
                   "they also carry an excluded tag (they'll still appear in "
                   "your archive). Tags come from your recent eNotices.")
        if tag_options:
            prio = st.multiselect(
                "Priority tags", options=tag_options,
                default=[t for t in saved.get("priority_terms", [])
                         if t in tag_options],
                format_func=lambda b: f"#{b}",
                key=p + "priority", label_visibility="collapsed",
                on_change=_dedupe_tag_lists,
                placeholder="Choose priority tags…")
        else:
            prio = []

        st.markdown("#### Tags to Exclude")
        st.caption("eNotices tagged with any of these hashtags won't be sent to "
                   "you, unless they also carry a priority tag. Tags come from "
                   "your recent eNotices.")
        if tag_options:
            excl = st.multiselect(
                "Tags to exclude", options=tag_options,
                default=[t for t in saved.get("exclude_terms", [])
                         if t in tag_options],
                format_func=lambda b: f"#{b}",
                key=p + "excl", label_visibility="collapsed",
                on_change=_dedupe_tag_lists,
                placeholder="Choose tags to exclude…")
        else:
            excl = []

        if not tag_options:
            st.info("Tags will appear here once your eNotice Archive has loaded "
                    "some eNotices — open your archive first, then return here. "
                    "This is specific to the prototype, where tags are "
                    "generated on demand as you open the archive. In a "
                    "production system the tags would already be available, so "
                    "they'd be listed here right away.")

    with right:
        included_now = {u: _unit_name(u, unit_info)
                        for u in units_sorted if unit_checked[u]}
        st.markdown(_expected_panel_html(rates, included_now),
                    unsafe_allow_html=True)
        st.button("View your eNotice Archive", key="prefs_back",
                  on_click=_go_digest, use_container_width=True)

    st.divider()
    b1, b2, b3, _ = st.columns([1, 1, 1, 1])
    save = b1.button("Save Preferences", type="primary",
                     use_container_width=True, key="prefs_save")
    cancel = b2.button("Cancel", use_container_width=True, key="prefs_cancel",
                       help="Discard any changes made since you last saved.")
    reset = b3.button("Reset to Default", use_container_width=True,
                      key="prefs_reset")
    st.markdown('</div>', unsafe_allow_html=True)

    _render_ieee_footer()

    if save:
        prio_terms = list(prio)
        prio_set = set(prio_terms)
        excl_terms = [t for t in excl if t not in prio_set]  # Priority wins
        prefs = st.session_state.setdefault("saved_prefs", {})
        prefs[sig] = {
            "units": {u for u in units_sorted if unit_checked[u]},
            "frequencies": {u: unit_freq[u] for u in units_sorted},
            "priority_terms": prio_terms,
            "priority_bodies": {b.lower() for b in prio_terms},
            "exclude_terms": excl_terms,
            "exclude_bodies": {b.lower() for b in excl_terms},
        }
        st.session_state.view = "digest"
        st.rerun()
    if cancel:
        # Discard unsaved edits: clear this selection's widget state (so the
        # controls re-seed from the last-saved preferences) but keep the saved
        # preferences themselves, then return to the Archive.
        for k in [k for k in st.session_state if k.startswith(p)]:
            del st.session_state[k]
        st.session_state.view = "digest"
        st.rerun()
    if reset:
        st.session_state.get("saved_prefs", {}).pop(sig, None)
        for k in [k for k in st.session_state if k.startswith(p)]:
            del st.session_state[k]
        st.rerun()


_SEARCH_HELP = (
    "**How search works in this prototype**\n\n"
    "Search looks within the same eNotices as your digest -- those sent to "
    "your selected units within the sidebar date range. It scans the full "
    f"text of the **{_SEARCH_SCAN_LIMIT} most recent** of those notices "
    "(fetched live from their public IEEE eNotice pages) and shows the ones "
    "that contain all of your terms, newest first.\n\n"
    "**In production**, every eNotice in your digest would already have a "
    "summary and tags, so search would filter your **entire** digest "
    "**instantly** by matching those **tags and summaries** -- with no live "
    "page fetching or scan limit.")


def _csv_records(digest, selected_norm):
    """Tile records (newest first) for the CSV-based digest."""
    recs = []
    for _, r in digest.sort_values("_sent_dt", ascending=False).iterrows():
        event_id = (str(r.get("event_id", "") or "").strip()
                    or _event_id_from_url(str(r.get("event_url", "") or "")))
        recs.append({
            "public_url": ("https://enotice.vtools.ieee.org/public/"
                           + str(r.get("id", ""))),
            "event_id": event_id,
            "subject": str(r.get("mailing_subject", "") or ""),
            "sent_dt": r["_sent_dt"],
            "recips": (parse_recipient_spoids(r.get("recipient_SPOIDs", ""))
                       & selected_norm),
        })
    return recs


def _search_records(digest, selected_norm, query):
    """Digest records whose eNotice matches every whitespace-separated term.

    Scopes to the same records as the digest (selected units + date range),
    takes the newest ``_SEARCH_SCAN_LIMIT``, fetches each one's public page,
    and keeps those whose subject + page text contain all terms (case-
    insensitive). A progress bar reports the live-scan phase. Newest first.
    """
    terms = [t for t in query.lower().split() if t]
    if not terms:
        return []
    candidates = _csv_records(digest, selected_norm)[:_SEARCH_SCAN_LIMIT]
    if not candidates:
        return []

    n = len(candidates)
    note = st.empty()
    note.caption(
        "Searching the full text of your selected units' recent eNotices by "
        "fetching their public pages. In production this would match "
        "pre-generated tags and summaries instantly, with no scan.")
    progress = st.progress(0.0, text=f"Scanning eNotice 1 of {n}...")

    def _matches(rec):
        text = (rec["subject"] + " "
                + _fetch_text(rec["public_url"], cap=_SEARCH_TEXT_CAP)).lower()
        return all(t in text for t in terms)

    hits = [False] * n
    with concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futures = {pool.submit(_matches, rec): i
                   for i, rec in enumerate(candidates)}
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            hits[futures[fut]] = fut.result()
            done += 1
            progress.progress(done / n,
                              text=f"Scanning eNotice {done} of {n}...")
    progress.empty()
    note.empty()

    # candidates are already newest-first; keep that order for the matches.
    return [rec for rec, hit in zip(candidates, hits) if hit]


def _render_tile_page(records, unit_info, scope_note, empty_msg,
                      exclude_bodies=frozenset(), collect_tags=False,
                      priority_bodies=frozenset()):
    """Paginate `records`, generate each tile's summary/image/tags, and render.

    `exclude_bodies` is a set of lower-cased tag bodies (from the preferences
    center's Tags to Exclude); a tile whose generated tags include any of them
    is hidden. `priority_bodies` (Priority Tags) overrides that: a tile with a
    priority tag is never hidden by an excluded tag, and its priority tag chips
    are styled distinctly. Because tags are produced per tile as the page is
    prepared, both are applied to the current page after generation.

    When `collect_tags` is set, every tag produced on this page is recorded in
    `st.session_state["seen_tags"]` (before exclusion), so the preferences
    center can offer exactly the tags present in the member's current digest."""
    total = len(records)
    sig = tuple(r["public_url"] for r in records)
    if st.session_state.get("digest_page_sig") != sig:
        st.session_state["digest_page_sig"] = sig
        st.session_state["digest_page"] = 0
    max_page = max(0, (total - 1) // _DIGEST_MAX_TILES)
    page = min(max(st.session_state.get("digest_page", 0), 0), max_page)
    st.session_state["digest_page"] = page
    start = page * _DIGEST_MAX_TILES
    tiles = records[start:start + _DIGEST_MAX_TILES]

    if not tiles:
        st.info(empty_msg)
        return

    placeholder_summary = (
        "Placeholder summary: a concise two- to three-sentence AI-generated "
        "overview of this eNotice will appear here, highlighting its key "
        "details and purpose. Final wording is pending integration.")
    placeholder_tags = "".join(
        f'<span class="digest-tag" title="Placeholder tag (AI tag '
        f'generation unavailable)">#{t}</span>'
        for t in ("PlaceholderOne", "PlaceholderTwo", "PlaceholderThree"))

    # Summaries, images, and tags are generated on demand and cached per notice
    # (a prototype artifact -- in production they'd exist when each eNotice is
    # published, so the digest would load instantly). Report per-notice progress.
    n_tiles = len(tiles)
    note = st.empty()
    note.caption(
        "This short wait is specific to the prototype: eNotice summaries, "
        "images, and tags are being generated on demand as you open the "
        "digest. In a production system this content would be created once, "
        "when each eNotice is published, so your digest would load instantly."
        "\n\n" + scope_note)
    progress = st.progress(0.0, text=f"Preparing eNotice 1 of {n_tiles}...")

    def _prepare(rec):
        summary = enotice_summary(rec["public_url"], rec["event_id"])
        return (summary,
                enotice_image(rec["public_url"], rec["event_id"], rec["subject"]),
                enotice_tags(rec["subject"], summary, rec["event_id"]))

    results = [None] * n_tiles
    with concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futures = {pool.submit(_prepare, rec): i for i, rec in enumerate(tiles)}
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            results[futures[fut]] = fut.result()
            done += 1
            progress.progress(done / n_tiles,
                              text=f"Preparing eNotice {done} of {n_tiles}...")
    progress.empty()
    note.empty()

    # Record the tags shown in this digest so the preferences center can offer
    # exactly the tags present in the member's current digest (before exclusion).
    if collect_tags:
        seen = st.session_state.setdefault("seen_tags", {})
        for _summary, _img, tags in results:
            for body, _src in (tags or []):
                if body:
                    seen[body.lower()] = body

    def _tag_span(body, src):
        """One tag chip. Priority tags are styled distinctly and their hover
        text notes that they trigger immediate delivery."""
        if body.lower() in priority_bodies:
            title = (f"{src} · Priority tag — eNotices with this tag are "
                     "delivered immediately")
            cls = "digest-tag digest-tag-priority"
        else:
            title, cls = src, "digest-tag"
        return (f'<span class="{cls}" title="{html.escape(title)}">'
                f'#{html.escape(body)}</span>')

    blocks = []
    excluded = 0
    for rec, (summary, img, tags) in zip(tiles, results):
        tag_bodies = {body.lower() for body, _src in (tags or [])}
        is_priority = bool(tag_bodies & priority_bodies)
        # Excluded tiles are hidden -- unless a priority tag overrides that.
        if exclude_bodies and (tag_bodies & exclude_bodies) and not is_priority:
            excluded += 1
            continue
        units_html = _join_units(rec["recips"], unit_info) or "—"
        sent = rec["sent_dt"]
        sent_txt = f"Sent {_fmt_date(sent)}" if pd.notna(sent) else ""
        subject = html.escape(rec["subject"])
        summary_html = _linkify(summary or placeholder_summary)
        tags_html = ("".join(_tag_span(body, src) for body, src in tags)
                     if tags else placeholder_tags)
        thumb = (f'<div class="digest-thumb"><img src="{html.escape(img)}" '
                 f'alt=""></div>' if img
                 else '<div class="digest-thumb">🖼️</div>')
        blocks.append(
            '<div class="digest-tile">'
            f'{thumb}'
            '<div class="digest-body">'
            f'<div class="digest-units">{units_html}</div>'
            f'<div class="digest-sent">{sent_txt}</div>'
            f'<a class="digest-subject" href="{html.escape(rec["public_url"])}" '
            f'target="_blank">{subject}</a>'
            f'<div class="digest-summary">{summary_html}</div>'
            f'<div class="digest-tags">{tags_html}</div>'
            '</div></div>')
    st.markdown("\n".join(blocks), unsafe_allow_html=True)
    if excluded:
        msg = (f"{excluded} eNotice(s) on this page were hidden by your "
               "excluded tags.")
        if not blocks:
            msg += (" Use the Newer/Older controls to see others, or adjust "
                    "your excluded tags under “Manage your digest settings.”")
        st.caption(msg)

    # Pager: Newer / Older with a position indicator, when there's >1 page.
    if total > _DIGEST_MAX_TILES:
        lo, hi = start + 1, start + len(tiles)
        nav_l, nav_c, nav_r = st.columns([2, 4, 2], vertical_alignment="center")
        with nav_l:
            if page > 0:
                st.button("‹ Newer", key="digest_prev", on_click=_digest_prev)
        with nav_c:
            st.markdown(
                f'<div class="digest-page">Showing eNotices {lo}–{hi} '
                f'of {total} (most recent first)</div>', unsafe_allow_html=True)
        with nav_r:
            if page < max_page:
                st.button("Older ›", key="digest_next", on_click=_digest_next)

    st.markdown(f'<div class="digest-scope">{html.escape(scope_note)}</div>',
                unsafe_allow_html=True)


def render_digest_view(df, digest, selected_norm, unit_info, end_date):
    """Render the tiled digest, with a search box over the selected units.

    Saved preferences (from the "Manage your digest settings" page) narrow the
    view: only the units the member left checked contribute eNotices, and tiles
    whose tags match an excluded tag are hidden. `df` is the whole (sent) data
    file, used by the preferences page to estimate emails/month per unit."""
    _seen_tags(selected_norm)  # ensure the seen-tags store matches this selection
    saved = _saved_prefs(selected_norm)
    included = saved.get("units")  # None => no saved prefs yet (all included)
    effective_norm = (selected_norm if included is None
                      else {s for s in selected_norm if s in included})
    exclude_bodies = saved.get("exclude_bodies", frozenset())
    priority_bodies = saved.get("priority_bodies", frozenset())
    if included is not None:
        digest = digest[digest["recipient_SPOIDs"].apply(
            lambda c: matches_recipients(c, effective_norm))]

    _render_view_picker(selected_norm)

    c_logo, c_title = st.columns([2, 8], vertical_alignment="center")
    with c_logo:
        if _LOGO_PATH.exists():
            st.image(str(_LOGO_PATH), width=150)
    with c_title:
        st.markdown('<div class="digest-title">Your IEEE eNotice Archive</div>',
                    unsafe_allow_html=True)

    date_txt = _fmt_full_date(end_date) if end_date is not None else ""
    st.markdown('<hr class="digest-rule">'
                f'<div class="digest-date">{date_txt}</div>',
                unsafe_allow_html=True)

    # Search box + info popover (how search works here vs. in production).
    c_q, c_i = st.columns([20, 1], vertical_alignment="center")
    with c_q:
        query = st.text_input(
            "Search eNotices", key="digest_search",
            placeholder="Search your archive's eNotices by full-text content…",
            label_visibility="collapsed").strip()
    with c_i:
        with st.popover("ℹ️"):
            st.markdown(_SEARCH_HELP)

    if query:
        records = _search_records(digest, effective_norm, query)
        st.caption(f"{len(records)} recent eNotice(s) matching your search.")
        scope_note = (
            "In this prototype, search scans the full text of the "
            f"{_SEARCH_SCAN_LIMIT} most recent eNotices in your digest "
            "(selected units + sidebar date range) and shows those matching "
            "your terms -- most recent first. See the info icon by the search "
            "box for how this differs from production.")
        empty_msg = ("No recent eNotices from your selected units match your "
                     "search. Try different or fewer terms, or clear the search "
                     "box to return to the digest.")
        _render_tile_page(records, unit_info, scope_note, empty_msg,
                          exclude_bodies, priority_bodies=priority_bodies)
    else:
        scope_note = (
            "This prototype shows the eNotices sent to your selected units "
            "within the date range set in the sidebar, from the selected data "
            f"file -- most recent first, {_DIGEST_MAX_TILES} at a time. Use the "
            "Newer/Older controls to move through them.")
        empty_msg = ("No eNotices to display for the selected units within the "
                     "sidebar date range. Widen the date range, adjust your "
                     "selection in the sidebar, or check your included "
                     "Organizational Units under “Manage your digest settings.”")
        _render_tile_page(_csv_records(digest, effective_norm), unit_info,
                          scope_note, empty_msg, exclude_bodies,
                          collect_tags=True, priority_bodies=priority_bodies)

    agg = _join_units(effective_norm, unit_info)
    if agg:
        st.markdown('<div class="digest-agg">This archive includes eNotices '
                    f'from {agg}.</div>', unsafe_allow_html=True)
    st.button("Manage your preferences", key="manage_prefs", on_click=_go_prefs)
    st.markdown(f'<div class="digest-copyright">© {date.today().year} IEEE. '
                'All rights reserved.</div>', unsafe_allow_html=True)


def render_digest_freq_view(freq, digest, selected_norm, unit_info):
    """Render the Daily/Weekly/Monthly digest: the most recent non-empty period
    (within the sidebar date range) for the units the member set to `freq`, as
    tiles with the same format and pagination as the eNotice Archive.

    `freq` is a stored frequency label ("Daily Digest" / "Weekly Digest" /
    "Monthly Digest"). Only the units assigned that frequency contribute, and
    the excluded-tag filter applies exactly as on the Archive."""
    _render_view_picker(selected_norm)

    units_f = _active_digest_freqs(selected_norm).get(freq, set())
    _saved = _saved_prefs(selected_norm)
    exclude_bodies = _saved.get("exclude_bodies", frozenset())
    priority_bodies = _saved.get("priority_bodies", frozenset())
    period_word = {"Daily Digest": "day", "Weekly Digest": "week",
                   "Monthly Digest": "month"}[freq]

    def _freq_footer():
        """The 'Manage your preferences' link + copyright, shown on every path."""
        st.button("Manage your preferences", key="manage_prefs_freq",
                  on_click=_go_prefs)
        st.markdown(f'<div class="digest-copyright">© {date.today().year} IEEE. '
                    'All rights reserved.</div>', unsafe_allow_html=True)

    c_logo, c_title = st.columns([2, 8], vertical_alignment="center")
    with c_logo:
        if _LOGO_PATH.exists():
            st.image(str(_LOGO_PATH), width=150)
    with c_title:
        st.markdown(f'<div class="digest-title">Your {freq}</div>',
                    unsafe_allow_html=True)

    # Restrict the (already date-ranged) set to this frequency's units.
    sub = digest[digest["recipient_SPOIDs"].apply(
        lambda c: matches_recipients(c, units_f))]
    if sub.empty or sub["_sent_dt"].dropna().empty:
        st.markdown('<hr class="digest-rule">', unsafe_allow_html=True)
        st.info(f"No eNotices for your {freq.lower()} units within the sidebar "
                "date range. Widen the date range in the sidebar, or adjust "
                "delivery frequencies in the Preferences Center.")
        _freq_footer()
        return

    # Find the most recent non-empty period and its rows + a display label.
    dts = sub["_sent_dt"]
    if freq == "Daily Digest":
        day = dts.dropna().dt.normalize().max()
        period_rows = sub[dts.dt.normalize() == day]
        period_label = _fmt_full_date(day)
    elif freq == "Weekly Digest":
        week_start = (dts - pd.to_timedelta(dts.dt.weekday, unit="D")).dt.normalize()
        start = week_start.dropna().max()
        period_rows = sub[week_start == start]
        end = start + pd.Timedelta(days=6)
        period_label = f"the week of {_fmt_date(start)} – {_fmt_date(end)}"
    else:  # Monthly Digest
        month = dts.dt.to_period("M")
        target = month.dropna().max()
        period_rows = sub[month == target]
        period_label = target.strftime("%B %Y")

    st.markdown('<hr class="digest-rule">'
                f'<div class="digest-date">{html.escape(period_label)}</div>',
                unsafe_allow_html=True)

    scope_note = (
        f"This prototype shows what your {freq.lower()} would have contained "
        f"for {period_label} — the most recent {period_word} with eNotices for "
        f"the units you set to “{freq}”, within the sidebar date range. Tiles "
        f"appear {_DIGEST_MAX_TILES} at a time; use the Newer/Older controls to "
        "move through them.")
    empty_msg = (f"Every eNotice in this {period_word} was hidden by your "
                 "excluded tags. Adjust your excluded tags in the Preferences "
                 "Center.")
    _render_tile_page(_csv_records(period_rows, units_f), unit_info,
                      scope_note, empty_msg, exclude_bodies,
                      priority_bodies=priority_bodies)

    agg = _join_units(units_f, unit_info)
    if agg:
        st.markdown(f'<div class="digest-agg">This {freq.lower()} covers '
                    f'eNotices from {agg}.</div>', unsafe_allow_html=True)
    _freq_footer()


# --------------------------------------------------------------------------- #
# UI

# On wide screens, widen the sidebar so the full data-file name shows in the
# picker and the Section-search prompt fits on one line. Left at Streamlit's
# responsive default on small/mobile viewports so it doesn't overflow.
st.markdown(
    """
    <style>
    @media (min-width: 768px) {
        section[data-testid="stSidebar"] {
            width: 500px !important;
            min-width: 500px !important;
        }
    }
    /* Related-units picker: let each selected-unit chip grow to fit its full
       name (no ellipsis) and use a calm teal instead of the default red. */
    span[data-baseweb="tag"] {
        background-color: #007377 !important;
        color: #ffffff !important;
        max-width: none !important;
        height: auto !important;
    }
    span[data-baseweb="tag"] span {
        max-width: none !important;
        overflow: visible !important;
        text-overflow: clip !important;
        white-space: normal !important;
    }
    /* --- Digest view --- */
    .digest-title { font-size: 1.9rem; font-weight: 600; color: #1a1a1a;
                    line-height: 1.1; }
    .digest-rule { border: none; border-top: 2px solid #00629B;
                   margin: -0.6rem 0 0.15rem !important; }
    .digest-date { color: #555; font-size: 0.9rem; margin: 0; }
    .digest-tile { display: flex; gap: 1rem; background: #ffffff;
                   border: 1px solid #e4e4e4; border-radius: 6px; padding: 1rem;
                   margin-bottom: 1rem;
                   box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    .digest-thumb { flex: 0 0 130px; width: 130px; height: 130px;
                    border-radius: 4px; display: flex; align-items: center;
                    justify-content: center; font-size: 2.2rem; color: #9bb0c1;
                    overflow: hidden;
                    background: linear-gradient(135deg, #EAF1F6, #C7D6E4); }
    .digest-thumb img { width: 100%; height: 100%; object-fit: cover;
                        display: block; }
    .digest-body { flex: 1; min-width: 0; }
    .digest-units { font-size: 0.9rem; margin-bottom: 0.1rem; }
    .digest-units a, .digest-subject, .digest-agg a,
    .digest-summary a { color: #00629B; text-decoration: none; }
    .digest-units a:hover, .digest-subject:hover,
    .digest-agg a:hover, .digest-summary a:hover { text-decoration: underline; }
    .digest-sent { color: #666; font-size: 0.85rem; margin-bottom: 0.35rem; }
    .digest-subject { display: inline-block; font-size: 1.15rem;
                      font-weight: 600; margin-bottom: 0.4rem; }
    .digest-summary { color: #333; font-size: 0.95rem; line-height: 1.45;
                      margin-bottom: 0.55rem; }
    .digest-tag { display: inline-block; background: #E7EEF4; color: #00629B;
                  font-size: 0.8rem; padding: 0.1rem 0.55rem; border-radius: 12px;
                  margin: 0 0.35rem 0.25rem 0; cursor: help; }
    /* Priority tag chips: orange instead of the blue of ordinary tags, so they
       stand out in the tag list. */
    .digest-tag-priority { background: #FFE7C6; color: #1a1a1a; font-weight: 700;
                           box-shadow: inset 0 0 0 1px #E87722; }
    .digest-tag-priority::before { content: "★ "; color: #E87722; }
    .digest-page { text-align: center; color: #444; font-size: 0.9rem; }
    .digest-scope { text-align: center; color: #777; margin-top: 1rem;
                    font-size: 0.85rem; font-style: italic; }
    .digest-agg { text-align: center; color: #444; margin-top: 1.2rem;
                  font-size: 1rem; }
    .digest-copyright { text-align: center; color: #777; font-size: 1rem;
                        margin-top: 0.3rem; }
    /* "Manage your preferences" rendered as a centered inline text link. */
    .st-key-manage_prefs button, .st-key-manage_prefs_freq button {
        color: #00629B !important; background: transparent !important;
        border: none !important; box-shadow: none !important;
        padding: 0 !important; min-height: 0 !important; font-weight: 400; }
    .st-key-manage_prefs button:hover,
    .st-key-manage_prefs_freq button:hover { text-decoration: underline; }
    .st-key-manage_prefs, .st-key-manage_prefs_freq {
        width: 100% !important; display: flex; justify-content: center; }
    /* --- Preferences center (Manage your digest settings) --- */
    .pref-panel { background: #EAF1F6; border: 1px solid #BAC6D9;
                  border-radius: 8px; padding: 1rem 1.1rem; }
    .pref-panel-title { font-size: 1.1rem; font-weight: 600; color: #00629B; }
    .pref-total { font-size: 2.6rem; font-weight: 700; color: #1a1a1a;
                  line-height: 1.1; margin: 0.2rem 0; }
    .pref-sub { color: #667; font-size: 0.82rem; }
    .pref-breakdown-title { font-weight: 600; color: #1a1a1a;
                            margin: 0.9rem 0 0.4rem; font-size: 0.92rem;
                            border-top: 1px solid #BAC6D9; padding-top: 0.7rem; }
    .pref-row { display: flex; justify-content: space-between; gap: 0.5rem;
                align-items: baseline; padding: 0.2rem 0; font-size: 0.9rem; }
    .pref-name { color: #333; }
    .pref-count { font-weight: 700; color: #00629B; min-width: 2rem;
                  text-align: right; }
    .pref-unit-head { display: flex; font-size: 0.72rem; font-weight: 700;
                      color: #8a8a8a; text-transform: uppercase;
                      letter-spacing: 0.04em; margin: 0.5rem 0 0.1rem; }
    .pref-unit-head span:first-child { flex: 3; }
    .pref-unit-head span:last-child { flex: 2; }
    /* --- IEEE Account portal chrome (member-facing preferences page) --- */
    .ieee-chrome-inner { max-width: 1120px; margin: 0 auto; }
    .ieee-topnav { background: #1a1a1a; color: #f2f2f2; font-size: 0.8rem;
                   padding: 0.55rem 1rem; }
    .ieee-tn-inner { display: flex; justify-content: space-between;
                     flex-wrap: wrap; gap: 0.4rem 1rem; }
    .ieee-tn-left, .ieee-tn-right { display: flex; align-items: center;
                                    flex-wrap: wrap; }
    .ieee-topnav em { font-style: italic; }
    .ieee-tn-name { font-weight: 700; color: #ffffff; }
    .ieee-tn-bar { color: #5a5a5a; margin: 0 0.55rem; }
    .ieee-home { color: #f2f2f2; }
    .ieee-brandbar { background: #ffffff; padding: 1.1rem 1rem 0.9rem; }
    .ieee-brand-inner { display: flex; justify-content: space-between;
                        align-items: center; }
    .ieee-account { color: #00629B; font-size: 2rem; font-weight: 700;
                    line-height: 1; }
    .ieee-logo { height: 38px; width: auto; }
    .ieee-breadcrumb { background: #f2f2f2; border-top: 1px solid #e2e2e2;
                       padding: 0.65rem 1rem; font-size: 0.95rem; }
    .ieee-crumb { color: #981D97; }
    .ieee-crumb.current { text-decoration: underline; }
    .ieee-crumb-sep { color: #999; margin: 0 0.55rem; }
    .ieee-footer { background: #00629B; color: #ffffff;
                   padding: 2.2rem 1rem 1.6rem; margin-top: 1.5rem; }
    .ieee-fcols { display: flex; flex-wrap: wrap; gap: 1.5rem; }
    .ieee-fcol { flex: 1 1 170px; min-width: 140px; }
    .ieee-fcol h4 { color: #fff; font-size: 1rem; font-weight: 700;
                    margin: 0 0 0.7rem; }
    .ieee-fblurb { color: #e6eef4; font-size: 0.82rem; line-height: 1.45;
                   margin: 0 0 0.7rem; }
    .ieee-fcol ul { list-style: none; padding: 0; margin: 0; }
    .ieee-fcol li { margin: 0.3rem 0; }
    .ieee-fcol li a { color: #fff; text-decoration: none; font-size: 0.85rem; }
    .ieee-fcol li a::before { content: "› "; color: #BAC6D9; }
    .ieee-fcol li a:hover { text-decoration: underline; }
    .ieee-contact-btn { display: inline-block; border: 1px solid #fff;
                        padding: 0.5rem 1.1rem; font-weight: 700;
                        font-size: 0.85rem; margin-bottom: 0.9rem; color: #fff;
                        text-decoration: none; }
    .ieee-contact-btn:hover { background: #fff; color: #00629B; }
    .ieee-flocrow { display: flex; justify-content: space-between;
                    align-items: flex-end; flex-wrap: wrap; gap: 1rem;
                    margin-top: 1.6rem; }
    .ieee-floc { flex: 1 1 260px; }
    .ieee-fsocials { display: flex; gap: 0.9rem; align-items: center; }
    .ieee-social { color: #fff; display: inline-flex; }
    .ieee-social svg { width: 20px; height: 20px; display: block; }
    .ieee-social:hover { color: #BAC6D9; }
    .ieee-footer-legal { background: #1a1a1a; padding: 1.1rem 1rem;
                         font-size: 0.8rem; }
    .ieee-legal-links { margin-bottom: 0.55rem; line-height: 1.9; }
    .ieee-footer-legal a { color: #FFC72C; text-decoration: none; }
    .ieee-footer-legal a:hover { text-decoration: underline; }
    .ieee-legal-bar { color: #5a5a5a; margin: 0 0.55rem; }
    .ieee-copyright { color: #B7B8BA; line-height: 1.55; }
    .ieee-copyright a { color: #FFC72C; text-decoration: underline; }
    /* Preferences page body (between the IEEE header and footer). */
    .pref-h1 { color: #00629B; font-size: 1.7rem; font-weight: 700;
               margin: 1rem 0 0.25rem; }
    .pref-intro { color: #444; font-size: 0.97rem; margin: 0 0 0.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("1 · eNotice data file")
    data_files = list_data_files()
    if not data_files:
        st.error(f"No CSV files found in {_DATA_DIR}.")
        st.stop()
    data_file = st.selectbox("Data file", data_files)
    df = load_enotices(data_file)

    st.header("2 · IEEE Section")
    if not load_section_index():
        st.error("units.csv not found or has no Sections.")
        st.stop()
    section_spoid = st_searchbox(
        search_sections, key="section_search",
        placeholder="Search Sections by name (3+ letters)",
        label="Search by name")

    selected = []
    section_ou = None
    if section_spoid:
        # Walk the OU graph once per Section (memoized in session_state), showing
        # a progress bar only on that first walk. Pool and Society mappings are
        # derived from that single result -- no repeat walks.
        rel_key = "related::" + str(section_spoid)
        if rel_key not in st.session_state:
            _prog = st.progress(0.0, text="Finding related units…")

            def _rel_progress(n):
                # Total is unknown during the walk, so ease toward 0.9 as units
                # are found (never a false 100%).
                frac = min(0.9, 1.0 - 1.0 / (1.0 + n / 20.0))
                _prog.progress(frac, text=f"Finding related units… ({n} found)")

            st.session_state[rel_key] = related_units(
                section_spoid, progress=_rel_progress)
            _prog.progress(1.0, text="Done")
            _prog.empty()
        related = st.session_state[rel_key]
        pool = recipient_pool(related, data_file)
        chapters_by_soc, sbc_links = society_links(related)
        section_ou = related.get(resolve_spoid(section_spoid)) or \
            related.get(section_spoid)
        root_spoid = section_ou.spoid if section_ou else section_spoid

        soc_key = "societies_pick"
        sel_key = f"units_{root_spoid}_{data_file}"
        prev_soc_key = f"prevsoc_{root_spoid}_{data_file}"
        prev_sel_key = sel_key + "_prev"

        def _inject_for_societies(sel, soc_list):
            """Append each society, its Chapter/Joint-Chapter children, and any
            Student Branch Chapter whose Student Branch is already in sel."""
            sel = list(sel)
            present = {resolve_spoid(s) for s in sel}

            def _add(sp):
                if resolve_spoid(sp) not in present:
                    sel.append(sp)
                    present.add(resolve_spoid(sp))

            for soc in soc_list:
                _add(soc)
                for ch in chapters_by_soc.get(resolve_spoid(soc), []):
                    _add(ch)
                for sbc, sb, sc in sbc_links:
                    if sc == resolve_spoid(soc) and sb is not None \
                            and resolve_spoid(sb) in present:
                        _add(sbc)
            return sel

        def _on_society_change():
            cur = st.session_state[soc_key]
            newly = [s for s in cur
                     if s not in st.session_state.get(prev_soc_key, [])]
            sel = _inject_for_societies(
                st.session_state.get(sel_key, []), newly)
            st.session_state[sel_key] = sel
            st.session_state[prev_sel_key] = sel
            st.session_state[prev_soc_key] = list(cur)

        def _on_units_change():
            cur = list(st.session_state[sel_key])
            newly = [u for u in cur
                     if u not in st.session_state.get(prev_sel_key, [])]
            chosen = {resolve_spoid(s)
                      for s in st.session_state.get(soc_key, [])}
            present = {resolve_spoid(u) for u in cur}
            for u in newly:
                ou = related.get(u)
                if ou is None or \
                        outype.classify_ou(ou) is not UnitType.STUDENT_BRANCH:
                    continue
                for sbc, sb, sc in sbc_links:
                    if sb is not None and resolve_spoid(sb) == resolve_spoid(u) \
                            and sc in chosen \
                            and resolve_spoid(sbc) not in present:
                        cur.append(sbc)
                        present.add(resolve_spoid(sbc))
            st.session_state[sel_key] = cur
            st.session_state[prev_sel_key] = cur

        # 3 -- Societies (drives auto-additions into Selected units below).
        st.header("3 · Societies")
        societies = load_society_index()
        soc_labels = {r["spoid"]: r["label"] for r in societies}
        st.multiselect(
            "Societies you belong to",
            options=[r["spoid"] for r in societies],
            format_func=lambda s: soc_labels.get(s, s),
            key=soc_key, on_change=_on_society_change,
            help="Adds the Society plus this Section's Chapters/Joint Chapters "
                 "that are children of it. A Student Branch Chapter is added "
                 "only once its Student Branch is also selected.")

        # 4 -- Selected units (the digest input). Options: recipient related
        # units (or all related units when the checkbox is on), the Section,
        # plus whatever is currently selected.
        st.header("4 · Selected units")
        include_all = st.checkbox(
            "Include units that received no eNotices", value=False,
            key="include_no_enotice",
            help="Also offer related units that received no eNotices in this "
                 "file (e.g. Student Branches and Student Branch Chapters), so "
                 "they can be added to the digest.")
        base_pool = related if include_all else pool
        label_pool = dict(base_pool)
        if section_ou is not None:
            label_pool.setdefault(root_spoid, section_ou)

        # Seed on first render: Section + its recipient ancestors, then inject
        # anything owed to already-selected societies.
        if sel_key not in st.session_state:
            recips = load_recipient_spoids(data_file)
            seed = [root_spoid] if section_ou is not None else []
            for a, aou in section_ancestors(section_spoid).items():
                if resolve_spoid(a) in recips and a not in seed:
                    seed.append(a)
                    label_pool.setdefault(a, aou)
            seed = _inject_for_societies(
                seed, st.session_state.get(soc_key, []))
            st.session_state[sel_key] = seed
            st.session_state[prev_sel_key] = seed
            st.session_state[prev_soc_key] = list(
                st.session_state.get(soc_key, []))

        # Ensure every currently-selected unit is a valid, labelled option.
        for sp in st.session_state.get(sel_key, []):
            if sp not in label_pool:
                ou = related.get(sp) or load_ou(resolve_spoid(sp))
                if ou is not None:
                    label_pool[sp] = ou

        options = sorted(
            label_pool,
            key=lambda s: (s != root_spoid, (label_pool[s].name or s).lower()))
        selected = st.multiselect(
            "Units to include in the digest",
            options=options,
            format_func=lambda s: unit_label(label_pool[s]),
            key=sel_key, on_change=_on_units_change)
        st.caption(f"{len(base_pool)} related unit(s) available"
                   + ("" if include_all else " (with eNotices)")
                   + "; societies add Chapters (Academic / Grouping / Other "
                   "excluded).")

    st.header("5 · Date range")
    _dates = df["_sent_dt"].dropna()
    if _dates.empty:
        start_date = end_date = None
        st.caption("No dated notices in this file.")
    else:
        min_d, max_d = _dates.min().date(), _dates.max().date()
        start_key, end_key = f"start_{data_file}", f"end_{data_file}"
        # Seed defaults via session state (rather than value=) so the quick-set
        # buttons below, which write end_key, don't trip Streamlit's "created
        # with a default value but also had its value set via Session State"
        # warning.
        st.session_state.setdefault(start_key, min_d)
        st.session_state.setdefault(end_key, max_d)
        start_date = st.date_input("Start date",
                                   min_value=min_d, max_value=max_d,
                                   key=start_key)
        end_date = st.date_input("End date",
                                 min_value=min_d, max_value=max_d,
                                 key=end_key)

        def _quick_end(kind):
            """Set the end date relative to the start date, clamped to range."""
            s = st.session_state.get(start_key, min_d)
            if kind == "same":
                cand = s
            elif kind == "week":
                cand = s + timedelta(days=7)
            else:  # "month"
                cand = (pd.Timestamp(s) + pd.DateOffset(months=1)).date()
            st.session_state[end_key] = min(max(cand, min_d), max_d)

        b1, b2, b3 = st.columns(3)
        b1.button("= Start", on_click=_quick_end, args=("same",),
                  help="Set end date equal to the start date")
        b2.button("+1 week", on_click=_quick_end, args=("week",),
                  help="Set end date one week after the start date")
        b3.button("+1 month", on_click=_quick_end, args=("month",),
                  help="Set end date one month after the start date")

# --------------------------------------------------------------------------- #
# Main panel

if not section_spoid:
    st.info("Pick a data file, then search for a Section in the sidebar.")
    st.stop()

# The Selected-units picker is the single source of truth for the digest
# (Societies auto-populate it in the sidebar).
selected_units = list(selected)

if not selected_units:
    st.info("Select one or more units (or a Society) in the sidebar to build "
            "the digest.")
    st.stop()

selected_norm = {resolve_spoid(s) for s in selected_units}

if "recipient_SPOIDs" not in df.columns:
    st.error("Column 'recipient_SPOIDs' is missing from the data file.")
    st.stop()

mask = df["recipient_SPOIDs"].apply(
    lambda c: matches_recipients(c, selected_norm))

# Date-range filter (inclusive of the whole end day). Undated rows are dropped.
if start_date and end_date:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    mask &= (df["_sent_dt"] >= start_ts) & (df["_sent_dt"] < end_ts)

digest = df[mask]

# Resolve selected units to their OU (name + website) for both views.
unit_info = {resolve_spoid(sp): ou for sp, ou in label_pool.items()}

st.session_state.setdefault("view", "digest")
# Fall back to the Archive if the member is on a digest view whose frequency is
# no longer assigned to any included unit (e.g. after editing preferences).
if st.session_state.view in _VIEW_FREQ and \
        _VIEW_FREQ[st.session_state.view] not in _active_digest_freqs(selected_norm):
    st.session_state.view = "digest"

if st.session_state.view == "preferences":
    render_preferences_view(df, selected_norm, unit_info)
    st.stop()
if st.session_state.view == "digest":
    render_digest_view(df, digest, selected_norm, unit_info, end_date)
    st.stop()
if st.session_state.view in _VIEW_FREQ:
    render_digest_freq_view(_VIEW_FREQ[st.session_state.view],
                            digest, selected_norm, unit_info)
    st.stop()

# --- Table view (internal) ---
_render_view_picker(selected_norm)

# Drop the excluded statistic columns, the status column, and hidden helpers.
drop_cols = [c for c in _STAT_COLUMNS + _HIDDEN_COLUMNS if c in digest.columns]
drop_cols += [c for c in digest.columns if c.startswith("_")]
summary = digest.drop(columns=drop_cols)

summary = filter_dataframe(summary)

# Per-unit received counts over the sent + date-range rows (before the table's
# column filters), so the numbers are stable and match the date range shown.
received = Counter()
for cell in digest["recipient_SPOIDs"]:
    for u in parse_recipient_spoids(cell) & selected_norm:
        received[u] += 1
# Each SPOID gets a hover tooltip showing the unit's name (title attribute).
name_by_spoid = {resolve_spoid(sp): (ou.name or "")
                 for sp, ou in label_pool.items()}
units_html = ", ".join(
    f'<span title="{html.escape(name_by_spoid.get(u, ""))}">{u}</span> '
    f'({received.get(u, 0)})'
    for u in sorted(selected_norm))

st.subheader(f"{len(summary)} sent eNotice(s) for the selected unit(s)")
caption = ("File: " + html.escape(data_file)
           + "  ·  Selected units (eNotices received): " + units_html)
if start_date and end_date:
    caption += f"  ·  {start_date} to {end_date}"
st.caption(caption, unsafe_allow_html=True)

# Add a public_url link column (between mailing_subject and sent_at) and render
# event_url as a link too. id stays plain text.
display_df = summary.copy()
column_config = {}
if "id" in display_df.columns:
    public = ("https://enotice.vtools.ieee.org/public/"
              + display_df["id"].astype(str))
    if "mailing_subject" in display_df.columns:
        pos = display_df.columns.get_loc("mailing_subject") + 1
    elif "sent_at" in display_df.columns:
        pos = display_df.columns.get_loc("sent_at")
    else:
        pos = len(display_df.columns)
    display_df.insert(pos, "public_url", public)
    column_config["public_url"] = st.column_config.LinkColumn("public_url")
if "event_url" in display_df.columns:
    column_config["event_url"] = st.column_config.LinkColumn("event_url")

st.dataframe(display_df, width="stretch", hide_index=True,
             column_config=column_config)
