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

# Digest view shows at most this many recent eNotices, none older than this many
# months before the sidebar end date.
_DIGEST_MAX_TILES = 6
_DIGEST_MAX_AGE_MONTHS = 1

# AI summary agent (Anthropic). A light model keeps per-digest cost down.
_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
_SUMMARY_MAX_TOKENS = 220
_FETCH_TIMEOUT = 20  # seconds, per source page fetch

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


def _walk(start_ou, edges_of, seen, collected):
    """Breadth-first walk in one direction (parents or children).

    ``edges_of(ou)`` returns the next SPOIDs to consider. A node is added to
    ``collected`` and expanded only if it fetches successfully and is not an
    excluded type; excluded nodes are neither shown nor traversed through.
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


@st.cache_data(show_spinner=False)
def related_units(section_spoid):
    """Return {spoid: OU} for the Section plus its ancestors and descendants.

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

    _walk(root, parents_of, seen, collected)     # ancestors
    _walk(root, children_of, seen, collected)    # descendants
    return collected


@st.cache_data(show_spinner=False)
def recipient_pool(section_spoid, filename):
    """Related units that received at least one eNotice in the file.

    The selectable list in the Related-units picker: the Section plus its
    ancestors/descendants (Academic/Grouping/Other already excluded), kept only
    if they were a recipient in the chosen file.
    """
    recipients = load_recipient_spoids(filename)
    return {u: ou for u, ou in related_units(section_spoid).items()
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


@st.cache_data(show_spinner=False)
def society_links(section_spoid):
    """Map the Section's chapters to the selectable Societies that parent them.

    Returns (chapters_by_soc, sbc_links):
      chapters_by_soc[society_spoid] -> [Chapter / Joint Chapter spoids] whose
          direct parent is that Society.
      sbc_links -> list of (sbc_spoid, student_branch_spoid, society_spoid) for
          Student Branch Chapters, which are added only once their Student
          Branch is also selected.
    society_spoids are normalized and limited to the Societies in units.csv.
    """
    societies = {resolve_spoid(r["spoid"]) for r in load_society_index()}
    related = related_units(section_spoid)
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


def _fetch_text(url):
    """Visible text of a page (scripts/markup stripped, capped)."""
    doc = _fetch_raw(url)
    doc = re.sub(r"(?is)<script.*?</script>", " ", doc)
    doc = re.sub(r"(?is)<style.*?</style>", " ", doc)
    text = html.unescape(re.sub(r"(?s)<[^>]+>", " ", doc))
    return re.sub(r"\s+", " ", text).strip()[:6000]


@st.cache_data(show_spinner=False, ttl=86400)
def enotice_summary(public_url, event_url):
    """A 2-3 sentence AI summary of an eNotice from its public page (primary)
    and event page (secondary). Returns None when unavailable (no key, fetch or
    LLM failure) so the caller can fall back to placeholder text.

    Cached per (public_url, event_url) so each notice is summarized once.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    primary = _fetch_text(public_url)
    if not primary:
        return None
    content = f"eNotice page ({public_url}):\n{primary}"
    secondary = _fetch_text(event_url) if event_url else ""
    if secondary:
        content += f"\n\nRelated event page ({event_url}):\n{secondary}"
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
def enotice_image(public_url, event_url, subject):
    """An image src for a digest tile, or None.

    Prefers a suitable content picture on the public page, then the event page
    (both confirmed by the vision agent); otherwise generates one. Cached per
    notice so pages aren't re-scanned and images aren't re-generated on rerun.
    """
    for src in (public_url, event_url):
        if not src:
            continue
        raw = _fetch_raw(src)
        if not raw:
            continue
        for cand in _content_image_candidates(raw, src)[:2]:
            if _image_is_suitable(cand, subject):
                return cand
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


def _go_archive():
    st.session_state.view = "archive"


def _go_digest():
    st.session_state.view = "digest"


@st.dialog("Digest settings")
def _settings_dialog():
    st.write("Digest settings can currently be adjusted in the sidebar. In the "
             "future, this will be managed through your IEEE profile settings.")


def render_digest_view(digest, selected_norm, unit_info, end_date):
    """Render the tiled digest of the most recent eNotices."""
    c_logo, c_title, c_link = st.columns([2, 6, 2],
                                         vertical_alignment="center")
    with c_logo:
        if _LOGO_PATH.exists():
            st.image(str(_LOGO_PATH), width=150)
    with c_title:
        st.markdown('<div class="digest-title">Your IEEE eNotice Digest</div>',
                    unsafe_allow_html=True)
    with c_link:
        st.button("Digest Archive", key="to_archive", on_click=_go_archive)

    date_txt = _fmt_full_date(end_date) if end_date is not None else ""
    st.markdown('<hr class="digest-rule">'
                f'<div class="digest-date">{date_txt}</div>',
                unsafe_allow_html=True)

    # Most-recent eNotices, none older than the max age before the end date.
    tiles = digest
    if end_date is not None:
        cutoff = (pd.Timestamp(end_date)
                  - pd.DateOffset(months=_DIGEST_MAX_AGE_MONTHS))
        tiles = tiles[tiles["_sent_dt"] >= cutoff]
    tiles = tiles.sort_values("_sent_dt", ascending=False).head(_DIGEST_MAX_TILES)

    if tiles.empty:
        window = (f" in the month before {_fmt_date(end_date)}"
                  if end_date is not None else "")
        st.info(f"No recent eNotices to display{window} for the selected "
                "units. Widen the date range or adjust your selection in the "
                "sidebar.")
    else:
        placeholder_summary = (
            "Placeholder summary: a concise two- to three-sentence AI-generated "
            "overview of this eNotice will appear here, highlighting its key "
            "details and purpose. Final wording is pending integration.")
        placeholder_tags = "".join(
            f'<span class="digest-tag">#{t}</span>'
            for t in ("PlaceholderOne", "PlaceholderTwo", "PlaceholderThree"))
        rows = [row for _, row in tiles.iterrows()]
        metas = [("https://enotice.vtools.ieee.org/public/" + str(r.get("id", "")),
                  str(r.get("event_url", "") or ""),
                  str(r.get("mailing_subject", "") or "")) for r in rows]

        # Summaries and tile images for the displayed notices, fetched/generated
        # in parallel and cached per notice so reruns don't re-spend credits.
        with st.spinner("Preparing recent eNotices..."):
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=_MAX_WORKERS) as pool:
                summaries = list(pool.map(
                    lambda m: enotice_summary(m[0], m[1]), metas))
                images = list(pool.map(
                    lambda m: enotice_image(m[0], m[1], m[2]), metas))

        blocks = []
        for row, (public_url, _ev, subj), summary, img in zip(
                rows, metas, summaries, images):
            recips = (parse_recipient_spoids(row.get("recipient_SPOIDs", ""))
                      & selected_norm)
            units_html = _join_units(recips, unit_info) or "—"
            sent = row["_sent_dt"]
            sent_txt = _fmt_date(sent) if pd.notna(sent) else ""
            subject = html.escape(subj)
            summary_html = _linkify(summary or placeholder_summary)
            thumb = (f'<div class="digest-thumb"><img src="{html.escape(img)}" '
                     f'alt=""></div>' if img
                     else '<div class="digest-thumb">🖼️</div>')
            blocks.append(
                '<div class="digest-tile">'
                f'{thumb}'
                '<div class="digest-body">'
                f'<div class="digest-units">{units_html}</div>'
                f'<div class="digest-sent">{sent_txt}</div>'
                f'<a class="digest-subject" href="{html.escape(public_url)}" '
                f'target="_blank">{subject}</a>'
                f'<div class="digest-summary">{summary_html}</div>'
                f'<div class="digest-tags">{placeholder_tags}</div>'
                '</div></div>')
        st.markdown("\n".join(blocks), unsafe_allow_html=True)

    agg = _join_units(selected_norm, unit_info)
    st.markdown(f'<div class="digest-agg">This digest aggregates content from '
                f'{agg}.</div>', unsafe_allow_html=True)
    if st.button("Manage your digest settings.", key="manage_settings"):
        _settings_dialog()
    st.markdown(f'<div class="digest-copyright">© {date.today().year} IEEE. '
                'All rights reserved.</div>', unsafe_allow_html=True)


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
        background-color: #3f7d8c !important;
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
                    background: linear-gradient(135deg, #eef3f7, #d9e2ea); }
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
    .digest-tag { display: inline-block; background: #eaf1f8; color: #00629B;
                  font-size: 0.8rem; padding: 0.1rem 0.55rem; border-radius: 12px;
                  margin: 0 0.35rem 0.25rem 0; }
    .digest-agg { text-align: center; color: #444; margin-top: 1.2rem;
                  font-size: 1rem; }
    .digest-copyright { text-align: center; color: #777; font-size: 1rem;
                        margin-top: 0.3rem; }
    /* Buttons styled as inline text links (view switch + manage settings). */
    .st-key-to_archive button, .st-key-to_digest button,
    .st-key-manage_settings button {
        color: #00629B !important; background: transparent !important;
        border: none !important; box-shadow: none !important;
        padding: 0 !important; min-height: 0 !important; font-weight: 400;
    }
    .st-key-to_archive button:hover, .st-key-to_digest button:hover,
    .st-key-manage_settings button:hover { text-decoration: underline; }
    .st-key-manage_settings { width: 100% !important; display: flex;
        justify-content: center; }
    .st-key-manage_settings button { font-size: 1rem !important; }
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
        with st.spinner("Finding related units..."):
            related = related_units(section_spoid)
            pool = recipient_pool(section_spoid, data_file)
            chapters_by_soc, sbc_links = society_links(section_spoid)
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
if st.session_state.view == "digest":
    render_digest_view(digest, selected_norm, unit_info, end_date)
    st.stop()

# --- Archive view ---
_archive_cols = st.columns([8, 2])
with _archive_cols[1]:
    st.button("← Digest", key="to_digest", on_click=_go_digest)

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
