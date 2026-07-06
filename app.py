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
import html
from collections import Counter
from datetime import timedelta
from pathlib import Path

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
# UI

st.title("📬 eNotice Digest")

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
