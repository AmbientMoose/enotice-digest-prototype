# eNotice Digest (prototype)

A Streamlit prototype that turns IEEE **eNotice** data into a member‑facing
**digest**. Pick an eNotice data file, an IEEE Section, and the units/Societies a
member belongs to, and the app shows the notices that were sent to those units —
either as a tiled **digest** (with AI‑written summaries and pictures) or as a
sortable **archive** table.

> Prototype / proof of concept. It runs on exported eNotice CSVs plus the public
> IEEE OU List and Events APIs, and uses AI for each tile's summary, picture, and
> topical tags.

---

## Two views

**Digest view** (default) — an IEEE‑branded page showing the eNotices for the
selected units within the sidebar date range, newest first, six per page (page
through the rest with the Newer/Older controls). Each
tile has:

- a picture — pulled from the notice's public page (or the event's image from
  the Events API), and otherwise AI‑generated;
- the selected recipient units, linked to their websites;
- the date sent;
- the subject, linked to the notice's public URL;
- a 2–3 sentence AI summary (bare URLs become live links);
- AI‑generated topical tags (see [Tags](#tags) below);

followed by an aggregation statement, a "Manage your digest settings" link, and
a copyright line. A **Digest Archive** link switches to the archive view.

**Archive view** — the full, sortable table of matching eNotices with per‑column
filters, per‑unit "eNotices received" counts, and clickable `public_url` /
`event_url` links.

---

## How it works

- **Selecting units.** Starting from the chosen Section, the app walks the public
  **OU List API** (`vtools.vtools.ieee.org`) up through parents and down through
  children to build the set of related units, excluding *Academic / Grouping /
  Other* types. Parent/child edges are supplemented by `reciprocity_violations.csv`
  (the same approach as the [OU Explorer](https://github.com/AmbientMoose/ou-explorer)).
  By default only units that **received** at least one eNotice in the file are
  offered; a checkbox reveals the rest.
- **Societies.** Societies come from `units.csv`. Selecting one adds the Section's
  **Chapters / Joint Chapters** that are children of it; a **Student Branch
  Chapter** is added only once its Student Branch is also selected.
- **Matching notices.** A notice is included when one of the selected units'
  SPOIDs appears in its `recipient_SPOIDs` column (pipe‑delimited) and its status
  is `sent`, within the selected date range.
- **Event data.** When a notice has an `event_id`, the app pulls structured event
  details (title, description, tags, category, location, image) from the public
  **vTools Events API** (`events.vtools.ieee.org/api/public/v5/events/list`)
  rather than scraping the event page. This feeds the summary, image, and tag
  agents.
- **AI content.** For each digest tile, an Anthropic model (Claude Haiku)
  summarizes the notice from its public page plus the event's API record, and a
  vision check decides whether a candidate image (a page image, or the event's
  API image) is a suitable content picture. When none is, OpenAI `gpt-image-1`
  generates one from a Claude‑written prompt. All AI results are cached per
  notice, and the app degrades gracefully (placeholder summary/image/tags) when
  a key is missing or a call fails.
- **Preparation is on‑demand.** Because summaries, images, and tags are generated
  live when you open the digest, the view shows a per‑notice progress indicator
  ("Preparing eNotice *n* of *N*…"). In a production system this content would be
  created once, when each eNotice is published, so the digest would load
  instantly — the wait is a prototype artifact.

### Tags

Each tile's tags are generated per notice from the eNotice **subject**, its AI
**summary**, and the associated event's own **tags** (so they track what the
reader sees, not the full page text). The rules:

1. **Limit & relevance.** At most `_DIGEST_MAX_TAGS` (6) generated tags per
   eNotice (a compile‑time parameter). A tag is generated **only if** it
   satisfies one of rules 3–7, and every generated tag must be closely related
   to the eNotice's content (or its associated event).
2. **Normalization.** Tags are camel‑case with no separators; acronyms are
   preserved (`greenhouse-gas` → `#GreenhouseGas`, `STEM` stays `#STEM`).
3. **Category.** If the content clearly fits one, the matching **category** from
   `tag_categories.csv` is added (the file is loaded at startup, so the list can
   grow without code changes).
4. **Event tags.** The associated event's own tags (from the Events API, e.g.
   `#greenhouse-gas`) are normalized, deduped, and filtered to those closely
   related to the content. These are **exempt** from the six‑tag limit.
5. **IEEE Taxonomy.** For technical content, the topic(s) are matched against the
   **IEEE Taxonomy** (`taxonomy/ieee_taxonomy.csv`), leaning toward a single
   higher‑level term that covers multiple topics; a lower‑level term is used only
   if it is named in the content. The matched term counts toward the limit; the
   broader terms on its `full_path` are added **exempt**.
6. **Geography.** If the eNotice is tied to a specific place (e.g. the city of an
   in‑person event), a tag for that place is added.
7. **Conference/symposium.** If the notice is about a specific conference or
   symposium, a tag from its name is added: its short name/acronym (location
   suffix dropped, year kept — `ISIE2026-Nagoya` → `#ISIE2026`), or otherwise its
   full name with the `IEEE` and any `Nth` prefix removed (`IEEE 35th
   International Symposium on Industrial Electronics` →
   `#InternationalSymposiumOnIndustrialElectronics`). Near‑duplicate variants
   (e.g. `#IEEESoutheastCon2025` and `#SoutheastCon2025`) are deduped, keeping
   the shorter.

The category, conference/symposium, taxonomy‑leaf, and geography tags count
toward the six‑tag limit; the event's own tags and the taxonomy ancestors are
added on top and do **not** count against it.

**Hover a tag** to see its provenance — whether it was AI‑generated (and by which
rule) or imported from the associated event.

---

## Requirements

- Python 3.10+
- Network access to the IEEE OU List and vTools Events APIs and, for AI features,
  to the Anthropic and OpenAI APIs.
- Dependencies (`requirements.txt`): `streamlit`, `streamlit-searchbox`,
  `urllib3`, `pandas`, `anthropic`, `openai`.

## Setup and running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Secrets

Configuration lives in `.streamlit/secrets.toml` (git‑ignored — never commit it):

```toml
# Access password for the app gate (leave unset to run the app open)
app_password = "your-password"

# Enables AI summaries and page-image suitability checks
ANTHROPIC_API_KEY = "sk-ant-..."

# Enables AI image generation for tiles with no suitable page image
OPENAI_API_KEY = "sk-..."
```

All three are optional:

- Without `app_password`, the app runs without a login prompt.
- Without `ANTHROPIC_API_KEY`, tiles fall back to placeholder summaries/images.
- Without `OPENAI_API_KEY`, image generation is skipped (page images still work;
  otherwise the placeholder box is shown).

`app_password` / `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` can also be supplied as
environment variables (the app copies the API keys from secrets into the
environment on startup; the password gate also reads `APP_PASSWORD`).

### Data files

- `eNotice_data/*.csv` — the exported eNotice files; each one appears in the
  sidebar's data‑file picker. Expected columns include `id`, `mailing_subject`,
  `status`, `sent_at`, `owner_region`, `owner_section`, `recipient_SPOIDs`,
  `recipient_OUs`, `submitter_name`, `event_id`, `event_url` (plus per‑notice
  `Sent/Delivered/Bounced/Opened` stats, which the app hides).
- `units.csv` — the OU index (`spoid,name,type`) used for the Section/Society
  search and for unit names.
- `reciprocity_violations.csv` — parent/child edges the OU List API omits.
- `tag_categories.csv` — the controlled list of eNotice categories used for tag
  generation (one `category` per row); loaded at startup, so it can be extended
  without code changes.
- `taxonomy/ieee_taxonomy.csv` — the IEEE Taxonomy terms (`term,level,parent,
  term_family,full_path`) used to add technical tags and their ancestors. See
  [Taxonomy source](#taxonomy-source).
- `assets/ieee-logo.png` — the IEEE Master Brand + Tagline logo used in the header.

> **Data note.** The bundled eNotice CSVs have their `submitter_name` column
> **pseudonymized** (`Submitter NNNN`). The real‑name mapping lives in `private/`,
> which is git‑ignored and never committed.

### Taxonomy source

`taxonomy/ieee_taxonomy.csv` is derived from the **IEEE Taxonomy** (January 2026,
Version 1.05), published by IEEE at
<https://ieee-org.widen.net/s/jwk9pcxxvd/ieee-taxonomy>. The CSV flattens the
document's term hierarchy into one row per term — 8,037 terms across 51
term‑families — with each term's full ancestor path in the `full_path` column.
The source PDF is kept locally in `taxonomy/` (git‑ignored, `*.pdf`). The IEEE
Taxonomy is © IEEE.

## Deployment (Streamlit Community Cloud)

Point Streamlit Cloud at `app.py`. Because `.streamlit/secrets.toml` is not
committed, set `app_password`, `ANTHROPIC_API_KEY`, and `OPENAI_API_KEY` in the
app's **Secrets** UI. The repo is a public mirror so the app can be deployed from
GitHub.

## Configuration knobs

A few constants near the top of `app.py` tune behavior:

- `_DIGEST_MAX_TILES` (6) — how many eNotice tiles the digest shows per page
  (newest first); the user pages through the rest with the Newer/Older controls.
- `_DIGEST_MAX_TAGS` (6) — the maximum number of *generated* tags per tile (event
  tags and taxonomy ancestors are added on top and don't count against it).
- `_ANTHROPIC_MODEL` — the model used for summaries, tags, the vision suitability
  check, and image prompts.
- `_IMAGE_MODEL` / `_IMAGE_SIZE` / `_IMAGE_QUALITY` — OpenAI image generation.

## Project layout

```
app.py                     # the Streamlit app (both views, AI agents, gate)
ouclient.py                # IEEE OU List API client (from OU Explorer)
outype.py                  # OU type classification + styling (from OU Explorer)
units.csv                  # OU index (spoid, name, type)
reciprocity_violations.csv # supplemental parent/child edges
tag_categories.csv         # controlled category list for tag generation
taxonomy/ieee_taxonomy.csv # IEEE Taxonomy terms (for technical tags)
eNotice_data/              # exported eNotice CSVs (submitter names pseudonymized)
assets/ieee-logo.png       # IEEE Master Brand + Tagline logo
requirements.txt
.streamlit/secrets.toml    # git-ignored: password + API keys
private/                   # git-ignored: real submitter-name mapping
```
