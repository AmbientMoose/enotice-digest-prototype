# eNotice Digest (prototype)

A Streamlit prototype that turns IEEE **eNotice** data into a member‑facing
**digest**. Pick an eNotice data file, an IEEE Section, and the units/Societies a
member belongs to, and the app shows the notices that were sent to those units —
either as a tiled **digest** (with AI‑written summaries and pictures) or as a
sortable **archive** table.

> Prototype / proof of concept. It runs on exported eNotice CSVs plus the public
> IEEE OU List API, and uses AI only for the per‑tile summary and image.

---

## Two views

**Digest view** (default) — an IEEE‑branded page showing up to six of the most
recent eNotices (none older than one month before the selected end date). Each
tile has:

- a picture — pulled from the notice's public page (or its event page), and
  otherwise AI‑generated;
- the selected recipient units, linked to their websites;
- the date sent;
- the subject, linked to the notice's public URL;
- a 2–3 sentence AI summary (bare URLs become live links);
- placeholder tags;

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
- **AI content.** For each digest tile, an Anthropic model (Claude Haiku)
  summarizes the notice from its fetched public/event pages, and a vision check
  decides whether a page image is a suitable content picture. When no suitable
  page image exists, OpenAI `gpt-image-1` generates one from a Claude‑written
  prompt. All AI results are cached per notice, and the app degrades gracefully
  (placeholder summary/image) when a key is missing or a call fails.

---

## Requirements

- Python 3.10+
- Network access to the IEEE OU List API and, for AI features, to the Anthropic
  and OpenAI APIs.
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
- `assets/ieee-logo.png` — the IEEE Master Brand + Tagline logo used in the header.

> **Data note.** The bundled eNotice CSVs have their `submitter_name` column
> **pseudonymized** (`Submitter NNNN`). The real‑name mapping lives in `private/`,
> which is git‑ignored and never committed.

## Deployment (Streamlit Community Cloud)

Point Streamlit Cloud at `app.py`. Because `.streamlit/secrets.toml` is not
committed, set `app_password`, `ANTHROPIC_API_KEY`, and `OPENAI_API_KEY` in the
app's **Secrets** UI. The repo is a public mirror so the app can be deployed from
GitHub.

## Configuration knobs

A few constants near the top of `app.py` tune behavior:

- `_DIGEST_MAX_TILES` (6) and `_DIGEST_MAX_AGE_MONTHS` (1) — how many recent
  notices the digest shows and how far back.
- `_ANTHROPIC_MODEL` — the model used for summaries, the vision suitability
  check, and image prompts.
- `_IMAGE_MODEL` / `_IMAGE_SIZE` / `_IMAGE_QUALITY` — OpenAI image generation.

## Project layout

```
app.py                     # the Streamlit app (both views, AI agents, gate)
ouclient.py                # IEEE OU List API client (from OU Explorer)
outype.py                  # OU type classification + styling (from OU Explorer)
units.csv                  # OU index (spoid, name, type)
reciprocity_violations.csv # supplemental parent/child edges
eNotice_data/              # exported eNotice CSVs (submitter names pseudonymized)
assets/ieee-logo.png       # IEEE Master Brand + Tagline logo
requirements.txt
.streamlit/secrets.toml    # git-ignored: password + API keys
private/                   # git-ignored: real submitter-name mapping
```
