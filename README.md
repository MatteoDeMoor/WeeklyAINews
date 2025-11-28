# Weekly AI News Digest

Generates a weekly digest of AI news from a set of RSS feeds. Articles are de-duplicated, classified and ranked per section. Summaries are produced with OpenAI when an API key is provided. The resulting Markdown is saved to `out/` and can optionally be emailed.

## Prerequisites

- Python 3.11+ (tested with CPython; virtualenv recommended).
- System packages: `libxml2`/`libxslt` headers and `gcc` are commonly required for `lxml`, `trafilatura`, and `sentence-transformers` builds. On Debian/Ubuntu: `sudo apt-get install -y build-essential libxml2-dev libxslt1-dev`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Edit `config/feeds.yaml` to add or remove RSS feeds (see sample below).

### Environment

Create a `.env` file (optional) to keep secrets out of your shell:

```bash
OPENAI_API_KEY=sk-...
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USER=mailer@example.com
SMTP_PASSWORD=...
MAIL_FROM=mailer@example.com
MAIL_TO=you@example.com
# Optional overrides
DIGEST_OUT_DIR=out
DIGEST_DATE=2024-09-01
```

Minimal `.env.example`:

```bash
OPENAI_API_KEY=
SMTP_HOST=
SMTP_PORT=587
MAIL_TO=
```

Sample `config/feeds.yaml` block:

```yaml
feeds:
  - https://openai.com/blog/rss.xml
  - https://techcrunch.com/tag/ai/feed/
  - https://developer.nvidia.com/blog/feed/
```

## Running

Generate the digest locally:

```bash
python digest.py
```

- Without OpenAI summaries: leave `OPENAI_API_KEY` empty; headlines will use extracted snippets.
- With OpenAI summaries: export `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`); summaries and categorization will use the API.
- Custom output directory: `DIGEST_OUT_DIR=./reports python digest.py`
- Fixed date (useful for reruns/backfills): `DIGEST_DATE=2024-09-01 python digest.py`

Output Markdown is written to `<out_dir>/digest_<date>.md` (defaults to `out/` and today’s date).

Sample generated Markdown snippet:

```markdown
# Weekly AI News Digest — 2024-09-01

**Jump to:** [Investments & Business](#investments--business) • [Model updates](#model-updates)

## Model updates
- New model X launches with 8x faster inference (2024-08-31) - Concise summary here. [openai.com](https://openai.com/...)

---
```

## Troubleshooting / FAQ

- **HTTP 401/403/429 while fetching feeds** — Many sites block anonymous scraping. Retry later, ensure your IP is not rate-limited, or add feeds that provide stable RSS URLs.
- **Missing system dependencies during `pip install`** — Install `build-essential libxml2-dev libxslt1-dev` (Debian/Ubuntu) or equivalent developer tools, then reinstall requirements.
- **Empty or truncated summaries** — Check that `OPENAI_API_KEY` is set and reachable. If not using OpenAI, expect shorter heuristics.

## Automation

- GitHub Actions workflow `.github/workflows/digest.yml` runs every Monday at 06:00 UTC and commits the generated digest when secrets are configured.
- You can also schedule a local cron job, e.g., `0 7 * * 1 cd /path/to/WeeklyAINews && ./venv/bin/python digest.py`.
- To opt out locally, simply do not set up the cron entry (runs are manual-only), and disable the GitHub Actions workflow by pausing or removing the scheduled job in your fork.
