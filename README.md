# Weekly AI News Digest

Generates a weekly digest of AI news from a set of RSS feeds. Articles are de-duplicated, classified and ranked per section. Summaries are produced with OpenAI when an API key is provided. The resulting Markdown is saved to `out/` and can optionally be emailed.

## Setup

```bash
pip install -r requirements.txt
```

Edit `config/feeds.yaml` to add or remove RSS feeds.

Set environment variables for optional features:

- `OPENAI_API_KEY` – enable LLM summaries
- `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `MAIL_FROM`, `MAIL_TO` – send digest via email

## Usage

```bash
python digest.py
```

Output files are stored under `out/` using the current date.

## Automation

The GitHub Actions workflow `.github/workflows/digest.yml` runs every Monday at 06:00 UTC and commits the generated digest. Configure the secrets mentioned above in the repository settings to enable summaries and email delivery.
