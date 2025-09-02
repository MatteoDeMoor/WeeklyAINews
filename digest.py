import os
import re
import hashlib
import textwrap
import yaml
import feedparser
import datetime as dt
import logging
import smtplib
from email.mime.text import MIMEText
from dateutil import parser as dparser
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

from tqdm import tqdm
import trafilatura
from jinja2 import Template
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

"""Weekly AI News Digest generator.

This script fetches news from RSS feeds, deduplicates similar stories,
classifies and ranks them, summarises with OpenAI (optional) and renders
a Markdown digest. When SMTP credentials are provided it can also send
the digest as an email.
"""

# ---- Settings ----
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIM_THRESHOLD = 0.86
SECTIONS = {
    "Model updates": r"(model|weights|gpt|gemini|claude|mixtral|llama|blackwell|rtx|inference|realtime)",
    "New AI tools and features": r"(launch|update|feature|plugin|extension|beta|preview|app|notebooklm|translate|editor|vids)",
    "Investments & Business": r"(raise|funding|series|revenue|acquire|merger|partnership|deal|ipo|earnings|stake)",
    "Ethical & Privacy": r"(privacy|copyright|lawsuit|policy|safety|guardrail|misuse|security|breach|teen|compliance|dsar|gdpr)"
}
MAX_PER_SECTION = 8

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

# ---- Utils ----
def load_feeds(fp: str = "config/feeds.yaml") -> List[str]:
    with open(fp, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["feeds"]

def norm_date(s: str) -> dt.date:
    try:
        return dparser.parse(s).date()
    except Exception:
        return dt.date.today()

def extract_text(url: str) -> str:
    try:
        downloaded = trafilatura.fetch_url(url, no_ssl=True)
        if not downloaded:
            return ""
        return trafilatura.extract(downloaded, include_comments=False, include_tables=False) or ""
    except Exception as exc:
        logging.warning("Failed to fetch %s: %s", url, exc)
        return ""

def sha(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def categorize(title: str, text: str) -> str:
    blob = f"{title}\n{text[:400]}".lower()
    for sec, pat in SECTIONS.items():
        if re.search(pat, blob):
            return sec
    return "New AI tools and features"

def rank_score(item: Dict, today: dt.date) -> float:
    days = max(0, (today - item["date"]).days)
    recency = max(0.0, 1.0 - min(days, 14) / 14.0)
    trust = 1.0 if any(k in item["link"] for k in [
        "openai.com", "ai.googleblog", "anthropic.com", "techcrunch.com",
        "theverge.com", "venturebeat.com", "arxiv.org"
    ]) else 0.5
    novelty = item.get("novelty", 0.7)
    return 0.45 * recency + 0.35 * trust + 0.20 * novelty

def safe_short(text: str, n: int = 220) -> str:
    text = " ".join(text.split())
    return textwrap.shorten(text, width=n, placeholder="…")

def send_email(subject: str, body: str) -> None:
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    pwd = os.getenv("SMTP_PASSWORD")
    to_addr = os.getenv("MAIL_TO")
    from_addr = os.getenv("MAIL_FROM", user)
    if not all([host, user, pwd, to_addr, from_addr]):
        logging.info("Email credentials missing, skipping mail send.")
        return
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    try:
        with smtplib.SMTP(host, port) as smtp:
            smtp.starttls()
            smtp.login(user, pwd)
            smtp.send_message(msg)
            logging.info("Mail sent to %s", to_addr)
    except Exception as exc:
        logging.error("Failed to send mail: %s", exc)

# ---- Ingest ----
def collect_entries(feeds: List[str]) -> List[Dict]:
    items = []
    for url in feeds:
        try:
            d = feedparser.parse(url)
        except Exception as exc:
            logging.warning("Failed to parse feed %s: %s", url, exc)
            continue
        for e in d.entries:
            link = e.get("link", "")
            title = e.get("title", "").strip()
            if not link or not title:
                continue
            items.append({
                "id": sha(link),
                "title": title,
                "link": link,
                "date": norm_date(e.get("published", e.get("updated", ""))),
                "source": re.sub(r"^https?://(www\.)?", "", link).split("/")[0]
            })
    return items

# ---- Deduplicate / cluster ----
def cluster_and_pick(items: List[Dict]) -> List[Dict]:
    try:
        model = SentenceTransformer(EMBED_MODEL)
        titles = [it["title"] for it in items]
        emb = model.encode(titles, normalize_embeddings=True)
        sim = cosine_similarity(emb)
        n = len(items)
        visited, clusters = set(), []
        for i in range(n):
            if i in visited:
                continue
            cluster = [i]
            for j in range(i + 1, n):
                if sim[i, j] >= SIM_THRESHOLD:
                    cluster.append(j)
                    visited.add(j)
            visited.add(i)
            clusters.append(cluster)

        picked = []
        for c in clusters:
            group = [items[i] for i in c]
            best = sorted(
                group,
                key=lambda x: (
                    -(
                        "arxiv.org" in x["link"]
                        or "openai.com" in x["link"]
                        or "ai.googleblog" in x["link"]
                    ),
                    -x["date"].toordinal(),
                ),
            )[0]
            for it in group:
                it["novelty"] = float(sim[c[0], c[-1]]) if len(c) > 1 else 0.7
            picked.append(best)
        return picked
    except Exception as exc:
        logging.error("Embedding/cluster step failed: %s", exc)
        # fallback: unique by link hash
        uniq = {}
        for it in items:
            uniq.setdefault(it["id"], it)
        return list(uniq.values())

# ---- Summaries ----
def llm_summary(client: OpenAI, text: str, title: str, url: str, date: dt.date) -> str:
    if not text.strip():
        return f"{title} — (Geen tekst kunnen extraheren). Lees meer: {url}"
    prompt = f"""Vat dit nieuws SAMENVATTEND samen in 2 zinnen (NL), met 1 emoji die het thema past.
Geef concrete cijfers/data indien in de tekst aanwezig. Noem expliciet de datum ({date.isoformat()}) en eindig met een korte duiding in 5–8 woorden.
Baseer je uitsluitend op onderstaande tekst en hallucineer niet.
Titel: {title}
URL: {url}

TEKST:
{text[:6000]}"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=160,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        logging.warning("LLM summarisation failed: %s", exc)
        return safe_short(text or title)

# ---- Render ----
MD_TEMPLATE = """Weekly AI News Digest — {{ today.isoformat() }}

{% for sec, items in sections.items() if items %}
- {{ sec }}
{% for it in items %}
{{ it['bullet'] }}
{% endfor %}
{% endfor %}
"""

def main() -> None:
    feeds = load_feeds()
    raw = collect_entries(feeds)
    logging.info("Fetched %d items", len(raw))
    today = dt.date.today()
    prelim = sorted(raw, key=lambda x: x["date"], reverse=True)[:200]

    deduped = cluster_and_pick(prelim)

    # Extract text concurrently
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(extract_text, it["link"]): it for it in deduped}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Extract"):
            it = futures[fut]
            it["text"] = fut.result()
            it["section"] = categorize(it["title"], it["text"])

    buckets = defaultdict(list)
    for it in deduped:
        it["score"] = rank_score(it, today)
        buckets[it["section"]].append(it)
    for sec in buckets:
        buckets[sec] = sorted(buckets[sec], key=lambda x: x["score"], reverse=True)[:MAX_PER_SECTION]

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
    for sec in buckets:
        for it in buckets[sec]:
            if client:
                summ = llm_summary(client, it["text"], it["title"], it["link"], it["date"])
            else:
                summ = safe_short(it["text"] or it["title"])
            it["bullet"] = f"{it['title']} ({it['date'].isoformat()}) — {summ} [{it['source']}]({it['link']})"

    tpl = Template(MD_TEMPLATE)
    md = tpl.render(today=today, sections=buckets)

    os.makedirs("out", exist_ok=True)
    outfp = f"out/digest_{today.isoformat()}.md"
    with open(outfp, "w", encoding="utf-8") as f:
        f.write(md)
    logging.info("Wrote %s", outfp)

    send_email(f"Weekly AI Digest {today.isoformat()}", md)

if __name__ == "__main__":
    main()
