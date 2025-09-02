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
from html import unescape as html_unescape
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
try:
    import requests
    from urllib3.util.retry import Retry
    from requests.adapters import HTTPAdapter
except Exception:
    requests = None

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
# reduce noisy third-party logs
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("trafilatura").setLevel(logging.WARNING)

# ---- Utils ----
def load_feeds(fp: str = "config/feeds.yaml") -> List[str]:
    with open(fp, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["feeds"]

def norm_date(s: str) -> dt.date:
    try:
        return dparser.parse(s).date()
    except Exception:
        return dt.date.today()

def clean_url(url: str) -> str:
    try:
        url = (url or "").strip()
        if not url:
            return ""
        parts = urlparse(url)
        # strip common tracking params
        q = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=False)
             if not k.lower().startswith("utm_") and k.lower() not in {"fbclid", "gclid"}]
        new_query = urlencode(q)
        return urlunparse((parts.scheme, parts.netloc, parts.path, parts.params, new_query, parts.fragment))
    except Exception:
        return url or ""

def host_of(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc[4:] if netloc.startswith("www.") else netloc
    except Exception:
        return ""

def _build_session():
    if requests is None:
        return None
    sess = requests.Session()
    sess.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,nl;q=0.8",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    })
    retry = Retry(total=2, backoff_factor=0.4,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET", "HEAD"), raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retry, pool_connections=30, pool_maxsize=30)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    return sess

HTTP = _build_session()

def _fetch_html(url: str) -> str:
    if HTTP is not None:
        try:
            r = HTTP.get(url, timeout=15)
            if r.status_code != 200:
                logging.error("not a 200 response: %s for URL %s", r.status_code, url)
                return ""
            return r.text
        except Exception as exc:
            logging.warning("requests fetch failed for %s: %s", url, exc)
            return ""
    try:
        downloaded = trafilatura.fetch_url(url, no_ssl=True)
        return downloaded or ""
    except Exception as exc:
        logging.warning("trafilatura.fetch_url failed for %s: %s", url, exc)
        return ""

def extract_text(url: str) -> str:
    try:
        html = _fetch_html(url)
        if not html:
            return ""
        return trafilatura.extract(html, url=url, include_comments=False, include_tables=False) or ""
    except Exception as exc:
        logging.warning("Failed to extract %s: %s", url, exc)
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
            status = getattr(d, "status", None)
            if isinstance(status, int) and status >= 400:
                logging.warning("Skip feed %s (HTTP %s)", url, status)
                continue
            if getattr(d, "bozo", 0) and not getattr(d, "entries", []):
                logging.warning("Bozo feed %s: %s", url, getattr(d, "bozo_exception", ""))
                continue
        except Exception as exc:
            logging.warning("Failed to parse feed %s: %s", url, exc)
            continue
        if not getattr(d, "entries", None):
            logging.warning("No entries in feed %s", url)
            continue
        for e in d.entries:
            link = clean_url(e.get("link", ""))
            title = html_unescape(e.get("title", "")).strip()
            if not link or not title:
                continue
            # fallback description from feed
            desc = e.get("summary") or e.get("description") or ""
            if not desc and e.get("content"):
                try:
                    desc = e.get("content")[0].get("value", "")
                except Exception:
                    pass
            desc = html_unescape(re.sub(r"<[^>]+>", " ", desc))
            desc = " ".join(desc.split())
            items.append({
                "id": sha(link),
                "title": title,
                "link": link,
                "date": norm_date(e.get("published", e.get("updated", ""))),
                "source": host_of(link),
                "desc": desc,
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

# Newer, cleaner Markdown template (keeps old for compatibility)
NEW_MD_TEMPLATE = """Weekly AI News Digest — {{ today.isoformat() }}

{% for sec, items in sections.items() if items %}
**{{ sec }}**
{% for it in items %}
- {{ it['bullet'] }}
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
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(extract_text, it["link"]): it for it in deduped}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Extract"):
            it = futures[fut]
            it["text"] = fut.result() or it.get("desc", "")
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
            it["bullet"] = f"{it['title']} ({it['date'].isoformat()}) — {summ} [{it['source']}]({it['link']})"

    # ensure bullet format is clean (override any earlier assignment)
    for sec in buckets:
        for it in buckets[sec]:
            if "bullet" in it:
                it["bullet"] = f"{it['title']} ({it['date'].isoformat()}) — {it.get('bullet').split(')')[-1].strip()}"
    tpl = Template(NEW_MD_TEMPLATE)
    md = tpl.render(today=today, sections=buckets)

    os.makedirs("out", exist_ok=True)
    outfp = f"out/digest_{today.isoformat()}.md"
    with open(outfp, "w", encoding="utf-8") as f:
        f.write(md)
    logging.info("Wrote %s", outfp)

    send_email(f"Weekly AI Digest {today.isoformat()}", md)

if __name__ == "__main__":
    main()
