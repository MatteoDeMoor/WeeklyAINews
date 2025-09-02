import os
import re
import hashlib
import textwrap
import yaml
import feedparser
import datetime as dt
import logging
import smtplib
from bs4 import BeautifulSoup
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

# Load environment variables from .env file if present
from dotenv import load_dotenv
load_dotenv()

# ---- Settings ----
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIM_THRESHOLD = 0.86
RECENCY_WINDOW_DAYS = 7
MAX_PER_SECTION = 8

SECTIONS = {
    "Model updates": r"(model|weights|gpt|gemini|claude|mixtral|llama|blackwell|rtx|inference|realtime)",
    "New AI tools and features": r"(launch|update|feature|plugin|extension|beta|preview|app|notebooklm|translate|editor|vids)",
    "Investments & Business": r"(raise|funding|series|revenue|acquire|merger|partnership|deal|ipo|earnings|stake)",
    "Ethical & Privacy": r"(privacy|copyright|lawsuit|policy|safety|guardrail|misuse|security|breach|teen|compliance|dsar|gdpr)"
}

SKIP_EXTRACT_HOSTS = {
    "openai.com", "medium.com", "bloomberg.com", "wsj.com",
    "ft.com", "nytimes.com"
}

# ---- Logging ----
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("trafilatura").setLevel(logging.WARNING)

# ---- Utils ----
def load_feeds(fp: str = "config/feeds.yaml") -> List[str]:
    with open(fp, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["feeds"]
    
def normalize_feeds(feed_urls: List[str]) -> List[str]:
    normed = []
    for u in feed_urls:
        u = clean_url(u)
        if not u:
            continue
        d = feedparser.parse(u)
        status = getattr(d, "status", None)
        if (isinstance(status, int) and status >= 400) or (getattr(d, "bozo", 0) and not getattr(d, "entries", [])):
            wp = wp_feed_fallback(u)
            if wp:
                normed.append(wp)
                continue
            if u.endswith("/") or any(x in u for x in ["/blog", "/category", "/tag", "/news"]):
                disc = discover_feed_url(u)
                if disc:
                    normed.append(disc)
                    continue
        normed.append(u)
    return list(dict.fromkeys(normed))

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
    retry = Retry(
        total=2,
        backoff_factor=0.4,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        raise_on_status=False,
    )
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
                level = logging.WARNING if r.status_code in (401, 403, 429) else logging.ERROR
                logging.log(level, "Fetch %s returned %s", url, r.status_code)
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
        if host_of(url) in SKIP_EXTRACT_HOSTS:
            return ""
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
    # Use the 8-day window for recency scaling.
    recency = max(0.0, 1.0 - min(days, RECENCY_WINDOW_DAYS) / float(RECENCY_WINDOW_DAYS))
    trust = 1.0 if any(k in item["link"] for k in [
        "openai.com", "ai.googleblog", "anthropic.com", "techcrunch.com",
        "theverge.com", "venturebeat.com", "zdnet.com",
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
    to_raw = os.getenv("MAIL_TO", "")
    cc_raw = os.getenv("MAIL_CC", "")
    bcc_raw = os.getenv("MAIL_BCC", "")
    from_addr = os.getenv("MAIL_FROM", user)

    def parse_list(s: str) -> list[str]:
        return [a.strip() for a in s.split(",") if a.strip()]

    to_addrs = parse_list(to_raw)
    cc_addrs = parse_list(cc_raw)
    bcc_addrs = parse_list(bcc_raw)

    recipients = to_addrs + cc_addrs + bcc_addrs

    if not all([host, user, pwd, from_addr]) or not recipients:
        logging.info("Email settings incomplete or no recipients; skipping mail send.")
        return

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = from_addr
    if to_addrs:
        msg["To"] = ", ".join(to_addrs)
    if cc_addrs:
        msg["Cc"] = ", ".join(cc_addrs)

    try:
        with smtplib.SMTP(host, port) as smtp:
            smtp.starttls()
            smtp.login(user, pwd)
            smtp.send_message(msg, from_addr=from_addr, to_addrs=recipients)
            logging.info("Mail sent to: %s", ", ".join(recipients))
    except Exception as exc:
        logging.error("Failed to send mail: %s", exc)

def discover_feed_url(page_url: str) -> str:
    try:
        html = _fetch_html(page_url)
        if not html:
            return ""
        soup = BeautifulSoup(html, "lxml")
        alt = soup.find("link", rel=lambda x: x and "alternate" in x, type=lambda t: t and "rss" in t.lower() or "atom" in t.lower())
        if alt and alt.get("href"):
            return clean_url(alt["href"])
    except Exception:
        pass
    return ""

def wp_feed_fallback(url: str) -> str:
    try:
        parts = urlparse(url)
        # als het al eindigt op /feed of /feed/ doe niets
        if parts.path.rstrip("/").endswith("/feed"):
            return url
        candidate = urlunparse((parts.scheme, parts.netloc, parts.path.rstrip("/") + "/feed/", "", "", ""))
        # quick probe (HEAD/GET) via requests sessie
        if HTTP:
            r = HTTP.get(candidate, timeout=10)
            if r.status_code == 200 and ("<rss" in r.text or "<feed" in r.text):
                return candidate
    except Exception:
        pass
    return ""


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
        return f"{title} — (Could not extract text). Read more: {url}"
    prompt = f"""Summarize this news in exactly 2 concise sentences (EN). Include 1 fitting emoji.
If the text contains concrete numbers/dates, include them. Explicitly name the date ({date.isoformat()}) and end with a short 5–8 word take-away.
Rely only on the text below. Do not hallucinate.
Title: {title}
URL: {url}

TEXT:
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
        logging.warning("LLM summarization failed: %s", exc)
        return safe_short(text or title)

# ---- Summaries (robust) ----
def robust_llm_summary(client: OpenAI, text: str, title: str, url: str, date: dt.date) -> str:
    if not text or not text.strip():
        return f"{title} — (could not extract text). Read more: {url}"
    prompt = (
        "Summarize this news in exactly 2 concise sentences (EN) and include 1 fitting emoji.\n"
        f"Explicitly name the date ({date.isoformat()}) and end with a short 5–8 word take-away.\n"
        "Rely only on the text below. Do not hallucinate.\n"
        f"Title: {title}\n"
        f"URL: {url}\n\nTEXT:\n{text[:6000]}"
    )
    try:
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=180,
        )
        content = resp.choices[0].message.content or ""
        return content.strip() or safe_short(text or title)
    except Exception as exc:
        logging.warning("LLM summarization failed: %s", exc)
        return safe_short(text or title)

# ---- Render ----
NEW_MD_TEMPLATE = """# Weekly AI News Digest — {{ today.isoformat() }}

What happened in the last {{ window_days }} days?

{% for sec, items in sections.items() if items %}
## {{ sec }}
{% for it in items %}
- {{ it['bullet'] }}
{% endfor %}

---
{% endfor %}
"""

def main() -> None:
    feeds = load_feeds()
    feeds = normalize_feeds(feeds)
    raw = collect_entries(feeds)
    logging.info("Fetched %d items", len(raw))
    today = dt.date.today()

    # Keep only items from the last RECENCY_WINDOW_DAYS days
    cutoff = today - dt.timedelta(days=RECENCY_WINDOW_DAYS)
    recent = [it for it in raw if it["date"] >= cutoff]

    # Still sort by date and keep a reasonable cap
    prelim = sorted(recent, key=lambda x: x["date"], reverse=True)[:200]

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

    # Init OpenAI client from env, robustly
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    client = None
    if api_key:
        try:
            from httpx import Client as HTTPXClient
            httpx_kwargs = dict(timeout=30.0)
            proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
            if proxy:
                httpx_kwargs["proxies"] = proxy
            httpx = HTTPXClient(**httpx_kwargs)

            if base_url:
                client = OpenAI(api_key=api_key, base_url=base_url, http_client=httpx)
            else:
                client = OpenAI(api_key=api_key, http_client=httpx)
        except Exception as exc:
            logging.warning("Failed to init OpenAI client: %s", exc)
            client = None

    for sec in buckets:
        for it in buckets[sec]:
            if client:
                summ = robust_llm_summary(client, it["text"], it["title"], it["link"], it["date"])
            else:
                summ = safe_short(it["text"] or it["title"])
            it["bullet"] = f"{it['title']} ({it['date'].isoformat()}) — {summ} [{it['source']}]({it['link']})"

    tpl = Template(NEW_MD_TEMPLATE)
    md = tpl.render(today=today, sections=buckets, window_days=RECENCY_WINDOW_DAYS)

    os.makedirs("out", exist_ok=True)
    outfp = f"out/digest_{today.isoformat()}.md"
    with open(outfp, "w", encoding="utf-8") as f:
        f.write(md)
    logging.info("Wrote %s", outfp)

    # Mention the x-day window in the subject for clarity
    send_email(f"Weekly AI Digest {today.isoformat()} (last {RECENCY_WINDOW_DAYS} days)", md)

if __name__ == "__main__":
    main()
