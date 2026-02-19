import json
import requests
import trafilatura
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque

def crawl_website(url: str, max_depth: int = 2, max_pages: int = 10, same_domain: bool = True):
    """
    Robust website crawler with anti-block headers.
    Now safely handles string parameters from LLM tool calls.
    """
    # === SAFE TYPE CONVERSION (fixes the error) ===
    try:
        max_depth = int(max_depth)
    except (TypeError, ValueError):
        max_depth = 2
    try:
        max_pages = int(max_pages)
    except (TypeError, ValueError):
        max_pages = 10
    try:
        same_domain = str(same_domain).lower() in ('true', '1', 'yes')
    except:
        same_domain = True

    # Safety clamps
    max_depth = max(1, min(max_depth, 5))
    max_pages = max(1, min(max_pages, 30))

    visited = set()
    queue = deque([(url, 0)])
    results = []
    base_domain = urlparse(url).netloc.lower()

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.google.com/",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    while queue and len(results) < max_pages:
        current_url, depth = queue.popleft()
        if current_url in visited:
            continue
        visited.add(current_url)

        downloaded = None
        try:
            downloaded = trafilatura.fetch_url(current_url, headers=headers, timeout=15, decode=True)
        except:
            pass

        if not downloaded:
            try:
                r = requests.get(current_url, headers=headers, timeout=15, allow_redirects=True)
                r.raise_for_status()
                downloaded = r.text
            except:
                continue

        if not downloaded:
            continue

        text = trafilatura.extract(downloaded, include_comments=False, include_tables=True, no_fallback=False, favor_precision=True)
        if not text or len(text.strip()) < 80:
            continue

        soup = BeautifulSoup(downloaded, 'html.parser')
        title = soup.find('title').get_text(strip=True) if soup.find('title') else "No title"

        content = text[:7200] + ("..." if len(text) > 7200 else "")

        results.append({
            "url": current_url,
            "title": title[:250],
            "depth": depth,
            "content_length": len(text),
            "content": content
        })

        if depth < max_depth:
            for a in soup.find_all('a', href=True):
                link = a['href'].split('#')[0].split('?')[0].rstrip('/')
                next_url = urljoin(current_url, link)
                if not next_url.startswith(('http://', 'https://')):
                    continue
                parsed = urlparse(next_url)
                if same_domain and parsed.netloc.lower() != base_domain:
                    continue
                if any(ext in parsed.path.lower() for ext in ['.pdf','.jpg','.png','.gif','.zip','.css','.js','.svg']):
                    continue
                if next_url not in visited:
                    queue.append((next_url, depth + 1))

        time.sleep(0.8)

    return {
        "start_url": url,
        "pages_crawled": len(results),
        "max_depth_used": max_depth,
        "same_domain_only": same_domain,
        "results": results,
        "note": "Fixed version - now works with local models"
    }


# ====================== TOOL SCHEMA ======================
schema = {
    "type": "function",
    "function": {
        "name": "crawl_website",
        "description": "Crawl a whole website starting from one URL. Follows links and extracts clean readable text from multiple pages. Excellent for documentation sites, wikis, engineering pages, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Starting URL"},
                "max_depth": {"type": "integer", "description": "How deep to crawl (recommended 2)", "default": 2},
                "max_pages": {"type": "integer", "description": "Max pages to return (recommended 8-15)", "default": 10},
                "same_domain": {"type": "boolean", "description": "Stay on same domain", "default": True}
            },
            "required": ["url"]
        }
    }
}
