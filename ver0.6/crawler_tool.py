import json
import requests
import trafilatura
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque

def crawl_website(
    url: str,
    max_depth: int = 2,
    max_pages: int = 10,
    same_domain: bool = True
):
    """
    Robust website crawler with anti-block headers + fallback.
    Extracts clean text using trafilatura + BeautifulSoup link discovery.
    """
    # Safety limits
    max_depth = max(1, min(max_depth, 5))
    max_pages = max(1, min(max_pages, 30))

    visited = set()
    queue = deque([(url, 0)])          # (url, depth)
    results = []
    base_domain = urlparse(url).netloc.lower()

    # Realistic browser headers (this fixes engineeringtoolbox.com and most protected sites)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
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

        # 1. Try trafilatura with good headers
        try:
            downloaded = trafilatura.fetch_url(
                current_url,
                headers=headers,
                timeout=15,
                decode=True
            )
        except:
            downloaded = None

        # 2. Fallback to raw requests if trafilatura fails
        if not downloaded:
            try:
                r = requests.get(current_url, headers=headers, timeout=15, allow_redirects=True)
                r.raise_for_status()
                downloaded = r.text
            except Exception as e:
                continue

        if not downloaded:
            continue

        # Extract clean readable text
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            no_fallback=False,      # allow fallback extractors
            favor_precision=True
        )

        if not text or len(text.strip()) < 80:
            continue

        # Get title
        soup = BeautifulSoup(downloaded, 'html.parser')
        title_tag = soup.find('title')
        title = title_tag.get_text(strip=True) if title_tag else "No title"

        # Truncate to safe size for local LLMs
        content = text[:7200] + ("..." if len(text) > 7200 else "")

        results.append({
            "url": current_url,
            "title": title[:250],
            "depth": depth,
            "content_length": len(text),
            "content": content
        })

        # Discover links for deeper crawling
        if depth < max_depth:
            for a in soup.find_all('a', href=True):
                link = a['href'].split('#')[0].split('?')[0].rstrip('/')
                next_url = urljoin(current_url, link)

                if not next_url.startswith(('http://', 'https://')):
                    continue

                parsed = urlparse(next_url)
                if same_domain and parsed.netloc.lower() != base_domain:
                    continue

                # Skip files we don't want
                if any(ext in parsed.path.lower() for ext in ['.pdf', '.jpg', '.png', '.gif', '.zip', '.css', '.js', '.svg']):
                    continue

                if next_url not in visited and next_url not in [q[0] for q in queue]:
                    queue.append((next_url, depth + 1))

        # Polite delay
        time.sleep(0.8)

    return {
        "start_url": url,
        "pages_crawled": len(results),
        "max_depth_used": max_depth,
        "same_domain_only": same_domain,
        "results": results,
        "note": "If pages_crawled == 0, the site may require JavaScript (rare). Try scrape_url on the exact page instead."
    }


# ====================== TOOL SCHEMA ======================
schema = {
    "type": "function",
    "function": {
        "name": "crawl_website",
        "description": "Crawl a whole website starting from one URL. Follows links, extracts clean readable text from multiple pages using anti-bot headers + fallback. Excellent for documentation, wikis, blogs, product catalogs, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Starting page URL (e.g. https://www.engineeringtoolbox.com/air-diffusion-coefficient-gas-mixture-temperature-d_2010.html)"
                },
                "max_depth": {
                    "type": "integer",
                    "description": "How many link levels to follow (1 = only start page, 2 = recommended, 3 = very deep)",
                    "default": 2
                },
                "max_pages": {
                    "type": "integer",
                    "description": "Maximum pages to return (recommended 8-15)",
                    "default": 10
                },
                "same_domain": {
                    "type": "boolean",
                    "description": "Stay on the same domain only (highly recommended)",
                    "default": True
                }
            },
            "required": ["url"]
        }
    }
}
