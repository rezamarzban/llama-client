import requests
import xml.etree.ElementTree as ET
from urllib.parse import urlencode

def search_arxiv(query: str, max_results: int = 10):
    """
    Search arXiv official API — now with safe string-to-int conversion.
    """
    # === SAFE TYPE CONVERSION (fixes the error) ===
    try:
        max_results = int(max_results)
    except (TypeError, ValueError):
        max_results = 10

    max_results = min(max(1, max_results), 20)

    base_url = "http://export.arxiv.org/api/query?"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }

    url = base_url + urlencode(params)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"}

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

        results = []
        for entry in root.findall("atom:entry", ns):
            title = entry.find("atom:title", ns)
            summary = entry.find("atom:summary", ns)
            published = entry.find("atom:published", ns)

            authors = [a.find("atom:name", ns).text.strip() for a in entry.findall("atom:author", ns) if a.find("atom:name", ns) is not None]

            pdf_url = abs_url = ""
            for link in entry.findall("atom:link", ns):
                if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                    pdf_url = link.get("href")
                elif link.get("type") == "text/html":
                    abs_url = link.get("href")

            results.append({
                "title": (title.text.strip() if title is not None else "")[:320],
                "authors": authors[:10],
                "abstract": (summary.text.strip() if summary is not None else "")[:950] + ("..." if (summary is not None and len(summary.text or "") > 950) else ""),
                "abs_url": abs_url,
                "pdf_url": pdf_url,
                "published": (published.text[:10] if published is not None else ""),
                "arxiv_id": abs_url.split("/")[-1] if abs_url else ""
            })

        return {
            "query": query,
            "results_found": len(results),
            "results": results,
            "source": "arXiv Official API",
            "note": "Fixed version - works perfectly with local models"
        }

    except Exception as e:
        return {"error": f"arXiv search failed: {str(e)}"}


# ====================== TOOL SCHEMA ======================
schema = {
    "type": "function",
    "function": {
        "name": "search_arxiv",
        "description": "Search arXiv for scientific papers in physics, math, computer science, engineering, etc. Returns title, authors, abstract, PDF link.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query (supports ti:, au:, cat: etc.)"},
                "max_results": {"type": "integer", "description": "Number of papers (max 20)", "default": 10}
            },
            "required": ["query"]
        }
    }
}
