To make this agent truly powerful, a scraper tool is the perfect companion to a search tool. While search tools give you **snippets**, the scraper allows the agent to "read" the entire article for deep research.

Here is designed this using **Trafilatura**, which is widely considered the best library for AI agents because it ignores ads, headers, and footers, returning only the core text of the webpage.

### 1. The Professional Choice: `scraper_tool.py` (using Trafilatura)

This version provides the cleanest text for your Llama model to process.

**Setup:** Run `pip install trafilatura requests`

```python
import json
import requests
import trafilatura

def scrape_url(url):
    """
    Visits a URL and extracts the main text content.
    Ideal for reading full articles or documentation.
    """
    try:
        # Fetch the page content
        downloaded = trafilatura.fetch_url(url)
        
        if downloaded is None:
            return {"error": f"Could not download content from {url}. The site might be blocking automated requests."}

        # Extract the main body text
        # include_comments=False and include_tables=True help keep it relevant for AI
        result = trafilatura.extract(downloaded, include_comments=False, include_tables=True)

        if not result:
            return {"error": "Could not extract meaningful text from the page."}

        # Truncate to avoid hitting context limits of local models (e.g., ~8000 characters)
        return {
            "url": url,
            "content": result[:8000] + ("..." if len(result) > 8000 else "")
        }

    except Exception as e:
        return {"error": f"An error occurred while scraping: {str(e)}"}

# Tool Schema
schema = {
    "type": "function",
    "function": {
        "name": "scrape_url",
        "description": "Visit a specific URL to read its full content. Use this when search snippets are not enough.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL to scrape (e.g., https://example.com/article)."
                }
            },
            "required": ["url"]
        }
    }
}

```

---

### 2. The Lightweight Alternative (using BeautifulSoup)

If prefer not to install `trafilatura`, you can use this standard version. It's a bit "noisier" but very reliable for basic needs.

**Setup:** Run `pip install beautifulsoup4 requests`

```python
import requests
from bs4 import BeautifulSoup

def scrape_url(url):
    """Standard scraper using BeautifulSoup."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script or style in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # Get text and clean up whitespace
        text = soup.get_text(separator=' ')
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)

        return {"url": url, "content": text[:6000]}
    except Exception as e:
        return {"error": str(e)}

schema = {
    "type": "function",
    "function": {
        "name": "scrape_url",
        "description": "Extract text content from a webpage URL.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to scrape"}
            },
            "required": ["url"]
        }
    }
}

```

---

### How the Agent Uses These Together

When ask agent a question, the logic usually follows this flow:

1. **Step 1:** The Agent calls `search_web` to find relevant links.
2. **Step 2:** The Agent looks at the snippets and identifies the most promising URL.
3. **Step 3:** The Agent calls `scrape_url` to read the full details of that specific page.
4. **Step 4:** The Agent synthesizes all that data into final answer.

### Pro-Tips for Scraping

* **User-Agent:** Some websites block requests that don't look like a real browser. The BeautifulSoup code above includes a `headers` dictionary to help bypass this.
* **Context Limits:** Local models like Llama 3.1 8B have a limited "memory" (context window). Always truncate scraping results (as I did with `[:8000]`) to prevent the model from forgetting the beginning of conversation.
* **Rate Limiting:** Don't scrape too many pages from the same site too quickly, or IP might get temporarily blacklisted.

