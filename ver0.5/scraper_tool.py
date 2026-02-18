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
