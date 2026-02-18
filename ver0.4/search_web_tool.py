import os
import requests

def search_web(query):
    """
    Search the web for information using SerpAPI.
    
    Requires 'SERPAPI_KEY' environment variable to be set.
    """
    api_key = os.environ.get("SERPAPI_KEY")
    
    if not api_key:
        return {
            "error": "Missing API Key. Please set the SERPAPI_KEY environment variable."
        }

    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "api_key": api_key,
        "engine": "google"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Parse organic results
        results = []
        if "organic_results" in data:
            for item in data["organic_results"][:5]:  # Limit to top 5 results
                results.append({
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "snippet": item.get("snippet")
                })
        
        # Check for knowledge graph (often contains quick answers)
        if "knowledge_graph" in data:
            kg = data["knowledge_graph"]
            results.insert(0, {
                "title": kg.get("title", "Knowledge Graph"),
                "snippet": kg.get("description", "No description available.")
            })

        if not results:
            return {"message": f"No results found for '{query}'"}

        return {"results": results}

    except requests.exceptions.RequestException as e:
        return {"error": f"Search request failed: {str(e)}"}


# Tool Schema
schema = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the web for current information, news, or facts.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query."
                }
            },
            "required": ["query"]
        }
    }
}
