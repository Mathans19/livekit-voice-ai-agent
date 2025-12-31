import os
from tavily import TavilyClient

def web_search(query: str, max_results: int = 3) -> str:
    """
    100% Free web search using Tavily API.
    1000 free searches per month, no credit card required.
    Get your free API key at: https://tavily.com
    """
    try:
        print(f"🔍 Searching Tavily...")
        
        api_key = os.environ.get("TAVILY_API_KEY")
        
        if not api_key:
            print("❌ TAVILY_API_KEY not found in environment variables")
            print("Get your free key at: https://tavily.com")
            return ""
        
        client = TavilyClient(api_key=api_key)
        
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic"
        )
        
        results = response.get("results", [])
        
        if not results:
            print("⚠️ No results found")
            return ""
        
        snippets = []
        for r in results:
            title = r.get("title", "")
            content = r.get("content", "")
            if title and content:
                snippets.append(f"{title}: {content}")
        
        if snippets:
            print(f"✅ Got {len(snippets)} results")
            return "\n\n".join(snippets)
        
        return ""
        
    except Exception as e:
        print(f"❌ Search failed: {str(e)[:150]}")
        return ""