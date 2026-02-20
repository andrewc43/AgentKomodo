import os
import requests
import json

def brave_search(query: str, num_results: int = 8) -> str:
    """
    Search Brave using the Brave Search API and return a JSON-formatted string
    in the same format google_search would (list of dicts).
    """

    if not query:
        return json.dumps([])

    # Get subscription token from environment
    BRAVE_SUBSCRIPTION_TOKEN = os.getenv("BRAVE_SUBSCRIPTION_TOKEN")
    if not BRAVE_SUBSCRIPTION_TOKEN:
        raise ValueError("BRAVE_SUBSCRIPTION_TOKEN not set in environment")

    url = "https://api.search.brave.com/res/v1/web/search"

    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": BRAVE_SUBSCRIPTION_TOKEN
    }

    params = {
        "q": query,
        "count": num_results  # Brave uses 'count' for number of results
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code != 200:
        # print raw output for debugging
        print("Brave API error:", response.status_code, response.text)
        response.raise_for_status()

    data = response.json()

    results = []
    # Pull the actual list of result objects from data['web']['results']
    for item in data.get("web", {}).get("results", []):
        results.append({
            "title": item.get("title"),
            "url": item.get("url"),
            "snippet": item.get("description") or item.get("snippet")
        })

    return json.dumps(results, ensure_ascii=False, indent=4)
