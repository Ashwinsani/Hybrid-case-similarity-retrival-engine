"""
google_search.py — Google Custom Search API integration with semantic ranking.
Uses sentence-transformers to rank web results by semantic similarity to user query.
"""

import os
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import logging

load_dotenv()

logger = logging.getLogger(__name__)

# Lazy-load the model once
_model = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model


def google_search(query: str, num_results: int = 10) -> list:
    """
    Call Google Custom Search API and return raw results.
    
    Args:
        query: Search query string
        num_results: Number of results to fetch (max 10 per API call)
    
    Returns:
        List of dicts with title, snippet, link
    """
    api_key = os.getenv("API_KEY")
    cx_id = os.getenv("CX_ID")

    if not api_key or not cx_id:
        raise ValueError("API_KEY and CX_ID must be set in the .env file.")

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cx_id,
        "q": query,
        "num": min(num_results, 10)
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    results = []
    for item in data.get("items", []):
        results.append({
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "link": item.get("link", "")
        })

    logger.info(f"Google Search: fetched {len(results)} results for query: {query[:60]}")
    return results


def rank_google_results(user_input: str, google_results: list, top_k: int = 10) -> list:
    """
    Rank Google search results by semantic similarity to the user input.
    
    Args:
        user_input: The original user query / idea description
        google_results: Raw results from google_search()
        top_k: Maximum number of ranked results to return
    
    Returns:
        List of ranked results with a 'score' field added, sorted descending
    """
    if not google_results:
        return []

    model = _get_model()

    # Encode user query and all result texts
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    
    ranked = []
    for r in google_results:
        text = f"{r['title']} {r['snippet']}"
        result_embedding = model.encode(text, convert_to_tensor=True)
        score = float(util.cos_sim(user_embedding, result_embedding)[0][0])
        ranked.append({
            "title": r["title"],
            "snippet": r["snippet"],
            "link": r["link"],
            "score": round(score, 4)
        })

    # Sort by descending similarity score
    ranked.sort(key=lambda x: x["score"], reverse=True)

    return ranked[:top_k]
