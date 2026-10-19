"""Real web search via DuckDuckGo (no API key needed)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .base import Tool


class SearchArgs(BaseModel):
    query: str = Field(..., description="Search query")
    max_results: int = Field(5, ge=1, le=10)


def web_search(query: str, max_results: int = 5) -> str:
    try:
        from ddgs import DDGS
    except ImportError as exc:  # optional dependency
        raise RuntimeError("Web search needs the 'ddgs' package: pip install ddgs") from exc

    results = DDGS().text(query, max_results=max_results)
    if not results:
        return "No results found."
    return "\n\n".join(
        f"[{i}] {r.get('title', '')}\nURL: {r.get('href', '')}\n{r.get('body', '')}" for i, r in enumerate(results, 1)
    )


web_search_tool = Tool(
    name="web_search",
    description="Search the web for current information. Returns titles, URLs and snippets; cite URLs you use.",
    args_model=SearchArgs,
    fn=web_search,
)
