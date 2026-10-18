"""Optional anonymous Parallel Search MCP caller for Fara-1.5."""

from __future__ import annotations

import asyncio

from ...version import __version__

SEARCH_MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = f"Fara/{__version__} (+https://github.com/microsoft/fara)"


async def parallel_search(query: str, *, session_id: str, timeout: float) -> str:
    """Return search excerpts and source URLs without moving the browser.

    Each request owns its transport, so timeout/cancellation closes the MCP
    session. The conversation identifier stays stable for the agent's task.
    No ambient API keys or saved MCP credentials are used.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("web_search requires a nonempty query")

    try:
        import httpx
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
    except ImportError as exc:
        raise RuntimeError(
            "Install Parallel search support with pip install 'fara[search]'."
        ) from exc

    async def request() -> str:
        async with httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT}, timeout=timeout
        ) as client:
            async with streamable_http_client(SEARCH_MCP_URL, http_client=client) as (
                read,
                write,
                _,
            ):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    if not any(tool.name == "web_search" for tool in tools.tools):
                        raise RuntimeError(
                            "Parallel Search MCP did not advertise web_search."
                        )
                    result = await session.call_tool(
                        "web_search",
                        {
                            "objective": query,
                            "search_queries": [query],
                            "session_id": session_id,
                        },
                    )
                    text = "\n".join(
                        item.text for item in result.content if item.type == "text"
                    )
                    if result.isError:
                        raise RuntimeError(
                            text or "Parallel Search MCP returned a tool error."
                        )
                    if not text.strip():
                        raise RuntimeError(
                            "Parallel Search MCP returned no search text."
                        )
                    return f"Parallel search results for {query!r}:\n{text}"

    try:
        return await asyncio.wait_for(request(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            f"Parallel search timed out after {timeout:g} seconds."
        ) from exc
    except Exception as exc:
        # The MCP transport can wrap server errors in task groups. Surface the
        # actual error instead of only saying that a task group failed.
        cause = exc
        while getattr(cause, "exceptions", None):
            cause = cause.exceptions[0]
        if cause is not exc:
            raise RuntimeError(str(cause)) from exc
        raise
