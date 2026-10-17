"""Anonymous Parallel Search MCP over Streamable HTTP, without an MCP runtime."""
import json
import time
import uuid

import requests


MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "LLM_Web_search (https://github.com/mamei16/LLM_Web_search)"
PROTOCOL_VERSION = "2025-03-26"
# Reuse one quota/correlation identity throughout this extension process.
SESSION_ID = str(uuid.uuid4())


def _read_reply(response, request_id, timeout):
    if response.headers.get("Content-Type", "").split(";", 1)[0] == "text/event-stream":
        deadline = time.monotonic() + timeout
        data = []
        for line in response.iter_lines(chunk_size=1):
            if time.monotonic() > deadline:
                raise requests.Timeout("Parallel search response timed out")
            if not line:
                if data:
                    reply = json.loads("\n".join(data))
                    if reply.get("id") == request_id:
                        return reply
                    data = []
            elif line.startswith(b"data:"):
                data.append(line[5:].lstrip().decode("utf-8"))
        raise ValueError("Parallel search returned no matching MCP response")
    reply = response.json()
    if not isinstance(reply, dict) or reply.get("id") != request_id:
        raise ValueError("Parallel search returned an invalid MCP response")
    return reply


def _reject_redirects(response, **kwargs):
    if 300 <= response.status_code < 400:
        # Session.send prepares Response.next even when redirect following is off.
        # A response hook stops that preparation before it can consult netrc.
        response.close()
        raise requests.HTTPError("Parallel search does not allow HTTP redirects", response=response)
    return response


def _request(session, method, params, request_id, timeout):
    payload = {"jsonrpc": "2.0", "method": method, "params": params}
    if request_id is not None:
        payload["id"] = request_id
    with session.post(MCP_URL, json=payload, timeout=timeout, stream=True, allow_redirects=False,
                      hooks={"response": _reject_redirects}) as response:
        _reject_redirects(response)
        response.raise_for_status()
        if response.headers.get("Mcp-Session-Id"):
            session.headers["Mcp-Session-Id"] = response.headers["Mcp-Session-Id"]
        if request_id is None:
            return None
        reply = _read_reply(response, request_id, timeout)
    if "error" in reply:
        raise ValueError(f"Parallel search MCP error: {reply['error'].get('message', 'Unknown error')}")
    if not isinstance(reply.get("result"), dict):
        raise ValueError("Parallel search returned no MCP result")
    return reply["result"]


def perform_parallel_search(query, max_results=10, timeout=10):
    """Return title/url/content records, or propagate HTTP, timeout and MCP errors."""
    if max_results <= 0:
        return []
    with requests.Session() as session:
        session.headers.update({"User-Agent": USER_AGENT,
                                "Accept": "application/json, text/event-stream"})
        # This backend is anonymous, including when the user has a .netrc file.
        # Leave proxy and certificate environment settings available to requests.
        session.auth = lambda request: request
        initialized = _request(session, "initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "LLM_Web_search", "version": "1.0"},
        }, 1, timeout)
        session.headers["MCP-Protocol-Version"] = initialized["protocolVersion"]
        _request(session, "notifications/initialized", {}, None, timeout)
        result = _request(session, "tools/call", {
            "name": "web_search",
            "arguments": {"objective": query, "search_queries": [query], "session_id": SESSION_ID},
        }, 2, timeout)

    if result.get("isError"):
        message = "\n".join(block.get("text", "") for block in result.get("content", [])
                            if block.get("type") == "text")
        raise ValueError(f"Parallel search failed: {message}")
    content = result.get("structuredContent")
    if content is None:
        # Older MCP servers expose the same structured object as a text block.
        blocks = [block["text"] for block in result.get("content", []) if block.get("type") == "text"]
        if not blocks:
            raise ValueError("Parallel search returned no search content")
        content = json.loads("\n".join(blocks))
    if not isinstance(content, dict) or not isinstance(content.get("results"), list):
        raise ValueError("Parallel search returned an invalid search result")

    results = []
    for item in content["results"][:max_results]:
        results.append({"title": item.get("title") or item["url"], "url": item["url"],
                        "content": "\n".join(item["excerpts"])})
    return results
