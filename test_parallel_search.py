"""Backend/caller tests without loading embedding models or a web UI server."""
import ast
from dataclasses import dataclass
import importlib.util
import io
import json
import logging
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace
from typing import Dict
from unittest.mock import patch

import pytest
import requests

import parallel_search


ROOT = Path(__file__).parent


def load_definitions(filename, names, namespace):
    """Execute unmodified definitions, leaving optional model/UI imports unloaded."""
    tree = ast.parse((ROOT / filename).read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), namespace)
    return namespace


@pytest.fixture
def backend():
    utils = ModuleType("utils")
    definitions = load_definitions("utils.py", {"Document", "Generator"},
                                   {"dataclass": dataclass, "Dict": Dict})
    utils.__dict__.update(definitions)
    retrieval = ModuleType("retrieval")
    retrieval.DocumentRetriever = object
    spec = importlib.util.spec_from_file_location("isolated_llm_web_search", ROOT / "llm_web_search.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"utils": utils, "retrieval": retrieval}):
        spec.loader.exec_module(module)
    return module


class Retriever:
    num_results = 3
    client_timeout = 7

    def __init__(self):
        self.calls = []
        self.empty = False

    def retrieve_from_snippets(self, query, documents):
        self.calls.append(("snippets", query, documents))
        yield "Retrieving relevant results..."
        return [] if self.empty else documents

    def retrieve_from_webpages(self, query, urls):
        self.calls.append(("webpages", query, urls))
        yield "Downloading and chunking webpages..."
        return []  # Verify excerpt fallback without downloading/model inference.


SEARCH_CONTENT = {"results": [
    {"title": "Python releases", "url": "https://www.python.org/downloads/", "excerpts": ["Python release information."]},
    {"title": None, "url": "https://docs.python.org/3/", "excerpts": ["Python documentation."]},
]}


def rpc(request_id, result):
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


@pytest.fixture
def wire(monkeypatch):
    """Observe fully prepared requests; only replace the network send boundary."""
    calls = []
    replies = []

    def send(session, request, **kwargs):
        calls.append((request, kwargs))
        answer = replies.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        status, content_type, body = answer
        response = requests.Response()
        response.request = request
        response.status_code = status
        response.headers.update({"Content-Type": content_type, "Mcp-Session-Id": "transport-session"})
        response._content = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        response._content_consumed = True
        response.raw = io.BytesIO(response._content)
        return response

    monkeypatch.setattr(requests.Session, "send", send)

    def enqueue(result=None, sse=False):
        replies.extend([(200, "application/json", rpc(1, {"protocolVersion": "2025-03-26"})),
                        (202, "application/json", "")])
        answer = rpc(2, result if result is not None else {"structuredContent": SEARCH_CONTENT})
        body = json.dumps(answer)
        if sse:
            body = ': heartbeat\n\ndata: {"jsonrpc":"2.0","method":"notifications/progress"}\n\n' + \
                   "event: message\ndata: " + body + "\n\n"
        replies.append((200, "text/event-stream" if sse else "application/json", body))

    return SimpleNamespace(calls=calls, replies=replies, enqueue=enqueue)


def defaults():
    tree = ast.parse((ROOT / "script.py").read_text())
    return ast.literal_eval(next(node.value for node in tree.body if isinstance(node, ast.Assign)
                                 and any(isinstance(t, ast.Name) and t.id == "params" for t in node.targets)))


def native_tool(backend, retriever, params):
    extension = SimpleNamespace(params=params, retrieve_search=backend.retrieve_search,
                                Generator=backend.Generator, document_retriever=retriever)
    extensions = ModuleType("modules.extensions")
    extensions.state = {"LLM_Web_search": [extension]}
    spec = importlib.util.spec_from_file_location("isolated_search_tool", ROOT / "tool.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"modules": ModuleType("modules"), "modules.extensions": extensions}):
        spec.loader.exec_module(module)
    return module


def test_native_search_and_anonymous_wire(backend, wire, monkeypatch, tmp_path):
    netrc = tmp_path / "netrc"
    netrc.write_text("machine search.parallel.ai login saved password secret\n")
    monkeypatch.setenv("NETRC", str(netrc))
    monkeypatch.setenv("PARALLEL_API_KEY", "unused-saved-key")
    wire.enqueue()
    retriever = Retriever()
    params = defaults()
    assert params["use parallel search"] is False
    params.update({"use parallel search": True, "simple search": True, "search results per query": 1,
                   "searxng url": "https://saved.example", "client timeout": 7})
    output = native_tool(backend, retriever, params).execute({"query": "Python release information"})
    assert output["Results"] == [{"Content": "Title: Python releases\nPython release information.",
                                  "Source URL": "https://www.python.org/downloads/"}]
    assert retriever.calls[0][:2] == ("snippets", "Python release information")
    payloads = [json.loads(request.body) for request, _ in wire.calls]
    assert [p["method"] for p in payloads] == ["initialize", "notifications/initialized", "tools/call"]
    assert payloads[2]["params"] == {"name": "web_search", "arguments": {
        "objective": "Python release information", "search_queries": ["Python release information"],
        "session_id": parallel_search.SESSION_ID}}
    for request, options in wire.calls:
        assert request.url == "https://search.parallel.ai/mcp"
        assert request.headers["User-Agent"] == parallel_search.USER_AGENT
        assert "Authorization" not in request.headers
        assert options["timeout"] == 7
        assert options["stream"] is True
    assert wire.calls[2][0].headers["Mcp-Session-Id"] == "transport-session"
    assert wire.calls[2][0].headers["MCP-Protocol-Version"] == "2025-03-26"


@pytest.mark.parametrize("simple", [False, True])
def test_retrieval_and_excerpt_fallback(backend, wire, simple):
    wire.enqueue(sse=True)
    retriever = Retriever()
    retriever.empty = True
    gen = backend.Generator(backend.retrieve_search('"Python releases"', retriever, 2,
                                                    simple_search=simple, use_parallel=True))
    assert list(gen)[0] == "Getting results from Parallel..."
    assert len(gen.retval) == 2
    assert gen.retval[1].page_content.startswith("Title: https://docs.python.org/3/")
    assert retriever.calls[0][0] == ("snippets" if simple else "webpages")
    assert retriever.calls[0][1] == "Python releases"
    if not simple:
        assert retriever.calls[0][2] == [item["url"] for item in SEARCH_CONTENT["results"]]


def test_text_content_and_result_limit(wire):
    wire.enqueue({"content": [{"type": "text", "text": json.dumps(SEARCH_CONTENT)}]})
    assert len(parallel_search.perform_parallel_search("Python releases", max_results=1)) == 1


def test_empty_results_skip_local_retrieval(backend, wire):
    wire.enqueue({"structuredContent": {"results": []}})
    retriever = Retriever()
    gen = backend.Generator(backend.retrieve_search("query", retriever, 2, use_parallel=True))
    list(gen)
    assert gen.retval == []
    assert retriever.calls == []


def test_original_backend_selection(backend):
    retriever = Retriever()
    with patch.object(backend, "retrieve_from_duckduckgo", return_value="ddg") as ddg, \
            patch.object(backend, "retrieve_from_searxng", return_value="searx") as searx:
        assert backend.retrieve_search("query", retriever, 2) == "ddg"
        ddg.assert_called_once_with("query", retriever, 2, False)
        assert backend.retrieve_search("query", retriever, 2, True, "https://saved.example", False) == "searx"
        searx.assert_called_once_with("query", "https://saved.example", retriever, 2, False, True)


@pytest.mark.parametrize("failure", [requests.Timeout("search timed out"), requests.HTTPError("429 Too Many Requests")])
def test_native_errors_are_reported(backend, wire, failure):
    wire.enqueue()
    wire.replies[-1] = failure
    params = defaults()
    params["use parallel search"] = True
    assert "Error" in native_tool(backend, Retriever(), params).execute({"query": "Python releases"})
    assert len(wire.calls) == 3  # No retry or fallback to an unselected provider.


@pytest.mark.parametrize("result", [
    {"isError": True, "content": [{"type": "text", "text": "quota exceeded"}]},
    {"structuredContent": {"unexpected": []}},
    {"content": []},
])
def test_tool_and_malformed_result_errors(wire, result):
    wire.enqueue(result)
    with pytest.raises(ValueError):
        parallel_search.perform_parallel_search("Python releases")


def test_http_and_jsonrpc_errors(wire):
    wire.replies.append((429, "application/json", {"error": "quota"}))
    with pytest.raises(requests.HTTPError):
        parallel_search.perform_parallel_search("Python releases")
    wire.replies.append((200, "application/json", {"jsonrpc": "2.0", "id": 1,
                                                   "error": {"message": "invalid request"}}))
    with pytest.raises(ValueError, match="invalid request"):
        parallel_search.perform_parallel_search("Python releases")


def test_cancellation_does_not_start_network_or_swallow_interrupts(backend, wire):
    gen = backend.retrieve_search("Python releases", Retriever(), 2, use_parallel=True)
    assert next(gen) == "Getting results from Parallel..."
    gen.close()
    assert wire.calls == []
    wire.enqueue()
    wire.replies[-1] = KeyboardInterrupt()
    with patch.object(requests.Session, "close", autospec=True) as close:
        with pytest.raises(KeyboardInterrupt):
            parallel_search.perform_parallel_search("Python releases")
        close.assert_called_once()


def test_disabled_native_tool_never_searches(backend, wire):
    params = defaults()
    params.update({"use parallel search": True, "enable": False})
    assert "disabled" in native_tool(backend, Retriever(), params).execute({"query": "query"})["Error"]
    assert wire.calls == []


def test_legacy_command_consumes_parallel_results(backend, wire):
    wire.enqueue()
    params = defaults()
    params.update({"use parallel search": True, "simple search": True})
    shared = SimpleNamespace(model=object())

    def model_reply(question, *args, **kwargs):
        yield 'Search_web("Python releases")' if "Search tool:" not in question else "An answer using the search results."

    namespace = load_definitions("script.py", {"custom_generate_reply"}, {
        "params": params, "shared": shared, "document_retriever": Retriever(), "regex": re,
        "generate_reply_HF": model_reply, "generate_reply_custom": model_reply, "force_search": False,
        "logger": logging.getLogger(__name__), "Generator": backend.Generator,
        "retrieve_search": backend.retrieve_search, "GEN_LATENCY_THRESH": 0,
        "time": SimpleNamespace(sleep=lambda _: None),
        "docs_to_pretty_str": lambda docs: "\n".join(doc.page_content + "\n" + doc.metadata["source"] for doc in docs),
        "get_generation_prompt": lambda *args, **kwargs: ("", ""),
    })
    replies = list(namespace["custom_generate_reply"]("Find Python releases", "Find Python releases", {}, [], True))
    assert "Python release information." in replies[-1]
    assert "https://www.python.org/downloads/" in replies[-1]
    assert replies[-1].endswith("An answer using the search results.")
    assert len(wire.calls) == 3
