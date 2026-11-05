from fastapi.testclient import TestClient

from finrag import api


def test_health():
    assert TestClient(api.app).get("/health").json() == {"status": "ok"}


def test_search_endpoint(tiny_index, monkeypatch):
    monkeypatch.setattr(api, "get_index", lambda: tiny_index)
    monkeypatch.setattr(api, "get_router", lambda: None)
    r = TestClient(api.app).post("/search", json={"query": "purchases of property", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 2 and "latency_ms" in body
    assert body["results"][0]["citation"].endswith(tuple(f"p.{n}" for n in range(200)))


def test_search_validates_input(tiny_index, monkeypatch):
    monkeypatch.setattr(api, "get_index", lambda: tiny_index)
    monkeypatch.setattr(api, "get_router", lambda: None)
    client = TestClient(api.app)
    assert client.post("/search", json={"query": "x"}).status_code == 422
    assert client.post("/search", json={"query": "capex", "method": "magic"}).status_code == 422


def test_ask_without_key_returns_503(monkeypatch):
    def no_key():
        raise RuntimeError("Set ANTHROPIC_API_KEY to use the LLM agent.")
    monkeypatch.setattr(api, "get_agent", no_key)
    r = TestClient(api.app).post("/ask", json={"question": "What was 3M capex in 2018?"})
    assert r.status_code == 503
