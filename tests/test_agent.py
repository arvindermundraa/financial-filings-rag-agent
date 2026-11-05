from finrag.agent import FilingsAgent
from finrag.llm import ScriptedModel


def tool_call(id_, name, **inp):
    return {"type": "tool_use", "id": id_, "name": name, "input": inp}


def text(t):
    return {"type": "text", "text": t}


def test_agent_searches_calculates_and_cites(tiny_index):
    model = ScriptedModel([
        {"stop_reason": "tool_use", "content": [
            tool_call("t1", "search_filings", query="purchases of property plant equipment",
                      doc_name="3M_2018_10K")]},
        {"stop_reason": "tool_use", "content": [tool_call("t2", "calculator", expression="1577 / 32765 * 100")]},
        {"stop_reason": "end_turn", "content": [
            text("Capex was $1,577M, about 4.81% of net sales [3M_2018_10K p.59].")]},
    ])
    res = FilingsAgent(tiny_index, model, k=3).ask("What was 3M's FY2018 capex as % of sales?")
    assert res.steps == 3
    assert [c["name"] for c in res.tool_calls] == ["search_filings", "calculator"]
    assert res.citations == ["3M_2018_10K p.59"]
    assert res.unsupported_citations == []
    # the tool results were sent back to the model with matching ids
    sent = model.calls[2][-1]["content"][0]
    assert sent["type"] == "tool_result" and sent["tool_use_id"] == "t2"
    assert sent["content"].startswith("4.81")


def test_invented_citation_is_flagged(tiny_index):
    model = ScriptedModel([
        {"stop_reason": "end_turn", "content": [text("It was $9,999M [3M_2018_10K p.99].")]},
    ])
    res = FilingsAgent(tiny_index, model).ask("capex?")
    assert res.citations == []
    assert res.unsupported_citations == ["3M_2018_10K p.99"]


def test_step_limit_stops_runaway_loops(tiny_index):
    loop = {"stop_reason": "tool_use", "content": [tool_call("t", "calculator", expression="1+1")]}
    model = ScriptedModel([loop] * 10)
    res = FilingsAgent(tiny_index, model, max_steps=3).ask("loop forever")
    assert res.stopped_early and res.steps == 3


def test_bad_tool_input_returns_error_to_model(tiny_index):
    model = ScriptedModel([
        {"stop_reason": "tool_use", "content": [tool_call("t1", "calculator", expression="os.system('x')")]},
        {"stop_reason": "end_turn", "content": [text("I could not find this in the filings.")]},
    ])
    agent = FilingsAgent(tiny_index, model)
    agent.ask("bad")
    assert model.calls[1][-1]["content"][0]["content"].startswith("error:")


def test_router_applies_filter_when_model_does_not(tiny_index):
    from finrag.query import Router
    router = Router(["3M_2018_10K", "ADOBE_2017_10K"], {"3M": "3M", "Adobe": "ADOBE"})
    model = ScriptedModel([
        {"stop_reason": "tool_use", "content": [tool_call("t1", "search_filings", query="Adobe FY2017 purchases of property")]},
        {"stop_reason": "end_turn", "content": [text("done")]},
    ])
    res = FilingsAgent(tiny_index, model, router=router).ask("q")
    assert res.retrieved and all(c.startswith("ADOBE_2017_10K") for c in res.retrieved)
