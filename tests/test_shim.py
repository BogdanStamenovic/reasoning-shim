import json
from reasoning_shim import classify, extract_calls, repair_message

TOOLS = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}}}}]

def test_extract_stranded_call():
    rc = "The listing shows config.yaml. I'll read it.\n\n<tool_call>\n<function=read_file>\n<parameter=path>\n/srv/app/config.yaml\n</parameter>\n<parameter=limit>\n40\n</parameter>\n</function>\n</tool_call>"
    before, calls = extract_calls(rc, TOOLS)
    assert before.endswith("read it.")
    assert calls[0]["function"]["name"] == "read_file"
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "/srv/app/config.yaml", "limit": 40}

def test_truncated_close_still_parsed():
    rc = "<tool_call>\n<function=read_file>\n<parameter=path>\n/a\n</parameter>\n</function>"
    assert extract_calls(rc, TOOLS)[1]

def test_unknown_tool_ignored():
    rc = "<tool_call>\n<function=rm_rf>\n<parameter=path>\n/\n</parameter>\n</function>\n</tool_call>"
    assert extract_calls(rc, TOOLS)[1] == []

def test_repair_skips_when_content_present():
    m = {"content": "answer", "reasoning_content": "<tool_call>\n<function=read_file>\n</function>\n</tool_call>"}
    assert not repair_message(m, TOOLS)

def test_classify():
    U = lambda t: [{"role": "user", "content": t}]
    assert classify(U("hi")) == "trivial"
    assert classify(U("thanks!")) == "trivial"
    assert classify(U("hey there")) == "trivial"
    assert classify(U("What is 2+2?")) == "default"
    assert classify(U("hi, can you design a rate limiter?")) == "default"
    assert classify(U("Find all integer solutions (x, y) with x^2 - y^2 = 45.")) == "default"
    assert classify(U("x") + [{"role": "tool", "content": "r"}]) == "tool"

def test_reasoning_effort_maps_to_budget():
    from reasoning_shim import pick_budget
    U = [{"role": "user", "content": "Make a title for this chat about fixing a config"}]
    assert pick_budget({"messages": U, "reasoning_effort": "none", "thinking_budget_tokens": -1}) == 0
    assert pick_budget({"messages": U, "reasoning_effort": "low"}) == 256
    assert pick_budget({"messages": U, "thinking_budget_tokens": 300}) is None
    assert pick_budget({"messages": U, "reasoning_effort": "high"}) == -1
