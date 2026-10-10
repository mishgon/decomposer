from copy import deepcopy
import re
from xml.etree import ElementTree

import pytest

import decomposer.visualization as visualization
from decomposer.visualization import render_trace, write_trace_html


def _trace():
    agents = [
        {"agent_id": "child", "agent_type_id": "researcher", "created_at": 110.0,
         "forked_from": "root"},
        {"agent_id": "idle", "agent_type_id": "reviewer", "created_at": 119.0},
        {"agent_id": "root", "agent_type_id": "researcher", "created_at": 100.0},
        {"agent_id": "failed", "agent_type_id": "writer", "created_at": 104.0},
    ]
    runs = [
        {"agent_run_id": "root-2", "agent_id": "root", "started_at": 111.0,
         "collected_at": 118.0, "status": "responded", "prompt": "Compare the findings.",
         "response": "The two approaches agree."},
        {"agent_run_id": "root-1", "agent_id": "root", "started_at": 102.0,
         "collected_at": 108.0, "status": "responded", "prompt": "Find two approaches.",
         "response": "Here are two approaches."},
        {"agent_run_id": "child-1", "agent_id": "child", "started_at": 112.0,
         "collected_at": 116.0, "status": "responded", "prompt": "Check the second approach.",
         "response": "The second approach checks out."},
        {"agent_run_id": "failed-1", "agent_id": "failed", "started_at": 105.0,
         "collected_at": 115.0, "status": "error", "prompt": "Write a draft.",
         "error": "The writing tool failed."},
        {"agent_run_id": "root-3", "agent_id": "root", "started_at": 120.0,
         "collected_at": 124.0, "status": "responded", "prompt": "Prepare the final answer.",
         "response": "The final answer is ready."},
    ]
    return {
        "decomposer_agent_runs": [{
            "agent_run_id": "decomposer-1", "agent_id": "decomposer", "run_id": "decomposer-1",
            "started_at": 90.0, "collected_at": 130.0, "status": "responded",
            "prompt": "Compare two approaches.", "response": "Here is the comparison.",
        }],
        "messages": [
            {"type": "human", "content": "Compare two approaches."},
            {"type": "ai", "content": "Here is the comparison."},
        ],
        "agents": {agent["agent_id"]: agent for agent in agents},
        "agent_runs": {run["agent_run_id"]: run for run in runs},
    }


def _svg(document):
    return ElementTree.fromstring(document[document.index("<svg "):document.index("</svg>") + 6])


def _templates(document):
    return {
        element.attrib["id"]: element
        for element in map(ElementTree.fromstring, re.findall(r"<template .*?</template>", document, re.S))
    }


def _marker_positions(run):
    return [
        tuple(map(float, event.attrib["transform"].removeprefix("translate(").removesuffix(")").split()))
        for event in run.findall("{*}g[@class='event']")
    ]


def test_optional_html_failure_is_reported(tmp_path, monkeypatch, caplog):
    def fail(trace):
        raise ValueError("broken renderer")

    monkeypatch.setattr(visualization, "render_trace", fail)
    path = tmp_path / "trace.html"
    assert write_trace_html(_trace(), path) is False
    assert not path.exists()
    assert "broken renderer" in caplog.text
    assert str(path) in caplog.text


def test_run_times_rows_and_fork_geometry():
    trace = _trace()
    trace["agent_runs"]["root-1"]["finished_at"] = 104.0
    svg = _svg(render_trace(trace))

    assert [row.attrib["data-agent-id"] for row in svg.findall("{*}g[@class='agent']")] == [
        "root", "failed", "child", "idle",
    ]
    first = _marker_positions(svg.find("{*}g[@data-run-id='root-1']"))
    assert first[0] == pytest.approx((532.4, 162))
    assert first[1] == pytest.approx((668.6, 214))
    second = _marker_positions(svg.find("{*}g[@data-run-id='root-2']"))
    assert [y for _, y in second] == [162, 214]
    assert second[0][0] > first[1][0]
    fork = svg.find("{*}path[@class='fork']")
    parent_bar = svg.find("{*}g[@data-run-id='root-1']/{*}line[@class='run-line']")
    assert fork.attrib["d"] == (
        f'M {parent_bar.attrib["x2"]} 188 H 702.00 '
        'Q 714.00 188 714.00 200.00 V 364'
    )
    assert not svg.findall(".//{*}line[@class='lifeline']")
    parent_name = svg.find("{*}g[@data-agent-id='root']/{*}text[@class='agent-name']").text
    assert fork.find("{*}title").text == f"Форк от агента «{parent_name}»"
    assert len(svg.findall("{*}circle[@class='created']")) == 4
    assert [tick.text for tick in svg.findall("{*}text[@class='tick']")] == [
        "0 с", "8 с", "16 с", "24 с", "32 с", "40 с",
    ]


def test_animals_are_distinct_and_stable_across_runs_and_renders():
    trace = _trace()
    original = deepcopy(trace)
    document = render_trace(trace)
    assert trace == original
    assert render_trace(trace) == document
    reversed_trace = {
        **trace,
        "agents": dict(reversed(trace["agents"].items())),
        "agent_runs": dict(reversed(trace["agent_runs"].items())),
    }
    assert render_trace(reversed_trace) == document
    svg = _svg(document)
    animals = {
        row.attrib["data-agent-id"]: row.find("{*}text[@class='animal']").text
        for row in svg.findall("{*}g[@class='agent']")
    }
    assert len(set(animals.values())) == 4
    assert "🐶" not in animals.values()
    labels = [row.text for row in svg.findall("{*}g[@class='agent']/{*}text[@class='agent-name']")]
    assert len(set(labels)) == 4
    assert not svg.findall(".//{*}text[@class='agent-id']")
    for agent in trace["agents"].values():
        assert agent["agent_id"] not in "".join(svg.itertext())
        assert agent["agent_type_id"] not in "".join(svg.itertext())
    for run_id in ("root-1", "root-2"):
        events = svg.findall(f"{{*}}g[@data-run-id='{run_id}']/{{*}}g[@class='event']/{{*}}text")
        assert [event.text for event in events] == ["🐶", animals["root"]]


def test_uncollected_and_failed_runs():
    trace = _trace()
    trace["decomposer_agent_runs"][0].pop("collected_at")
    trace["decomposer_agent_runs"][0]["status"] = "running"
    trace["agent_runs"]["root-3"].pop("collected_at")
    trace["agent_runs"]["root-3"]["status"] = "running"
    document = render_trace(trace)
    svg = _svg(document)
    templates = _templates(document)
    pending = svg.find("{*}g[@data-run-id='root-3']")
    assert "uncollected" in pending.attrib["class"]
    assert len(pending.findall("{*}g[@class='event']")) == 1
    invocation = svg.find("{*}g[@class='invocation uncollected']")
    assert len(invocation.findall("{*}g[@class='event']")) == 1
    failed = svg.find("{*}g[@data-run-id='failed-1']")
    assert "failed" in failed.attrib["class"]
    result = failed.findall("{*}g[@class='event']")[1]
    tooltip = templates[result.attrib["data-tooltip"]]
    assert "".join(tooltip.itertext()).strip() == "The writing tool failed."


@pytest.mark.parametrize(("field", "status"), [("response", "responded"), ("error", "error")])
def test_trace_text_is_escaped(field, status):
    payload = '\n</pre></template><script>alert("x")</script><img src=x onerror="alert(1)">\n& 🐶'
    payload += "\nПолное сообщение без сокращений." * 300
    identifier = payload.replace("\n", " ")
    trace = {
        "decomposer_agent_runs": [{
            "agent_run_id": "decomposer-1", "agent_id": "decomposer", "run_id": "decomposer-1",
            "started_at": 0.0, "collected_at": 4.0, "status": "responded",
            "prompt": payload, "response": payload,
        }],
        "agents": {identifier: {"agent_id": identifier, "agent_type_id": payload, "created_at": 1.0}},
        "agent_runs": {identifier: {
            "agent_run_id": identifier, "agent_id": identifier, "started_at": 2.0,
            "collected_at": 3.0, "status": status, "prompt": payload, field: payload,
        }},
    }
    document = render_trace(trace)
    svg = _svg(document)
    assert svg.find("{*}g[@class='agent']").attrib["data-agent-id"] == identifier
    assert not svg.findall(".//{*}script")
    assert "<img" not in document
    assert document.count("<script>") == 1
    templates = _templates(document)
    assert len(templates) == 4
    assert all("".join(tooltip.itertext()).strip() == payload.strip() for tooltip in templates.values())


def test_all_messages_are_stripped_and_rendered_as_markdown():
    content = (
        ' \n\t\n  ## Заголовок\n\n'
        '**Важно**, *курсив*, ~~старое~~ и `x < 2`.\nНовая строка.\n\n'
        '- Первый\n  - Вложенный\n- Второй\n\n'
        '1. Шаг один\n2. Шаг два\n\n'
        '> Цитата\n\n'
        '```python\nif True:\n    print("<tag>")\n```\n\n'
        '| Имя | Значение |\n| --- | --- |\n| Ответ | 42 |\n\n'
        '[Ссылка](https://example.com/)\n\n \t '
    )
    trace = _trace()
    for run in [*trace["decomposer_agent_runs"], *trace["agent_runs"].values()]:
        run["prompt"] = content
        run["error" if run["status"] == "error" else "response"] = content
    templates = _templates(render_trace(trace))

    assert len(templates) == 2 + 2 * len(trace["agent_runs"])
    for tooltip in templates.values():
        message = tooltip.find("div")
        assert message.text is None
        assert message[0].tag == "h2"
        assert message[0].text == "Заголовок"
        assert message.find("p/strong").text == "Важно"
        assert message.find("p/em").text == "курсив"
        assert message.find("p/s").text == "старое"
        assert message.find("p/code").text == "x < 2"
        assert message.find("p/br") is not None
        assert message.find("ul/li/ul/li").text == "Вложенный"
        assert len(message.findall("ol/li")) == 2
        assert message.find("blockquote/p").text == "Цитата"
        assert message.find("pre/code").text == 'if True:\n    print("<tag>")\n'
        assert message.find("table/tbody/tr/td[2]").text == "42"
        assert message[-1].find("a").attrib["href"] == "https://example.com/"


@pytest.mark.parametrize("url", ["javascript:alert(1)", "data:text/html,<script>alert(1)</script>"])
def test_markdown_does_not_create_executable_links(url):
    trace = _trace()
    trace["decomposer_agent_runs"][0]["prompt"] = f"[Ссылка]({url})"
    document = render_trace(trace)
    svg = _svg(document)
    event = svg.find("{*}g[@class='invocation ']/{*}g[@class='event']")
    tooltip = _templates(document)[event.attrib["data-tooltip"]]

    assert not tooltip.findall(".//a")
    assert not tooltip.findall(".//script")
    assert "Ссылка" in "".join(tooltip.itertext())


def test_invocation_without_agents():
    trace = _trace()
    trace.update(agents={}, agent_runs={})
    svg = _svg(render_trace(trace))
    assert "Моська" in "".join(svg.itertext())
    assert len(svg.findall("{*}g[@class='invocation ']/{*}g[@class='event']")) == 2
    assert not svg.findall("{*}g[@class='agent']")


def test_zero_duration_run():
    trace = _trace()
    trace["agents"] = {"root": trace["agents"]["root"]}
    trace["agent_runs"] = {"root-1": trace["agent_runs"]["root-1"]}
    trace["agent_runs"]["root-1"].update(started_at=100.0, collected_at=100.0)
    trace["decomposer_agent_runs"][0].update(started_at=100.0, collected_at=100.0)
    svg = _svg(render_trace(trace))
    run = svg.find("{*}g[@data-run-id='root-1']")
    assert _marker_positions(run) == [(260, 162), (260, 214)]
    events = run.findall("{*}g[@class='event']")
    assert events[0].attrib["transform"] != events[1].attrib["transform"]


def test_decomposer_row_spans_invocation_and_shows_only_messages():
    trace = _trace()
    trace["messages"] = [
        {"type": "human", "content": "An older request."},
        {"type": "ai", "content": "An unrelated answer."},
    ]
    document = render_trace(trace)
    svg = _svg(document)
    templates = _templates(document)
    invocation = svg.find("{*}g[@class='invocation ']")
    assert _marker_positions(invocation) == [(260, 74), (1168, 126)]
    events = invocation.findall("{*}g[@class='event']")
    assert [event.find("{*}text").text for event in events] == ["🧑", "🐶"]
    assert ["".join(templates[event.attrib["data-tooltip"]].itertext()).strip() for event in events] == [
        "Compare two approaches.", "Here is the comparison.",
    ]


def test_multiple_decomposer_runs_share_one_row_and_keep_their_messages():
    trace = _trace()
    trace["decomposer_agent_runs"].append({
        "agent_run_id": "decomposer-2", "agent_id": "decomposer", "run_id": "decomposer-2",
        "started_at": 140.0, "collected_at": 150.0, "status": "responded",
        "prompt": "Check the answer again.", "response": "The answer is correct.",
    })
    trace["agent_runs"]["root-4"] = {
        "agent_run_id": "root-4", "agent_id": "root", "started_at": 142.0,
        "collected_at": 148.0, "status": "responded", "prompt": "Check again.",
        "response": "Confirmed.",
    }
    original = deepcopy(trace)
    document = render_trace(trace)
    svg = _svg(document)
    templates = _templates(document)

    assert trace == original
    assert len(svg.findall("{*}g[@class='decomposer']")) == 1
    bars = svg.findall("{*}g[@class='invocation ']")
    assert [bar.attrib["data-run-id"] for bar in bars] == ["decomposer-1", "decomposer-2"]
    for bar, coordinates, messages in zip(
        bars,
        [(260.0, 865.33), (1016.67, 1168.0)],
        [("Compare two approaches.", "Here is the comparison."),
         ("Check the answer again.", "The answer is correct.")],
    ):
        positions = _marker_positions(bar)
        assert tuple(x for x, _ in positions) == pytest.approx(coordinates)
        assert [y for _, y in positions] == [74, 126]
        events = bar.findall("{*}g[@class='event']")
        assert [event.find("{*}text").text for event in events] == ["🧑", "🐶"]
        assert tuple(
            "".join(templates[event.attrib["data-tooltip"]].itertext()).strip() for event in events
        ) == messages
    assert [y for _, y in _marker_positions(svg.find("{*}g[@data-run-id='root-4']"))] == [162, 214]
