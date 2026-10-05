import logging
from html import escape
from pathlib import Path
from random import Random
from typing import Any

from markdown_it import MarkdownIt


def write_trace_html(trace: dict[str, Any], path: str | Path) -> bool:
    """Write an optional visualization without changing the run's outcome."""
    if not trace.get("decomposer_agent_runs"):
        return False
    try:
        Path(path).write_text(render_trace(trace), encoding="utf-8")
    except Exception as error:
        logging.getLogger(__name__).warning("Could not write %s: %s", path, error)
        return False
    return True


# The index selects the masculine or feminine form of an epithet.
_ANIMALS = {
    "🐱": ("кот", 0), "🦊": ("лиса", 1), "🐻": ("медведь", 0),
    "🐼": ("панда", 1), "🐨": ("коала", 1), "🐯": ("тигр", 0),
    "🦁": ("лев", 0), "🐮": ("корова", 1), "🐷": ("поросёнок", 0),
    "🐸": ("лягушка", 1), "🐵": ("обезьяна", 1), "🐰": ("кролик", 0),
    "🐹": ("хомяк", 0), "🦝": ("енот", 0), "🦄": ("единорог", 0),
    "🐴": ("конь", 0), "🦓": ("зебра", 1), "🦌": ("олень", 0),
    "🦬": ("бизон", 0), "🐏": ("баран", 0), "🐑": ("овца", 1),
    "🐐": ("коза", 1), "🐪": ("дромадер", 0), "🐫": ("верблюд", 0),
    "🦙": ("лама", 1), "🦒": ("жираф", 0), "🐘": ("слон", 0),
    "🦏": ("носорог", 0), "🦛": ("бегемот", 0), "🐭": ("мышь", 1),
    "🦔": ("ёж", 0), "🦇": ("летучая мышь", 1), "🦥": ("ленивец", 0),
    "🦦": ("выдра", 1), "🦫": ("бобр", 0), "🦡": ("барсук", 0),
    "🦘": ("кенгуру", 0), "🦨": ("скунс", 0), "🐔": ("курица", 1),
    "🐧": ("пингвин", 0), "🐦": ("птица", 1), "🐤": ("цыплёнок", 0),
    "🦆": ("утка", 1), "🦅": ("орёл", 0), "🦉": ("сова", 1),
    "🦜": ("попугай", 0), "🦢": ("лебедь", 0), "🦩": ("фламинго", 0),
    "🦚": ("павлин", 0), "🦃": ("индюк", 0), "🐊": ("крокодил", 0),
    "🐢": ("черепаха", 1), "🦎": ("ящерица", 1), "🐍": ("змея", 1),
    "🐲": ("дракон", 0), "🦕": ("диплодок", 0), "🦖": ("тираннозавр", 0),
    "🐳": ("кит", 0), "🐬": ("дельфин", 0), "🦭": ("тюлень", 0),
    "🐟": ("рыба", 1), "🐠": ("рыбка", 1), "🐡": ("иглобрюх", 0),
    "🦈": ("акула", 1), "🐙": ("осьминог", 0), "🦑": ("кальмар", 0),
    "🦀": ("краб", 0), "🦞": ("омар", 0), "🦐": ("креветка", 1),
    "🦋": ("бабочка", 1), "🐝": ("пчела", 1), "🪲": ("жук", 0),
    "🐞": ("божья коровка", 1), "🦗": ("сверчок", 0), "🐜": ("муравей", 0),
    "🐌": ("улитка", 1), "🦂": ("скорпион", 0),
}
_EPITHETS = (
    ("Задумчивый", "Задумчивая"), ("Любопытный", "Любопытная"),
    ("Смелый", "Смелая"), ("Бодрый", "Бодрая"), ("Добрый", "Добрая"),
    ("Ловкий", "Ловкая"), ("Умелый", "Умелая"), ("Весёлый", "Весёлая"),
    ("Сонный", "Сонная"), ("Шустрый", "Шустрая"), ("Мудрый", "Мудрая"),
    ("Резвый", "Резвая"), ("Зоркий", "Зоркая"),
)

_STYLE = """
* { box-sizing: border-box; }
body { margin: 0; padding: 32px; background: #f5f7f8; color: #23343d;
       font: 15px/1.5 system-ui, sans-serif; }
.chart { overflow-x: auto; background: white; border: 1px solid #dce4e7;
         border-radius: 12px; }
svg { display: block; width: 100%; min-width: 1000px; height: auto; }
svg text { font-family: system-ui, sans-serif; fill: #23343d; }
.tick { font-size: 12px; fill: #60717b; }
.grid { stroke: #e4eaed; stroke-dasharray: 3 5; }
.fork, .run-thread { stroke: #d7e0e4; }
.agent-name { font-size: 14px; font-weight: 600; }
.animal { font-size: 23px; }
.fork, .run-thread { fill: none; }
.created { fill: white; stroke: #60717b; stroke-width: 2; }
.run-line { stroke: #338b79; stroke-width: 6; stroke-linecap: round; }
.failed .run-line, .failed .event:hover circle, .failed .event:focus circle {
  stroke: #bc5252; }
.uncollected .run-line { stroke: #a38145; stroke-dasharray: 5 6; }
.event { cursor: pointer; outline: none; }
.event circle { fill: white; stroke: #dce4e7; }
.event:hover circle, .event:focus circle { stroke: #338b79; stroke-width: 2; }
#tooltip { position: fixed; z-index: 1; width: min(560px, calc(100vw - 24px));
           max-height: min(440px, calc(100vh - 24px)); overflow: auto;
           padding: 16px 20px; border: 1px solid #c8d5da; border-radius: 18px 18px 18px 4px;
           background: white; box-shadow: 0 8px 32px #23343d26; }
.message { overflow-wrap: anywhere; line-height: 1.6; }
.message > :first-child { margin-top: 0; }
.message > :last-child { margin-bottom: 0; }
.message p, .message ul, .message ol, .message pre, .message blockquote,
.message table { margin: 0 0 12px; }
.message h1, .message h2, .message h3, .message h4, .message h5, .message h6 {
  margin: 20px 0 10px; line-height: 1.3; }
.message h1 { font-size: 22px; }
.message h2 { font-size: 20px; }
.message h3 { font-size: 17px; }
.message h4, .message h5, .message h6 { font-size: 15px; }
.message ul, .message ol { padding-left: 24px; }
.message li + li { margin-top: 4px; }
.message li > p { margin: 0; }
.message code { padding: 2px 5px; border-radius: 4px; background: #f0f4f6;
                font: 13px/1.5 ui-monospace, monospace; }
.message pre { padding: 12px; border-radius: 8px; background: #f0f4f6;
               overflow-x: auto; }
.message pre code { padding: 0; background: none; white-space: pre; }
.message blockquote { padding-left: 12px; border-left: 3px solid #c8d5da; color: #60717b; }
.message blockquote > :last-child { margin-bottom: 0; }
.message table { border-collapse: collapse; display: block; max-width: 100%; overflow-x: auto; }
.message th, .message td { padding: 6px 10px; border: 1px solid #dce4e7; text-align: left; }
.message th { background: #f0f4f6; }
.message a { color: #247360; text-underline-offset: 2px; }
.message hr { border: 0; border-top: 1px solid #dce4e7; margin: 16px 0; }
.message img { max-width: 100%; }
@media (max-width: 600px) { body { padding: 16px; } }
"""

_SCRIPT = """
const tooltip = document.getElementById("tooltip");
let hideTimer;
function hide() { tooltip.hidden = true; }
function delayHide() { hideTimer = setTimeout(hide, 180); }
function show(event) {
  clearTimeout(hideTimer);
  const marker = event.currentTarget;
  tooltip.replaceChildren(
    document.getElementById(marker.dataset.tooltip).content.cloneNode(true)
  );
  tooltip.hidden = false;
  tooltip.scrollTop = 0;
  const anchor = marker.getBoundingClientRect();
  const left = Math.max(12, Math.min(anchor.left, innerWidth - tooltip.offsetWidth - 12));
  const top = anchor.bottom + 8 + tooltip.offsetHeight <= innerHeight - 12
    ? anchor.bottom + 8 : Math.max(12, anchor.top - tooltip.offsetHeight - 8);
  tooltip.style.left = left + "px";
  tooltip.style.top = top + "px";
}
for (const marker of document.querySelectorAll("[data-tooltip]")) {
  marker.addEventListener("mouseenter", show);
  marker.addEventListener("focus", show);
  marker.addEventListener("click", show);
  marker.addEventListener("mouseleave", delayHide);
  marker.addEventListener("blur", delayHide);
}
tooltip.addEventListener("mouseenter", () => clearTimeout(hideTimer));
tooltip.addEventListener("mouseleave", delayHide);
tooltip.addEventListener("focus", () => clearTimeout(hideTimer));
tooltip.addEventListener("blur", delayHide);
document.addEventListener("keydown", event => { if (event.key === "Escape") hide(); });
document.addEventListener("click", event => {
  if (!event.target.closest("[data-tooltip], #tooltip")) hide();
});
document.querySelector(".chart").addEventListener("scroll", hide);
window.addEventListener("resize", hide);
window.addEventListener("scroll", hide);
"""


def render_trace(trace: dict[str, Any]) -> str:
    """Render Decomposer and delegated runs from raw thread state as standalone HTML.

    Timestamps are Unix seconds recorded in decomposer_agent_runs and agent_runs.
    A Decomposer run ends when its invocation returns; a delegated run ends when
    wait() returns its result. Uncollected runs have an open end at the last event.
    """
    agents = sorted(
        trace.get("agents", {}).values(),
        key=lambda agent: (agent["created_at"], agent["agent_id"]),
    )
    runs = sorted(
        trace.get("agent_runs", {}).values(),
        key=lambda run: (run["started_at"], run["agent_run_id"]),
    )
    decomposer_runs = trace["decomposer_agent_runs"]
    all_runs = [*decomposer_runs, *runs]
    origin = min(run["started_at"] for run in decomposer_runs)
    last_event = max(
        [origin]
        + [agent["created_at"] for agent in agents]
        + [run["started_at"] for run in all_runs]
        + [run["collected_at"] for run in all_runs if "collected_at" in run]
    )
    duration = last_event - origin
    left, right = 260, 1168
    height = 76 + 88 * (len(agents) + 1)
    rows = {agent["agent_id"]: 188 + 88 * i for i, agent in enumerate(agents)}
    animals = list(_ANIMALS)
    random = Random("|".join(rows))
    random.shuffle(animals)
    epithets = {icon: random.sample(_EPITHETS, len(_EPITHETS)) for icon in animals}
    icons = {
        agent_id: animals[i % len(animals)]
        for i, agent_id in enumerate(rows)
    }
    names = {}
    for i, agent_id in enumerate(rows):
        icon = icons[agent_id]
        animal, gender = _ANIMALS[icon]
        epithet = epithets[icon][i // len(animals) % len(_EPITHETS)][gender]
        names[agent_id] = f"{epithet} {animal}"
    markdown = MarkdownIt("commonmark", {"html": False, "breaks": True}).enable(
        ["table", "strikethrough"],
    )
    tooltips: list[str] = []

    def x(timestamp: float) -> float:
        return left + (timestamp - origin) / (duration or 1.0) * (right - left)

    def marker(
        position: float, y: float, icon: str, label: str, content: str,
    ) -> str:
        tooltip_id = f"tip-{len(tooltips)}"
        tooltips.append(
            f'<template id="{tooltip_id}">'
            f'<div class="message">{markdown.render(content.strip())}</div></template>'
        )
        return (
            f'<g class="event" data-tooltip="{tooltip_id}" tabindex="0" role="img" '
            f'aria-label="{escape(label)}" aria-describedby="tooltip" '
            f'transform="translate({position:.2f} {y})">'
            '<circle r="16"/><text class="animal" text-anchor="middle" '
            f'dominant-baseline="central">{icon}</text></g>'
        )

    def open_end(position: float, y: float) -> str:
        return (
            f'<circle class="created" cx="{position:.2f}" cy="{y}" r="4">'
            '<title>Ответ ещё не получен</title></circle>'
            f'<text x="{position + 9:.2f}" y="{y + 5}">…</text>'
        )

    def run_bar(run: dict[str, Any], *, is_decomposer: bool) -> str:
        agent_id = run["agent_id"]
        y = 100 if is_decomposer else rows[agent_id]
        start = x(run["started_at"])
        collected = "collected_at" in run
        end = x(run["collected_at"] if collected else last_event)
        bend = min(12, (end - start) / 4)
        path = (
            f'M {start:.2f} {y - 26} V {y - bend:.2f} '
            f'Q {start:.2f} {y} {start + bend:.2f} {y} '
        )
        if collected:
            path += (
                f'H {end - bend:.2f} Q {end:.2f} {y} {end:.2f} {y + bend:.2f} '
                f'V {y + 26}'
            )
        else:
            path += f'H {end:.2f}'
        status = run["status"]
        style = "uncollected" if not collected else "failed" if status != "responded" else ""
        kind = "invocation" if is_decomposer else "run"
        bar = [
            f'<g class="{kind} {style}" data-run-id="{escape(run["agent_run_id"])}">'
            f'<path class="run-thread" d="{path}"/>'
            f'<line class="run-line" x1="{start + bend:.2f}" x2="{end - bend:.2f}" '
            f'y1="{y}" y2="{y}"/>',
            marker(
                start, y - 26, "🧑" if is_decomposer else "🐶",
                "Запрос пользователя" if is_decomposer else "Сообщение Decomposer", run["prompt"],
            ),
        ]
        if collected:
            response = run.get("response")
            if response is None:
                response = run.get("error") or ""
            bar.append(marker(
                end, y + 26, "🐶" if is_decomposer else icons[agent_id],
                "Ответ Decomposer" if is_decomposer else f"Ответ агента «{names[agent_id]}»", response,
            ))
        else:
            bar.append(open_end(end, y))
        bar.append('</g>')
        return "".join(bar)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 {height}" '
        'aria-label="Временная шкала агентов Decomposer">',
        '<g class="decomposer"><text class="animal" x="24" y="107">🐶</text>'
        '<text class="agent-name" x="60" y="105">Decomposer</text></g>',
    ]
    for i, agent in enumerate(agents):
        agent_id = agent["agent_id"]
        y = rows[agent_id]
        name = names[agent_id]
        parts.append(
            f'<g class="agent" data-agent-id="{escape(agent_id)}">'
            f'<rect x="0" y="{y - 44}" width="1200" height="88" '
            f'fill="{"#f8fafb" if i % 2 == 0 else "#ffffff"}"/>'
            f'<text class="animal" x="24" y="{y + 7}">{icons[agent_id]}</text>'
            f'<text class="agent-name" x="60" y="{y + 5}">{escape(name)}</text></g>'
        )
    for i in range(6 if duration else 1):
        seconds = duration * i / 5
        position = x(origin + seconds)
        parts.append(
            f'<line class="grid" x1="{position:.2f}" x2="{position:.2f}" '
            f'y1="52" y2="{height - 20}"/>'
            f'<text class="tick" x="{position:.2f}" y="30" '
            f'text-anchor="middle">{seconds:.3g} с</text>'
        )

    parts.extend(run_bar(run, is_decomposer=True) for run in decomposer_runs)

    for agent in agents:
        if "forked_from" in agent:
            parent = agent["forked_from"]
            parent_run = next((
                run for run in reversed(runs)
                if run["agent_id"] == parent and "collected_at" in run
                and run["collected_at"] <= agent["created_at"]
            ), None)
            if parent_run is None:
                start = x(trace["agents"][parent]["created_at"])
            else:
                start = x(parent_run["collected_at"])
                start -= min(12, (start - x(parent_run["started_at"])) / 4)
            end = x(agent["created_at"])
            bend = min(12, end - start)
            parent_y = rows[parent]
            y = rows[agent["agent_id"]]
            parts.append(
                f'<path class="fork" d="M {start:.2f} {parent_y} H {end - bend:.2f} '
                f'Q {end:.2f} {parent_y} {end:.2f} {parent_y + bend:.2f} V {y}">'
                f'<title>Форк от агента «{escape(names[parent])}»</title></path>'
            )

    parts.extend(run_bar(run, is_decomposer=False) for run in runs)

    for agent in agents:
        parts.append(
            f'<circle class="created" cx="{x(agent["created_at"]):.2f}" '
            f'cy="{rows[agent["agent_id"]]}" r="4">'
            f'<title>Создание: {agent["created_at"] - origin:.6g} с</title></circle>'
        )
    parts.append('</svg>')
    chart = "\n".join(parts)
    return f"""<!doctype html>
<html lang="ru">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Decomposer: трейс</title><style>{_STYLE}</style></head>
<body>
<div class="chart">{chart}</div>
<div id="tooltip" role="tooltip" tabindex="0" hidden></div>
{"".join(tooltips)}
<script>{_SCRIPT}</script>
</body></html>
"""
