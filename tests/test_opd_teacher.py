from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from training.opd.samples import Sample, write_samples
from training.opd.teacher import TeacherEndpoint, TeacherError, score_file, token_logprobs


def _response(ids: list[int], *, echo: list[int] | None = None, key_offset: int = 0) -> dict:
    entries = [None] + [{str(token + key_offset): {"logprob": -float(token) / 10, "rank": 1}} for token in ids[1:]]
    return {"choices": [{"prompt_token_ids": echo if echo is not None else ids, "prompt_logprobs": entries}]}


def test_token_logprobs_reads_the_actual_tokens() -> None:
    assert token_logprobs(_response([5, 7, 9]), [5, 7, 9]) == [None, -0.7, -0.9]


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (_response([5, 7], echo=[5, 8]), "different token ids"),
        (_response([5, 7], key_offset=1), "not keyed by the actual token"),
        ({"choices": [{"prompt_logprobs": [None]}]}, "expected 2"),
    ],
)
def test_token_logprobs_rejects_inconsistent_responses(response: dict, message: str) -> None:
    with pytest.raises(TeacherError, match=message):
        token_logprobs(response, [5, 7])


class _FakeTeacher(BaseHTTPRequestHandler):
    calls = 0

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).calls += 1
        assert body["prompt_logprobs"] == 0 and body["return_token_ids"] is True
        payload = json.dumps(_response(body["prompt"])).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:  # noqa: ANN002
        pass


def test_score_file_scores_generated_positions_and_resumes(tmp_path: Path) -> None:
    server = HTTPServer(("127.0.0.1", 0), _FakeTeacher)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = TeacherEndpoint(base_url=f"http://127.0.0.1:{server.server_port}", api_key="k", model="m")
        samples = [
            Sample("e/0", "e", "shop", "t", 1.0, [1, 2, 30, 40, 5, 60], [(2, 4), (5, 6)], [-0.1, -0.2, -0.3]),
            Sample("e/1", "e", "shop", "t", 1.0, [1, 70], [(1, 2)], [-0.4]),
        ]
        write_samples(samples, tmp_path / "samples.jsonl")
        output = tmp_path / "scored.jsonl"
        summary = score_file(endpoint, tmp_path / "samples.jsonl", output, concurrency=2)
        assert summary["failed"] == 0 and summary["newly_scored"] == 2
        scored = {json.loads(line)["sample_id"]: json.loads(line) for line in output.read_text().splitlines()}
        assert scored["e/0"]["teacher_logprobs"] == pytest.approx([-3.0, -4.0, -6.0])
        assert scored["e/1"]["teacher_logprobs"] == pytest.approx([-7.0])

        calls = _FakeTeacher.calls
        again = score_file(endpoint, tmp_path / "samples.jsonl", output, concurrency=2)
        assert again["already_scored"] == 2 and _FakeTeacher.calls == calls
    finally:
        server.shutdown()


def test_missing_credentials_fail_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_SUCH_URL", raising=False)
    with pytest.raises(TeacherError, match="NO_SUCH_URL"):
        TeacherEndpoint.from_env(model="m", base_url_env="NO_SUCH_URL", api_key_env="NO_SUCH_KEY")
