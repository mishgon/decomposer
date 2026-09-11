"""One isolated Gym environment; no model client or training dependencies."""

import json
import re
import shlex
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

from .adaptive_scheduler import extract_partial_score


def native_reward(evaluation):
    stdout, stderr = evaluation["stdout"], evaluation["stderr"]
    if "Traceback (most recent call last)" in stdout + stderr:
        raise RuntimeError("Native evaluator crashed")
    if any(marker in (stdout + stderr).lower() for marker in (
        "connection refused", "could not connect to server", "db fetch failed", "[warn] expected fetch:"
    )):
        raise RuntimeError("Native evaluator could not access its environment")
    if "groundtruth not found" in stdout.lower() or "groundtruth" in stdout.lower() and "not found" in stdout.lower():
        raise RuntimeError("Native evaluator is missing ground truth")
    # Audited binary formats: missing output is a legitimate failure, unlike a
    # missing groundtruth file or a Python exception (both can also exit with 1).
    if evaluation["returncode"] == 1 and (
        stdout.startswith("FAIL: Agent output not found:")
        or re.search(r"=== RESULT: FAIL \(\d+ errors\) ===", stdout)
    ):
        return 0.0
    partial = extract_partial_score(evaluation)
    if partial is not None:
        if partial.fraction == 1.0 and evaluation["returncode"] != 0:
            raise RuntimeError("Native check counts contradict failure exit status")
        return partial.fraction
    if evaluation["returncode"] == 0:
        return 1.0
    if evaluation["returncode"] == 1 and re.search(
        r"=== SUMMARY ===\s+File errors: \d+\s+DB errors:\s+\d+", stdout
    ) and "Overall: FAIL" in stdout:
        return 0.0
    raise RuntimeError("Unscorable native evaluator output")


class Episode:
    def __init__(self, task: str, directory: Path, *, subagent_port: int = 8025,
                 subagent_model: str = "Qwen/Qwen3.5-4B",
                 subagent_url: str | None = None, subagent_host: str | None = None,
                 image: str = "decomposer-toolathlon-rl:latest", engine: str = "podman",
                 startup_timeout: float = 240):
        self.root = Path(__file__).resolve().parents[2]
        tasks = self.root / "external/toolathlon_gym/tasks/finalpool"
        if (tasks / task).resolve().parent != tasks.resolve() or not (tasks / task).is_dir():
            raise ValueError(f"Unknown task: {task}")
        self.task, self.directory = task, directory.resolve()
        self.subagent_port, self.subagent_model = subagent_port, subagent_model
        self.subagent_url, self.subagent_host = subagent_url, subagent_host
        self.image, self.engine, self.startup_timeout = image, engine, startup_timeout
        self.network = "decomposer-rl-" + uuid.uuid4().hex[:16]
        self.pg, self.container = self.network + "-pg", self.network + "-task"

    def command(self, *args, check=True, timeout=120):
        result = subprocess.run([self.engine, *map(str, args)], capture_output=True,
                                text=True, timeout=timeout)
        if check and result.returncode:
            raise RuntimeError(f"{args[:2]}: {result.stderr or result.stdout}")
        return result

    def start(self):
        self.directory.mkdir(parents=True, exist_ok=False)
        data = self.directory / "data"
        data.mkdir()
        try:
            self.command("network", "create", self.network)
            self.command("run", "--http-proxy=false", "-d", "--name", self.pg,
                         "--network", self.network, "--network-alias", "postgres",
                         "-e", "POSTGRES_DB=toolathlon_gym", "-e", "POSTGRES_USER=eigent",
                         "-e", "POSTGRES_PASSWORD=camel", "-v",
                         f"{self.root}/external/toolathlon_gym/db/init.sql.gz:/docker-entrypoint-initdb.d/init.sql.gz:ro",
                         "docker.io/library/postgres:15")
            deadline = time.monotonic() + self.startup_timeout
            while time.monotonic() < deadline:
                process = self.command("exec", self.pg, "cat", "/proc/1/comm", check=False)
                ready = self.command("exec", self.pg, "psql", "-U", "eigent", "-d",
                                     "toolathlon_gym", "-tAc",
                                     "SELECT to_regclass('email.messages') IS NOT NULL;", check=False)
                if process.stdout.strip() == "postgres" and ready.stdout.strip() == "t":
                    break
                time.sleep(1)
            else:
                raise TimeoutError("PostgreSQL initialization timed out")
            self.command("exec", self.pg, "psql", "-U", "eigent", "-d", "toolathlon_gym", "-c",
                         "ALTER TABLE email.sent_log DROP CONSTRAINT IF EXISTS sent_log_message_id_fkey; "
                         "ALTER TABLE email.sent_log ADD CONSTRAINT sent_log_message_id_fkey "
                         "FOREIGN KEY (message_id) REFERENCES email.messages(id) ON DELETE CASCADE;")
            self.command("exec", self.pg, "psql", "-U", "eigent", "-d", "toolathlon_gym", "-c",
                         "CREATE OR REPLACE VIEW gform.question_options AS "
                         "SELECT q.id AS question_id, option->>'value' AS value "
                         "FROM gform.questions q CROSS JOIN LATERAL "
                         "jsonb_array_elements(COALESCE(q.config->'options', '[]'::jsonb)) option;")
            address = self.command("inspect", "--format",
                                   "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", self.pg).stdout.strip()
            if not address:
                raise RuntimeError("PostgreSQL has no network address")
            env = {"PGHOST": address, "PG_HOST": address, "PGPORT": "5432",
                   "PGUSER": "eigent", "PGPASSWORD": "camel", "PGDATABASE": "toolathlon_gym",
                   "TOOLATHLON_TASK": self.task, "N_JOBS_PER_WORKER": "1000",
                   "TOOLATHLON_SUBAGENT_CALL_LOG": "/artifacts/data/subagent_model_calls.jsonl",
                   "DECOMPOSER_SUBAGENT_MODEL": self.subagent_model,
                   "DECOMPOSER_SUBAGENT_BASE_URL": self.subagent_url or f"http://host.docker.internal:{self.subagent_port}/v1",
                   "PYTHONPATH": "/rl-source/src"}
            # Pass credentials by environment name, never in command arguments or artifacts.
            credentials = ("-e", "VLLM_API_KEY") if self.subagent_url else ()
            self.start_task_container("run", "--http-proxy=false", "-d", "--name", self.container,
                         "--network", self.network, "--add-host", "host.docker.internal:host-gateway",
                         *(("--add-host", self.subagent_host) if self.subagent_host else ()),
                         *credentials,
                         "-p", "127.0.0.1::2024",
                         *[arg for key, value in env.items() for arg in ("-e", f"{key}={value}")],
                         "-v", f"{data}:/artifacts/data",
                         "-v", f"{self.root}/src:/rl-source/src:ro",
                         "-v", f"{self.root}/gyms/toolathlon_gym/subagents/graph.py:/opt/decomposer/gyms/toolathlon_gym/subagents/graph.py:ro",
                         self.image)
            port = self.command("port", self.container, "2024/tcp").stdout.strip().rsplit(":", 1)[1]
            self.url = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + self.startup_timeout
            while time.monotonic() < deadline:
                try:
                    with urllib.request.urlopen(self.url + "/ok", timeout=2) as response:
                        if response.status == 200:
                            self.runtime = json.loads((data / "runtime.json").read_text())
                            return self
                except OSError:
                    pass
                status = self.command("inspect", "--format", "{{.State.Status}}", self.container).stdout.strip()
                if status in {"dead", "exited"}:
                    raise RuntimeError(self.command("logs", self.container, check=False).stderr)
                time.sleep(1)
            raise TimeoutError("Subagent service startup timed out")
        except BaseException:
            self.close()
            raise

    def start_task_container(self, *args):
        for attempt in range(5):
            result = self.command(*args, check=False)
            if result.returncode == 0:
                return result
            if "address already in use" not in result.stderr or attempt == 4:
                raise RuntimeError(result.stderr or result.stdout)
            self.command("rm", "-f", self.container, check=False)
        raise RuntimeError("Could not allocate task port")

    def score(self, timeout=180, *, require_partial=True):
        config = self.runtime["task_config"]
        command = config["evaluation"]["evaluation_command"]
        if not command:
            raise RuntimeError("Task has no native evaluator")
        native_path = "/tmp/decomposer-rl-evaluation.json"
        args = [*shlex.split(command), "--agent_workspace", config["agent_workspace"]]
        for flag, value in (("--groundtruth_workspace", config["evaluation"]["groundtruth_workspace"]),
                            ("--launch_time", config["launch_time"])):
            if value is not None:
                args += [flag, value]
        args += ["--res_log_file", native_path]
        result = self.command("exec", "--workdir", "/workspace", self.container,
                              *args, check=False, timeout=timeout)
        raw = self.command("exec", self.container, "cat", native_path, check=False)
        try:
            native = json.loads(raw.stdout)
        except ValueError:
            native = None
        evaluation = {"pass": result.returncode == 0, "returncode": result.returncode,
                      "native_result": native, "stdout": result.stdout, "stderr": result.stderr}
        (self.directory / "evaluation.json").write_text(json.dumps(evaluation, indent=2))
        try:
            evaluation["reward"] = native_reward(evaluation)
        except RuntimeError as error:
            if require_partial or str(error) != "Unscorable native evaluator output":
                raise
            evaluation["reward"] = None
        (self.directory / "evaluation.json").write_text(json.dumps(evaluation, indent=2))
        return evaluation

    def close(self):
        errors = []
        volumes = set()
        for container in (self.container, self.pg):
            try:
                inspection = self.command("inspect", container, check=False)
                if inspection.returncode == 0:
                    for mount in json.loads(inspection.stdout)[0].get("Mounts", []):
                        if mount.get("Type") == "volume":
                            volumes.add(mount["Name"])
                elif self.command("container", "exists", container, check=False).returncode != 1:
                    errors.append(f"Could not inspect volumes: {container}")
            except Exception as error:
                errors.append(repr(error))
            try:
                logs = self.command("logs", container, check=False, timeout=15)
                (self.directory / f"{container}.log").write_text(logs.stdout + logs.stderr)
            except Exception as error:
                errors.append(repr(error))
            try:
                for attempt in range(3):
                    result = self.command("rm", "-f", "-v", container, check=False, timeout=30)
                    exists = self.command("container", "exists", container, check=False)
                    if exists.returncode == 1:
                        break
                    time.sleep(attempt + 1)
                else:
                    errors.append(f"Container cleanup not verified: {container}: {result.stderr}")
            except Exception as error:
                errors.append(repr(error))
        # Only volumes belonging to these exact episode containers; never global prune.
        for volume in volumes:
            try:
                result = self.command("volume", "rm", volume, check=False, timeout=30)
                if self.command("volume", "exists", volume, check=False).returncode != 1:
                    errors.append(f"Volume cleanup not verified: {volume}: {result.stderr}")
            except Exception as error:
                errors.append(repr(error))
        try:
            result = self.command("network", "rm", self.network, check=False, timeout=30)
            if result.returncode:
                errors.append(result.stderr)
        except Exception as error:
            errors.append(repr(error))
        (self.directory / "cleanup.json").write_text(json.dumps(
            {"errors": errors, "volumes": sorted(volumes), "finished_at": time.time()}, indent=2))
