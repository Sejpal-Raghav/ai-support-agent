"""Exercise the real Linux image, offline models, private bundle, HTTP API, and speech path."""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

SMOKE_CODE = r"""
import base64, json, re, urllib.error, urllib.request
from pathlib import Path
spec = json.loads(__import__('sys').stdin.read())
base = 'http://127.0.0.1:' + __import__('os').environ.get('PORT', '7860')
def request(path, data=None, headers=None):
    req = urllib.request.Request(base + path, data=data, headers=headers or {})
    with urllib.request.urlopen(req, timeout=180) as response:
        return response.status, response.read()
status, raw = request('/api/ready')
health = json.loads(raw)
assert status == 200
assert all(health[key] for key in ('asr_ready','classifier_ready','retrieval_ready'))
_, html = request('/')
asset = re.search(rb'src="(/assets/[^" ]+\.js)"', html).group(1).decode()
assert request(asset)[0] == 200
results = []
for text in ('Where is my delayed package?', 'I want to return an item',
             'My item arrived damaged', 'Please cancel my order',
             'Is this product in stock?', 'My password is secret', 'I was charged twice'):
    _, raw = request('/api/respond', json.dumps({'text':text}).encode(),
                     {'Content-Type':'application/json'})
    result = json.loads(raw)
    assert result['intent'] and result['response']
    if result['action'] == 'respond':
        assert result['evidence_case_ids']
    if 'password' in text or 'charged twice' in text:
        assert result['action'] == 'fallback'
    results.append({'text':text, 'intent':result['intent'], 'action':result['action']})
voice = None
if spec.get('audio'):
    boundary = 'support-agent-local-validation'
    audio = base64.b64decode(spec['audio'])
    audio_type = spec.get('audio_type', 'audio/wav')
    filename = spec.get('audio_filename', 'fixture.wav')
    body = (('--'+boundary+'\r\nContent-Disposition: form-data; name="audio"; '
             'filename="'+filename+'"\r\nContent-Type: '+audio_type+'\r\n\r\n').encode()
            + audio + ('\r\n--'+boundary+'--\r\n').encode())
    _, raw = request('/api/transcribe', body,
                     {'Content-Type':'multipart/form-data; boundary='+boundary})
    voice = json.loads(raw)
    assert voice['transcript'].strip()
    if spec.get('expected_transcript'):
        assert voice['transcript'] == spec['expected_transcript'], voice
    assert request('/api/respond',json.dumps({'text':voice['transcript']}).encode(),
                   {'Content-Type':'application/json'})[0] == 200
peak = None
for name in ('/sys/fs/cgroup/memory.peak','/sys/fs/cgroup/memory/memory.max_usage_in_bytes'):
    if Path(name).is_file():
        peak = int(Path(name).read_text().strip())
        break
print(json.dumps({'health':health,'static_asset_status':200,'requests':results,
                  'voice':voice,'peak_memory_bytes':peak}))
"""


def docker(*arguments: str, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *arguments], input=input_text, text=True, capture_output=True, check=False
    )


def verify(args: argparse.Namespace) -> dict[str, object]:
    bundle = args.bundle.resolve()
    if not bundle.is_file():
        raise ValueError(
            "Export the inference bundle with scripts/package_render_evidence.py first."
        )
    name = f"support-agent-check-{uuid4().hex[:10]}"
    result: dict[str, object] = {
        "image": args.image,
        "memory_limit": args.memory,
        "swap": "disabled",
        "runtime_network": "none",
        "passed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    created = False
    try:
        run = docker(
            "run",
            "--detach",
            "--name",
            name,
            "--network",
            "none",
            "--memory",
            args.memory,
            "--memory-swap",
            args.memory,
            "--cpus",
            args.cpus,
            "--env",
            "PORT=10000",
            "--mount",
            f"type=bind,source={bundle},target=/etc/secrets/support_agent_bundle.b64,readonly",
            args.image,
        )
        if run.returncode:
            raise RuntimeError(run.stderr.strip())
        created = True
        result["container_id"] = run.stdout.strip()
        result["image_id"] = docker(
            "image", "inspect", "--format", "{{.Id}}", args.image
        ).stdout.strip()
        while time.perf_counter() - started < args.timeout:
            inspection = docker("inspect", "--format", "{{json .State}}", name)
            state = json.loads(inspection.stdout)
            if not state["Running"]:
                result["container_state"] = state
                raise RuntimeError(
                    f"Container exited: code={state['ExitCode']}, OOMKilled={state['OOMKilled']}"
                )
            probe = docker(
                "exec",
                name,
                ".venv/bin/python",
                "-c",
                "import urllib.request; urllib.request.urlopen('http://127.0.0.1:10000/api/ready',timeout=2)",
            )
            if probe.returncode == 0:
                break
            time.sleep(2)
        else:
            raise RuntimeError("Container did not become ready before the validation timeout.")
        result["ready_after_seconds"] = round(time.perf_counter() - started, 2)
        payload: dict[str, object] = {}
        if args.audio_fixture is not None:
            payload["audio"] = base64.b64encode(args.audio_fixture.read_bytes()).decode("ascii")
            payload["expected_transcript"] = args.expected_transcript
            payload["audio_filename"] = args.audio_fixture.name
            payload["audio_type"] = {
                ".webm": "audio/webm", ".ogg": "audio/ogg",
            }.get(args.audio_fixture.suffix.lower(), "audio/wav")
        smoke = docker(
            "exec", "-i", name, ".venv/bin/python", "-c", SMOKE_CODE, input_text=json.dumps(payload)
        )
        if smoke.returncode:
            raise RuntimeError(smoke.stderr.strip())
        result["checks"] = json.loads(smoke.stdout)
        for _ in range(25):
            health_state = json.loads(docker("inspect", "--format", "{{json .State}}", name).stdout)
            if health_state.get("Health", {}).get("Status") == "healthy":
                break
            time.sleep(1)
        else:
            raise RuntimeError(
                "HTTP checks passed but Docker's health check did not become healthy."
            )
        result["passed"] = True
    except (ValueError, RuntimeError) as error:
        result["error"] = str(error)
    finally:
        if created:
            logs = docker("logs", name)
            args.output.with_suffix(".log").write_text(logs.stdout + logs.stderr, encoding="utf-8")
            final = docker("inspect", "--format", "{{json .State}}", name)
            if final.returncode == 0:
                result["final_container_state"] = json.loads(final.stdout)
            docker("rm", "--force", name)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="support-agent:render-free")
    parser.add_argument(
        "--bundle", type=Path, default=Path("artifacts/render/support_agent_bundle.b64")
    )
    parser.add_argument("--memory", default="512m")
    parser.add_argument("--cpus", default="0.1")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--audio-fixture", type=Path)
    parser.add_argument("--expected-transcript")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/render-container-check.json")
    )
    result = verify(parser.parse_args())
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
