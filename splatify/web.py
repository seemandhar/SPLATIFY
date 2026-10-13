#!/usr/bin/env python3
"""
Splatify — Web UI

Flask app with real-time SSE streaming of the multi-agent pipeline.
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import re
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, render_template, request, jsonify, Response, abort, send_from_directory
from werkzeug.utils import secure_filename

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    ResultMessage,
    SystemMessage,
    AssistantMessage,
    UserMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ToolResultBlock,
    TaskStartedMessage,
    TaskProgressMessage,
    TaskNotificationMessage,
)

from config import PipelineConfig, get_claude_auth_env
from agents.definitions import build_agent_definitions
from main import build_orchestrator_prompt
from invent import run_invent_pipeline, run_scene_invent_pipeline
from discover import run_discover_pipeline


# ═══════════════════════════════════════════════════════════════════
# APP SETUP
# ═══════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "splatify-claude-agents-dev")

JOBS: dict[str, dict[str, Any]] = {}
JOB_QUEUES: dict[str, queue.Queue] = {}
JOB_STOP_EVENTS: dict[str, threading.Event] = {}

DEFAULT_CONFIG = PipelineConfig()
DEFAULT_CONFIG.ensure_dirs()

# Known agent names for matching
KNOWN_AGENTS = {"parser", "planner", "coder", "reviewer", "validator", "integrator", "tester", "debugger"}

# Keywords to fuzzy-match agent descriptions to known agent names
_AGENT_KEYWORDS: dict[str, list[str]] = {
    "parser":     ["parser", "parse", "extract", "pdf", "paper", "mineru", "markdown"],
    "planner":    ["planner", "plan", "architecture", "dag", "design"],
    "coder":      ["coder", "code", "generator", "generat", "implement", "write code"],
    "reviewer":   ["reviewer", "review", "check code"],
    "validator":  ["validator", "validat", "novelty", "gap", "equation"],
    "integrator": ["integrator", "integrat", "consistency", "cross-file", "wiring"],
    "tester":     ["tester", "test", "smoke", "ns-train", "pip install"],
    "debugger":   ["debugger", "debug", "diagnos", "fix error"],
}


def _match_agent_from_description(text: str) -> str:
    """Fuzzy-match a text description to one of the KNOWN_AGENTS."""
    if not text:
        return "subagent"
    text_lower = text.lower()
    # Direct keyword match — first word wins
    best, best_score = "subagent", 0
    for agent_name, keywords in _AGENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > best_score:
            best, best_score = agent_name, score
    return best


# ═══════════════════════════════════════════════════════════════════
# EVENT HELPERS
# ═══════════════════════════════════════════════════════════════════

def push_event(job_id: str, event_type: str, data: dict):
    """Push a structured event to the job's SSE queue."""
    q = JOB_QUEUES.get(job_id)
    if q:
        q.put({"type": event_type, **data})


def _parse_eval_results(text: str) -> dict | None:
    """Try to parse eval.py results table from tool output text."""
    if "PSNR" not in text or "SSIM" not in text:
        return None
    # Look for the results table rows: "  scene   psnr   ssim   lpips   fps   #gaussians"
    scenes = {}
    averages = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Match rows like: bicycle     25.12    0.7654    0.2345     12.3      1,234,567
        m = re.match(
            r'^(\w+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d,]+|N/A)\s*$',
            line,
        )
        if m:
            name = m.group(1)
            row = {
                "psnr": float(m.group(2)),
                "ssim": float(m.group(3)),
                "lpips": float(m.group(4)),
                "fps": float(m.group(5)),
                "num_gaussians": m.group(6),
            }
            if name == "AVERAGE":
                averages = row
            else:
                scenes[name] = row
    if not scenes:
        return None
    return {"scenes": scenes, "averages": averages}


_PATH_STRIP_PREFIXES = [
    str(Path(__file__).resolve().parent) + "/",
    str(Path.home()) + "/",
]


def _shorten_path(val: str) -> str:
    """Strip long path prefixes to make UI output concise."""
    for prefix in _PATH_STRIP_PREFIXES:
        val = val.replace(prefix, "~/")
    val = val.replace("~/~/", "~/")
    return val


def _tool_input_summary(tool_input: dict) -> str:
    if not isinstance(tool_input, dict):
        return _shorten_path(str(tool_input)[:200])
    for key in ("file_path", "path", "command", "pattern", "prompt", "description", "query"):
        if key in tool_input:
            val = _shorten_path(str(tool_input[key]))
            return val[:120] + ("..." if len(val) > 120 else "")
    return _shorten_path(str(tool_input)[:200])


# ═══════════════════════════════════════════════════════════════════
# PIPELINE RUNNER (background thread)
# ═══════════════════════════════════════════════════════════════════

def _run_pipeline_bg(job_id: str, config: PipelineConfig, **kwargs):
    """Run the pipeline in a background thread, pushing events to SSE queue."""
    job = JOBS[job_id]
    workspace = Path(job["workspace"])
    output_dir = Path(job["output_dir"])

    agent_models = kwargs.get("agent_models")
    model = kwargs.get("model")
    agents = build_agent_definitions(config, agent_models=agent_models)

    prompt = build_orchestrator_prompt(
        arxiv=kwargs.get("arxiv"),
        pdf_path=kwargs.get("pdf_path"),
        method_name=kwargs.get("method_name"),
        data_path=kwargs.get("data_path") or config.default_dataset,
        config=config,
        workspace=workspace,
        output_dir=output_dir,
    )

    (workspace / "orchestrator_prompt.txt").write_text(prompt)

    push_event(job_id, "stage", {"stage": "starting", "message": "Orchestrator agent starting..."})

    if config.viewer_port and config.enable_training:
        push_event(job_id, "viewer_started", {
            "port": config.viewer_port,
            "url": f"https://viewer.nerfstudio.com/versions/24-05/?websocket_url=ws://localhost:{config.viewer_port}",
        })

    start_time = time.time()

    stop_event = JOB_STOP_EVENTS.get(job_id)

    async def _run():
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                model=model,
                cwd=str(config.base_dir),
                allowed_tools=["Read", "Write", "Bash", "Glob", "Grep", "Agent", "WebSearch", "WebFetch"],
                agents=agents,
                permission_mode="acceptEdits",
                max_turns=200,
                env=get_claude_auth_env(),
            ),
        ):
            # Check if user requested stop
            if stop_event and stop_event.is_set():
                job["status"] = "stopped"
                push_event(job_id, "error", {"message": "Pipeline stopped by user."})
                break

            elapsed = time.time() - start_time

            if isinstance(message, ResultMessage):
                job["status"] = "completed"
                job["result"] = message.result
                # Extract cost from ResultMessage (total_cost_usd is session-cumulative)
                cost = getattr(message, "total_cost_usd", None)
                if cost is not None:
                    job["total_cost_usd"] = cost
                # ResultMessage.usage contains only the final turn's tokens, NOT
                # the session cumulative total. Use it only if we have no better data
                # from TaskProgressMessage/TaskNotificationMessage tracking.
                result_usage = getattr(message, "usage", None)
                if isinstance(result_usage, dict):
                    input_tok = result_usage.get("input_tokens", 0) or 0
                    output_tok = result_usage.get("output_tokens", 0) or 0
                    result_total = input_tok + output_tok
                    # Only use ResultMessage tokens if they exceed what we accumulated
                    # (i.e., we had no progress messages at all)
                    if result_total > job.get("total_tokens", 0):
                        job["total_tokens"] = result_total
                        job["total_input_tokens"] = input_tok
                        job["total_output_tokens"] = output_tok
                push_event(job_id, "result", {
                    "message": message.result or "Pipeline complete.",
                    "elapsed": round(elapsed, 1),
                })

            elif isinstance(message, SystemMessage):
                sid = getattr(message, "session_id", None)
                if sid:
                    job["session_id"] = sid
                    push_event(job_id, "system", {"message": f"Session: {sid}"})

            elif isinstance(message, AssistantMessage):
                content = getattr(message, "content", [])
                for block in content:
                    if isinstance(block, ThinkingBlock):
                        text = getattr(block, "thinking", "")
                        if text:
                            push_event(job_id, "thinking", {
                                "message": text[:300] + ("..." if len(text) > 300 else ""),
                            })

                    elif isinstance(block, TextBlock):
                        text = getattr(block, "text", "")
                        if text:
                            push_event(job_id, "text", {"message": text})

                    elif isinstance(block, ToolResultBlock):
                        raw = getattr(block, "content", "")
                        if isinstance(raw, list):
                            parts = []
                            for item in raw:
                                if hasattr(item, "text"):
                                    parts.append(item.text)
                                elif isinstance(item, str):
                                    parts.append(item)
                            raw = "\n".join(parts)
                        if isinstance(raw, str) and raw:
                            # Send up to 8KB so terminal panel has real output
                            truncated = raw[:8000] + ("\n... (truncated)" if len(raw) > 8000 else "")
                            push_event(job_id, "tool_result", {
                                "content": truncated,
                                "message": truncated[:200],
                            })
                            # Check for eval results
                            eval_data = _parse_eval_results(raw)
                            if eval_data:
                                push_event(job_id, "eval_results", eval_data)

                    elif isinstance(block, ToolUseBlock):
                        name = getattr(block, "name", "?")
                        tool_input = getattr(block, "input", {})
                        summary = _tool_input_summary(tool_input)

                        if name == "Agent":
                            agent_type = tool_input.get("subagent_type") or tool_input.get("agent_type") or tool_input.get("type") or "subagent"
                            # Fuzzy-match description to known agent names
                            if agent_type == "subagent" or agent_type not in KNOWN_AGENTS:
                                agent_type = _match_agent_from_description(tool_input.get("description", "") + " " + tool_input.get("prompt", ""))
                            desc = tool_input.get("description", "")
                            print(f"[DEBUG] Agent tool_input keys={list(tool_input.keys())} subagent_type={tool_input.get('subagent_type')} → resolved={agent_type}")
                            push_event(job_id, "agent", {
                                "agent": agent_type,
                                "description": desc,
                                "message": f"Invoking {agent_type} agent",
                            })
                        else:
                            # Send full tool input for the detail panel
                            detail = {}
                            if name == "Bash":
                                detail["command"] = _shorten_path(tool_input.get("command", ""))
                            elif name in ("Read", "Write", "Edit"):
                                detail["file_path"] = _shorten_path(tool_input.get("file_path", ""))
                                if name == "Edit":
                                    detail["old_string"] = str(tool_input.get("old_string", ""))[:500]
                                    detail["new_string"] = str(tool_input.get("new_string", ""))[:500]
                                elif name == "Write":
                                    content_str = str(tool_input.get("content", ""))
                                    detail["content_preview"] = content_str[:1000] + ("..." if len(content_str) > 1000 else "")
                            elif name in ("Grep", "Glob"):
                                detail["pattern"] = tool_input.get("pattern", "")
                                detail["path"] = _shorten_path(tool_input.get("path", ""))

                            push_event(job_id, "tool", {
                                "tool": name,
                                "summary": summary,
                                "detail": detail,
                                "message": f"{name}: {summary}",
                            })

            elif isinstance(message, UserMessage):
                # Tool results come in UserMessage
                content = getattr(message, "content", None)
                if isinstance(content, str) and content:
                    truncated = content[:8000] + ("\n... (truncated)" if len(content) > 8000 else "")
                    push_event(job_id, "tool_result", {
                        "content": truncated,
                        "message": truncated[:200],
                    })
                    eval_data = _parse_eval_results(content)
                    if eval_data:
                        push_event(job_id, "eval_results", eval_data)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, ToolResultBlock):
                            raw = getattr(block, "content", "")
                            if isinstance(raw, list):
                                parts = []
                                for item in raw:
                                    if isinstance(item, dict) and "text" in item:
                                        parts.append(item["text"])
                                    elif hasattr(item, "text"):
                                        parts.append(item.text)
                                    elif isinstance(item, str):
                                        parts.append(item)
                                raw = "\n".join(parts)
                            if isinstance(raw, str) and raw:
                                truncated = raw[:8000] + ("\n... (truncated)" if len(raw) > 8000 else "")
                                push_event(job_id, "tool_result", {
                                    "content": truncated,
                                    "message": truncated[:200],
                                })
                                eval_data = _parse_eval_results(raw)
                                if eval_data:
                                    push_event(job_id, "eval_results", eval_data)

            elif isinstance(message, TaskStartedMessage):
                # TaskStartedMessage has task_type and description, not agent_type
                agent_type = getattr(message, "task_type", None) or "subagent"
                desc = getattr(message, "description", "")
                # Try to match known agent name from description
                if agent_type == "subagent" or agent_type not in KNOWN_AGENTS:
                    agent_type = _match_agent_from_description(desc)
                push_event(job_id, "task_started", {
                    "agent": agent_type,
                    "message": f"Subagent {agent_type} started",
                })

            elif isinstance(message, TaskProgressMessage):
                task_id = getattr(message, "task_id", None)
                usage = getattr(message, "usage", None)
                total_tok = 0
                tool_uses = 0
                dur_ms = 0
                if usage:
                    # TaskUsage is a TypedDict with total_tokens, tool_uses, duration_ms
                    if isinstance(usage, dict):
                        total_tok = usage.get("total_tokens", 0) or 0
                        tool_uses = usage.get("tool_uses", 0) or 0
                        dur_ms = usage.get("duration_ms", 0) or 0
                    else:
                        total_tok = getattr(usage, "total_tokens", 0) or 0
                        tool_uses = getattr(usage, "tool_uses", 0) or 0
                        dur_ms = getattr(usage, "duration_ms", 0) or 0
                # total_tokens from SDK is cumulative per task, so track the
                # latest value per task_id and sum across all tasks.
                task_tokens = job.setdefault("task_tokens", {})
                if task_id and total_tok:
                    task_tokens[task_id] = total_tok
                job["total_tokens"] = sum(task_tokens.values())
                dur_str = f"{dur_ms/1000:.1f}s" if dur_ms else ""
                tokens_str = f"{job['total_tokens']} tokens" if job["total_tokens"] else ""
                push_event(job_id, "task_progress", {
                    "total_tokens": job.get("total_tokens", 0),
                    "duration": dur_str,
                    "tool_uses": tool_uses,
                    "message": f"Progress: {tokens_str} {dur_str}".strip(),
                })

            elif isinstance(message, TaskNotificationMessage):
                status = getattr(message, "status", None)
                status_str = getattr(status, "value", str(status)) if status else "done"
                # Capture final task usage for accurate token tracking
                task_id = getattr(message, "task_id", None)
                usage = getattr(message, "usage", None)
                if task_id and usage:
                    total_tok = 0
                    if isinstance(usage, dict):
                        total_tok = usage.get("total_tokens", 0) or 0
                    else:
                        total_tok = getattr(usage, "total_tokens", 0) or 0
                    if total_tok:
                        task_tokens = job.setdefault("task_tokens", {})
                        task_tokens[task_id] = total_tok
                        job["total_tokens"] = sum(task_tokens.values())
                push_event(job_id, "task_done", {
                    "status": status_str,
                    "message": f"Subagent finished: {status_str}",
                })

    try:
        asyncio.run(_run())
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        push_event(job_id, "error", {"message": str(e)})

    # List generated files
    gen_files = sorted(str(f.relative_to(output_dir)) for f in output_dir.rglob("*") if f.is_file())
    job["generated_files"] = gen_files

    elapsed = time.time() - start_time
    job["duration"] = round(elapsed, 1)

    push_event(job_id, "done", {
        "message": "Pipeline finished",
        "files": gen_files,
        "duration": round(elapsed, 1),
        "status": job.get("status", "completed"),
        "total_tokens": job.get("total_tokens", 0),
        "total_input_tokens": job.get("total_input_tokens", 0),
        "total_output_tokens": job.get("total_output_tokens", 0),
        "total_cost_usd": job.get("total_cost_usd"),
    })


# ═══════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/demo")
def demo():
    return render_template("demo.html")


@app.route("/api/run", methods=["POST"])
def api_run():
    data = request.get_json() or {}
    arxiv = data.get("arxiv", "").strip()
    pdf_path = data.get("pdf_path", "").strip() or None
    method_name = data.get("method_name", "").strip() or None
    data_path = data.get("data_path", "").strip() or None

    enable_review = data.get("enable_review", True)
    enable_validation = data.get("enable_validation", True)
    enable_smoke_test = data.get("enable_smoke_test", True)
    enable_training = data.get("enable_training", False)
    expected_psnr = data.get("expected_psnr") or None
    if expected_psnr is not None:
        expected_psnr = float(expected_psnr)
    dataset = data.get("dataset", "mipnerf360").strip()
    scenes_raw = data.get("scenes", "").strip()
    scenes = [s.strip() for s in scenes_raw.split(",") if s.strip()] if scenes_raw else None
    model_preset = data.get("model", "sonnet")  # opus, sonnet, haiku, smart
    num_views = data.get("num_views", -1)
    if num_views is not None:
        num_views = int(num_views)
    else:
        num_views = -1
    viewer_port = data.get("viewer_port") or None
    if viewer_port is not None:
        viewer_port = int(viewer_port)

    # Handle PDF upload
    if not arxiv and not pdf_path:
        pdf_file = request.files.get("pdf")
        if pdf_file and pdf_file.filename:
            fname = secure_filename(pdf_file.filename)
            upload_dir = DEFAULT_CONFIG.workspace_dir / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = str(upload_dir / fname)
            pdf_file.save(pdf_path)

    if not arxiv and not pdf_path:
        return jsonify({"error": "Provide an arXiv URL/ID or PDF path"}), 400

    # Create job
    job_id = uuid.uuid4().hex[:12]
    config = PipelineConfig()
    config.enable_review = enable_review
    config.enable_validation = enable_validation
    config.enable_smoke_test = enable_smoke_test
    config.enable_training = enable_training
    config.expected_psnr = expected_psnr
    config.dataset = dataset if dataset in ("mipnerf360", "llff") else "mipnerf360"
    if scenes:
        config.scenes = scenes
    else:
        # Default scene per dataset
        config.scenes = ["garden"] if config.dataset == "mipnerf360" else ["fern"]
    config.num_views = num_views
    config.viewer_port = viewer_port
    config.ensure_dirs()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + job_id[:6]
    workspace = config.workspace_dir / run_id
    output_dir = config.generated_dir / run_id
    workspace.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    JOBS[job_id] = {
        "id": job_id,
        "status": "running",
        "run_id": run_id,
        "workspace": str(workspace),
        "output_dir": str(output_dir),
        "arxiv": arxiv,
        "created": datetime.now().isoformat(),
        "total_input_tokens": 0,
        "total_output_tokens": 0,
    }
    JOB_QUEUES[job_id] = queue.Queue()
    JOB_STOP_EVENTS[job_id] = threading.Event()

    # Resolve model selection into orchestrator model + per-agent overrides
    if model_preset == "smart":
        orchestrator_model = "sonnet"
        agent_models = {"coder": "opus", "debugger": "opus", "*": "sonnet"}
    elif model_preset in ("opus", "sonnet", "haiku"):
        orchestrator_model = model_preset
        agent_models = {"*": model_preset}
    else:
        orchestrator_model = "sonnet"
        agent_models = None

    t = threading.Thread(
        target=_run_pipeline_bg,
        args=(job_id, config),
        kwargs={
            "arxiv": arxiv or None,
            "pdf_path": pdf_path,
            "method_name": method_name,
            "data_path": data_path,
            "model": orchestrator_model,
            "agent_models": agent_models,
        },
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id, "run_id": run_id})


@app.route("/api/events/<job_id>")
def api_events(job_id):
    """SSE endpoint for real-time streaming."""
    q = JOB_QUEUES.get(job_id)
    if not q:
        return jsonify({"error": "Job not found"}), 404

    def generate():
        while True:
            try:
                event = q.get(timeout=30)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") in ("done", "error"):
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/job/<job_id>")
def api_job(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/api/jobs")
def api_jobs():
    return jsonify([
        {"id": j["id"], "status": j.get("status"), "arxiv": j.get("arxiv"), "created": j.get("created")}
        for j in JOBS.values()
    ])


@app.route("/api/stop/<job_id>", methods=["POST"])
def api_stop(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job.get("status") != "running":
        return jsonify({"error": "Job is not running"}), 400
    stop_event = JOB_STOP_EVENTS.get(job_id)
    if stop_event:
        stop_event.set()
    return jsonify({"ok": True})


# ═══════════════════════════════════════════════════════════════════
# INVENT ROUTES (Evolutionary Tournament)
# ═══════════════════════════════════════════════════════════════════

INVENT_JOBS: dict[str, dict[str, Any]] = {}
INVENT_QUEUES: dict[str, queue.Queue] = {}


def _run_invent_bg(job_id: str, arxiv: str, impl_path: str, data_path: str, gpu: str,
                    viewer_port: int | None = None, model: str | None = None):
    """Run the invent pipeline in a background thread."""
    job = INVENT_JOBS[job_id]
    workspace = Path(job["workspace"])
    q = INVENT_QUEUES[job_id]

    try:
        result = asyncio.run(run_invent_pipeline(
            arxiv=arxiv or None,
            impl_path=impl_path,
            workspace=workspace,
            event_queue=q,
            gpu=gpu,
            data_path=data_path or None,
            viewer_port=viewer_port,
            model=model,
        ))
        job["status"] = "completed"
        job["result"] = result
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        q.put({"type": "error", "message": str(e)})
        q.put({"type": "done", "message": f"Tournament failed: {e}"})


@app.route("/invent")
def invent_page():
    return render_template("invent.html")


@app.route("/api/invent/run", methods=["POST"])
def api_invent_run():
    data = request.get_json() or {}
    impl_path = data.get("impl_path", "").strip()
    arxiv = data.get("arxiv", "").strip()
    data_path = data.get("data_path", "").strip()
    gpu = data.get("gpu", "0").strip()
    model = data.get("model", "sonnet").strip()
    viewer_port = data.get("viewer_port") or None
    if viewer_port is not None:
        viewer_port = int(viewer_port)

    if not impl_path:
        return jsonify({"error": "Implementation path is required"}), 400

    impl = Path(impl_path)
    if not impl.exists():
        return jsonify({"error": f"Invalid implementation path: {impl_path}"}), 400
    # Find the method package directory (any dir with __init__.py, excluding __pycache__ and .egg-info)
    pkg_dirs = [
        d for d in impl.iterdir()
        if d.is_dir() and (d / "__init__.py").exists()
        and not d.name.startswith("__") and not d.name.endswith(".egg-info")
    ]
    if not pkg_dirs:
        return jsonify({"error": f"No method package directory found in: {impl_path}"}), 400

    job_id = uuid.uuid4().hex[:12]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_invent_" + job_id[:6]
    workspace = DEFAULT_CONFIG.workspace_dir / run_id
    workspace.mkdir(parents=True, exist_ok=True)

    INVENT_JOBS[job_id] = {
        "id": job_id,
        "status": "running",
        "run_id": run_id,
        "workspace": str(workspace),
        "impl_path": impl_path,
        "arxiv": arxiv,
        "gpu": gpu,
        "model": model,
        "created": datetime.now().isoformat(),
    }
    INVENT_QUEUES[job_id] = queue.Queue()

    t = threading.Thread(
        target=_run_invent_bg,
        args=(job_id, arxiv, impl_path, data_path, gpu, viewer_port, model),
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id, "run_id": run_id})


@app.route("/api/invent/events/<job_id>")
def api_invent_events(job_id):
    """SSE endpoint for invent pipeline events."""
    q = INVENT_QUEUES.get(job_id)
    if not q:
        return jsonify({"error": "Job not found"}), 404

    def generate():
        while True:
            try:
                event = q.get(timeout=30)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") in ("done", "error"):
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/invent/job/<job_id>")
def api_invent_job(job_id):
    job = INVENT_JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


# ═══════════════════════════════════════════════════════════════════
# SCENE-DRIVEN INVENT ROUTES
# ═══════════════════════════════════════════════════════════════════

def _run_scene_invent_bg(job_id: str, scene_description: str, data_path: str,
                         gpu: str, viewer_port: int | None = None,
                         method_name: str | None = None, model: str | None = None):
    """Run the scene-driven invent pipeline in a background thread."""
    job = INVENT_JOBS[job_id]
    workspace = Path(job["workspace"])
    q = INVENT_QUEUES[job_id]

    try:
        result = asyncio.run(run_scene_invent_pipeline(
            scene_description=scene_description,
            workspace=workspace,
            event_queue=q,
            gpu=gpu,
            data_path=data_path or None,
            viewer_port=viewer_port,
            method_name=method_name or None,
            model=model,
        ))
        job["status"] = "completed"
        job["result"] = result
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        q.put({"type": "error", "message": str(e)})
        q.put({"type": "done", "message": f"Scene tournament failed: {e}"})


@app.route("/api/invent/scene", methods=["POST"])
def api_invent_scene():
    """Start a scene-driven evolutionary tournament."""
    data = request.get_json() or {}
    scene_description = data.get("scene_description", "").strip()
    data_path = data.get("data_path", "").strip()
    gpu = data.get("gpu", "0").strip()
    model = data.get("model", "sonnet").strip()
    method_name = data.get("method_name", "").strip() or None
    viewer_port = data.get("viewer_port") or None
    if viewer_port is not None:
        viewer_port = int(viewer_port)

    if not scene_description:
        return jsonify({"error": "Scene description is required"}), 400

    job_id = uuid.uuid4().hex[:12]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_scene_" + job_id[:6]
    workspace = DEFAULT_CONFIG.workspace_dir / run_id
    workspace.mkdir(parents=True, exist_ok=True)

    INVENT_JOBS[job_id] = {
        "id": job_id,
        "status": "running",
        "mode": "scene",
        "run_id": run_id,
        "workspace": str(workspace),
        "scene_description": scene_description,
        "gpu": gpu,
        "model": model,
        "created": datetime.now().isoformat(),
    }
    INVENT_QUEUES[job_id] = queue.Queue()

    t = threading.Thread(
        target=_run_scene_invent_bg,
        args=(job_id, scene_description, data_path, gpu, viewer_port, method_name, model),
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id, "run_id": run_id})


# ═══════════════════════════════════════════════════════════════════
# DISCOVER ROUTES (Iterative Evolution)
# ═══════════════════════════════════════════════════════════════════

DISCOVER_JOBS: dict[str, dict[str, Any]] = {}
DISCOVER_QUEUES: dict[str, queue.Queue] = {}
DISCOVER_STOP_EVENTS: dict[str, threading.Event] = {}


def _run_discover_bg(job_id: str, impl_path: str, data_path: str, gpu: str,
                     scene: str, max_iters: int, model: str | None = None,
                     num_views: int = -1):
    """Run the discover pipeline in a background thread."""
    job = DISCOVER_JOBS[job_id]
    workspace = Path(job["workspace"])
    q = DISCOVER_QUEUES[job_id]
    stop_event = DISCOVER_STOP_EVENTS[job_id]

    try:
        result = asyncio.run(run_discover_pipeline(
            impl_path=impl_path,
            workspace=workspace,
            event_queue=q,
            stop_event=stop_event,
            gpu=gpu,
            data_path=data_path or None,
            scene=scene,
            max_iters=max_iters,
            model=model,
            num_views=num_views,
        ))
        job["status"] = "completed"
        job["result"] = result
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        q.put({"type": "error", "message": str(e)})
        q.put({"type": "done", "message": f"Discovery failed: {e}"})


@app.route("/discover")
def discover_page():
    return render_template("discover.html")


@app.route("/api/discover/run", methods=["POST"])
def api_discover_run():
    data = request.get_json() or {}
    impl_path = data.get("impl_path", "").strip()
    data_path = data.get("data_path", "").strip()
    gpu = data.get("gpu", "0").strip()
    model = data.get("model", "sonnet").strip()
    scene = data.get("scene", "garden").strip()
    max_iters = int(data.get("max_iters", 10000))
    num_views = int(data.get("num_views", -1))

    if not impl_path:
        return jsonify({"error": "Implementation path is required"}), 400

    impl = Path(impl_path)
    if not impl.exists():
        return jsonify({"error": f"Invalid implementation path: {impl_path}"}), 400

    # Verify it has a method package
    pkg_dirs = [
        d for d in impl.iterdir()
        if d.is_dir() and (d / "__init__.py").exists()
        and not d.name.startswith("__") and not d.name.endswith(".egg-info")
    ]
    if not pkg_dirs:
        return jsonify({"error": f"No method package directory found in: {impl_path}"}), 400

    job_id = uuid.uuid4().hex[:12]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_discover_" + job_id[:6]
    workspace = DEFAULT_CONFIG.workspace_dir / run_id
    workspace.mkdir(parents=True, exist_ok=True)

    DISCOVER_JOBS[job_id] = {
        "id": job_id,
        "status": "running",
        "run_id": run_id,
        "workspace": str(workspace),
        "impl_path": impl_path,
        "gpu": gpu,
        "model": model,
        "scene": scene,
        "max_iters": max_iters,
        "num_views": num_views,
        "created": datetime.now().isoformat(),
    }
    DISCOVER_QUEUES[job_id] = queue.Queue()
    DISCOVER_STOP_EVENTS[job_id] = threading.Event()

    t = threading.Thread(
        target=_run_discover_bg,
        args=(job_id, impl_path, data_path, gpu, scene, max_iters, model, num_views),
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id, "run_id": run_id})


@app.route("/api/discover/events/<job_id>")
def api_discover_events(job_id):
    """SSE endpoint for discover pipeline events."""
    q = DISCOVER_QUEUES.get(job_id)
    if not q:
        return jsonify({"error": "Job not found"}), 404

    def generate():
        while True:
            try:
                event = q.get(timeout=30)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") in ("done", "error"):
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/discover/stop/<job_id>", methods=["POST"])
def api_discover_stop(job_id):
    """Gracefully stop a running discover pipeline."""
    job = DISCOVER_JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job.get("status") != "running":
        return jsonify({"error": "Job is not running"}), 400
    stop_event = DISCOVER_STOP_EVENTS.get(job_id)
    if stop_event:
        stop_event.set()
    return jsonify({"ok": True, "message": "Stop signal sent — current generation will complete first"})


@app.route("/api/discover/job/<job_id>")
def api_discover_job(job_id):
    job = DISCOVER_JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5007"))
    print(f"Splatify Claude Agents — http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
