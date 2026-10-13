#!/usr/bin/env python3
"""
Splatify — Claude Agent SDK Multi-Agent Pipeline

Converts research papers about Gaussian Splatting methods into complete
nerfstudio/gsplat implementations using the gs_template framework and
specialized Claude subagents for each pipeline stage.

Usage:
    # From arXiv URL
    python main.py --arxiv 2308.12345

    # From local PDF
    python main.py --pdf /path/to/paper.pdf

    # With options
    python main.py --arxiv 2308.12345 --method-name my_gs --no-review --train

    # With custom data path
    python main.py --arxiv 2308.12345 --data /path/to/dataset --train --max-iters 30000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AgentDefinition,
    # Message types
    ResultMessage,
    SystemMessage,
    AssistantMessage,
    UserMessage,
    # Content block types
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ToolResultBlock,
    # Subagent task messages
    TaskStartedMessage,
    TaskProgressMessage,
    TaskNotificationMessage,
)

from config import PipelineConfig, get_claude_auth_env
from agents.definitions import build_agent_definitions


# ═══════════════════════════════════════════════════════════════════
# TERMINAL LOGGING HELPERS
# ═══════════════════════════════════════════════════════════════════

# ANSI color codes
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BG_BLUE = "\033[44m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_RED = "\033[41m"


def log_header(msg: str):
    print(f"\n{C.BOLD}{C.BG_BLUE}{C.WHITE} {msg} {C.RESET}")


def log_stage(msg: str):
    print(f"\n{C.BOLD}{C.GREEN}>>> {msg}{C.RESET}")


def log_agent(agent: str, msg: str):
    print(f"{C.BOLD}{C.CYAN}[{agent}]{C.RESET} {msg}")


def log_tool(name: str, summary: str):
    print(f"  {C.YELLOW}>> tool:{C.RESET} {C.BOLD}{name}{C.RESET} {C.DIM}{summary}{C.RESET}")


def log_tool_result(summary: str):
    print(f"  {C.DIM}<< {summary}{C.RESET}")


def log_thinking(text: str):
    # Show first few lines of thinking
    lines = text.strip().split("\n")
    preview = lines[0][:120]
    if len(lines) > 1 or len(lines[0]) > 120:
        preview += "..."
    print(f"  {C.MAGENTA}(thinking) {preview}{C.RESET}")


def log_text(text: str, prefix: str = ""):
    """Print agent text output, line by line."""
    for line in text.split("\n"):
        print(f"  {C.WHITE}{prefix}{line}{C.RESET}")


def log_error(msg: str):
    print(f"{C.BOLD}{C.RED}ERROR: {msg}{C.RESET}")


def log_success(msg: str):
    print(f"{C.BOLD}{C.GREEN}OK: {msg}{C.RESET}")


def log_info(msg: str):
    print(f"{C.DIM}{msg}{C.RESET}")


def log_task_event(event_type: str, msg: str):
    print(f"  {C.BLUE}[task:{event_type}]{C.RESET} {msg}")


def _summarize_tool_input(tool_input: dict) -> str:
    """Create a short summary of tool input for logging."""
    if not isinstance(tool_input, dict):
        return str(tool_input)[:80]
    # Common patterns
    if "file_path" in tool_input:
        return tool_input["file_path"]
    if "path" in tool_input:
        return tool_input["path"]
    if "command" in tool_input:
        cmd = tool_input["command"]
        return cmd[:100] + ("..." if len(cmd) > 100 else "")
    if "pattern" in tool_input:
        return f"pattern={tool_input['pattern']}"
    if "prompt" in tool_input:
        p = tool_input["prompt"]
        return p[:80] + ("..." if len(p) > 80 else "")
    if "description" in tool_input:
        return tool_input["description"][:80]
    return str(tool_input)[:80]


def _summarize_tool_result(content) -> str:
    """Create a short summary of a tool result."""
    if isinstance(content, str):
        lines = content.strip().split("\n")
        if len(lines) <= 2:
            return content.strip()[:150]
        return f"{lines[0][:100]}... ({len(lines)} lines)"
    if isinstance(content, list):
        return f"({len(content)} blocks)"
    return str(content)[:100]


# ═══════════════════════════════════════════════════════════════════
# MESSAGE HANDLER
# ═══════════════════════════════════════════════════════════════════

def handle_message(message, start_time: float):
    """Handle and log every message from the agent stream."""
    elapsed = time.time() - start_time

    # ── Result: final output ──
    if isinstance(message, ResultMessage):
        log_header(f"PIPELINE COMPLETE ({elapsed:.1f}s)")
        stop = getattr(message, "stop_reason", "end_turn")
        log_info(f"Stop reason: {stop}")
        if message.result:
            print()
            print(message.result)
        return {"type": "result", "text": message.result}

    # ── System messages ──
    if isinstance(message, SystemMessage):
        subtype = getattr(message, "subtype", "unknown")
        session_id = getattr(message, "session_id", None)
        if session_id:
            log_info(f"[system:{subtype}] session={session_id}")
            return {"type": "system", "session_id": session_id}
        else:
            log_info(f"[system:{subtype}] {_msg_preview(message)}")
        return {"type": "system"}

    # ── Assistant messages (text, thinking, tool calls) ──
    if isinstance(message, AssistantMessage):
        content = getattr(message, "content", [])
        for block in content:
            if isinstance(block, ThinkingBlock):
                thinking_text = getattr(block, "thinking", "")
                if thinking_text:
                    log_thinking(thinking_text)

            elif isinstance(block, TextBlock):
                text = getattr(block, "text", "")
                if text:
                    log_text(text)

            elif isinstance(block, ToolUseBlock):
                name = getattr(block, "name", "?")
                tool_input = getattr(block, "input", {})
                summary = _summarize_tool_input(tool_input)

                # Special handling for Agent tool (subagent invocation)
                if name == "Agent":
                    agent_type = tool_input.get("subagent_type") or tool_input.get("agent_type") or tool_input.get("type") or "subagent"
                    desc = tool_input.get("description", "")
                    prompt_preview = tool_input.get("prompt", "")[:100]
                    log_stage(f"Invoking subagent: {agent_type}")
                    if desc:
                        log_agent(agent_type, desc)
                    if prompt_preview:
                        log_info(f"  prompt: {prompt_preview}...")
                else:
                    log_tool(name, summary)

            else:
                # Other block types
                block_type = getattr(block, "type", type(block).__name__)
                log_info(f"  [{block_type}]")

        return {"type": "assistant"}

    # ── Task started (subagent spawned) ──
    if isinstance(message, TaskStartedMessage):
        task_id = getattr(message, "task_id", "?")
        agent_type = getattr(message, "agent_type", "subagent")
        log_task_event("started", f"agent={agent_type} task={task_id}")
        return {"type": "task_started"}

    # ── Task progress (subagent working) ──
    if isinstance(message, TaskProgressMessage):
        task_id = getattr(message, "task_id", "?")
        usage = getattr(message, "usage", None)
        tool_count = getattr(message, "tool_use_count", None)
        duration = getattr(message, "duration_ms", None)
        parts = []
        if usage:
            inp = getattr(usage, "input_tokens", 0) or 0
            out = getattr(usage, "output_tokens", 0) or 0
            if inp or out:
                parts.append(f"tokens={inp}+{out}")
        if tool_count:
            parts.append(f"tools={tool_count}")
        if duration:
            parts.append(f"{duration/1000:.1f}s")
        log_task_event("progress", " ".join(parts) if parts else f"task={task_id}")
        return {"type": "task_progress"}

    # ── Task notification (subagent done) ──
    if isinstance(message, TaskNotificationMessage):
        task_id = getattr(message, "task_id", "?")
        status = getattr(message, "status", None)
        status_str = ""
        if status:
            status_str = getattr(status, "value", str(status))
        tool_use_id = getattr(message, "tool_use_id", None)
        log_task_event("done", f"status={status_str} task={task_id}")
        return {"type": "task_notification"}

    # ── Catch-all for unknown message types ──
    msg_type = type(message).__name__
    log_info(f"[{msg_type}] {_msg_preview(message)}")
    return {"type": msg_type}


def _msg_preview(msg) -> str:
    """Get a short preview string for any message object."""
    # Try common attributes
    for attr in ("result", "text", "content", "message"):
        val = getattr(msg, attr, None)
        if val and isinstance(val, str):
            return val[:100]
    return ""


# ═══════════════════════════════════════════════════════════════════
# ORCHESTRATOR PROMPT
# ═══════════════════════════════════════════════════════════════════

def build_orchestrator_prompt(
    *,
    arxiv: str | None = None,
    pdf_path: str | None = None,
    method_name: str | None = None,
    data_path: str | None = None,
    config: PipelineConfig,
    workspace: Path,
    output_dir: Path,
) -> str:
    """Build the task prompt for the orchestrator agent."""

    # Input source
    if arxiv:
        input_section = f"ArXiv paper: {arxiv}"
    elif pdf_path:
        input_section = f"Local PDF: {pdf_path}"
    else:
        raise ValueError("Must provide either --arxiv or --pdf")

    method_override = ""
    if method_name:
        method_override = f"\nIMPORTANT: Use this exact method name: {method_name}"

    data_section = ""
    if data_path:
        data_section = f"\nTraining data path: {data_path}"

    # Pipeline steps based on config
    optional_steps = []
    if config.enable_review:
        optional_steps.append(
            f"5. **Review** (up to {config.max_review_iterations} iterations): "
            "Use the `reviewer` agent. If not approved, send feedback to `coder` to fix."
        )
    if config.enable_validation:
        optional_steps.append(
            "6. **Validate**: Use the `validator` agent. If gaps found, "
            "use `coder` to implement missing novelties."
        )
    optional_steps.append(
        "7. **Integrate**: Use the `integrator` agent to check cross-file consistency."
    )
    if config.enable_smoke_test:
        optional_steps.append(
            f"8. **Test**: Use the `tester` agent. If it fails, use `debugger` to fix "
            f"(up to {config.max_debug_iterations} attempts), then re-test."
        )
    if config.enable_training:
        training_output_dir = output_dir / "training_outputs"
        viewer_port_flag = f" --viewer-port {config.viewer_port}" if config.viewer_port else ""
        viewer_url_note = ""
        if config.viewer_port:
            viewer_url_note = (
                f"\n   The nerfstudio viewer will be available at:\n"
                f"   https://viewer.nerfstudio.com/versions/24-05/?websocket_url=ws://localhost:{config.viewer_port}"
            )
        scenes_str = config.get_scenes_str()
        dataset_flag = f" --dataset {config.dataset}" if config.dataset != "mipnerf360" else ""
        num_views_flag = f" --num-views {config.num_views}" if config.num_views > 0 else ""
        num_views_str = str(config.num_views) if config.num_views > 0 else "all"
        optional_steps.append(
            f"9. **Train**: Use the `tester` agent to run full training via eval.py:\n"
            f"   `cd {config.gs_template_root.parent} && "
            f"python eval.py --gpu 0 --method <METHOD_NAME_CLI> --scenes {scenes_str}"
            f"{dataset_flag}{num_views_flag} "
            f"--output-dir {training_output_dir}{viewer_port_flag}`\n"
            f"   This trains for {config.default_max_iters} iterations and evaluates PSNR/SSIM/LPIPS.\n"
            f"   Dataset: {config.dataset}, Scene(s): {scenes_str}, Views: {num_views_str}\n"
            f"   Training outputs (checkpoints, TensorBoard logs) will be saved under: {training_output_dir}"
            f"{viewer_url_note}"
        )
        psnr_target = config.get_psnr_target()
        first_scene = config.scenes[0] if config.scenes else "garden"
        optional_steps.append(
            f"10. **PSNR Review** (up to {config.max_training_review_iterations} iterations):\n"
            f"   After training completes, analyze the PSNR training curve to check quality:\n"
            f"   a. Run: `cd {config.gs_template_root.parent} && "
            f"python read_tb.py {training_output_dir}/{first_scene}/<METHOD_NAME_CLI>/<LATEST_TIMESTAMP>/`\n"
            f"      (Find the latest timestamp dir inside {training_output_dir}/{first_scene}/<METHOD_NAME_CLI>/)\n"
            f"   b. Review the PSNR curve output. A healthy training curve should:\n"
            f"      - Reach final PSNR >= {psnr_target} dB (baseline for {config.dataset}/{first_scene})\n"
            f"      - Show steady improvement without major drops\n"
            f"      - Not have NaN or zero values\n"
            f"      - Plateau or only slowly improve near the end (not still steeply rising)\n"
            f"   c. If the PSNR curve looks BAD (low final PSNR, NaN, sudden drops, training collapse):\n"
            f"      - Analyze what went wrong (likely a bug in loss computation, rendering, or parameter init)\n"
            f"      - Use the `coder` agent to fix the issue in the generated code\n"
            f"      - Re-install the package and re-run training (go back to step 9)\n"
            f"   d. If the PSNR curve looks GOOD — proceed to the final report.\n"
            f"   NOTE: The expected PSNR target for {config.dataset}/{first_scene} is >= {psnr_target} dB.\n"
            f"   This is derived from the splatfacto baseline for this dataset minus a 2 dB margin.\n"
            f"   If the method is a novel approach, PSNR within this range means it's working correctly."
        )
    optional_text = "\n".join(optional_steps)

    return f"""## Task
Convert a Gaussian Splatting research paper into a complete nerfstudio method implementation
using the gs_template framework (extends SplatfactoModel with gsplat rasterization).

## Input
{input_section}{method_override}{data_section}

## Workspace
- Working directory: {workspace}
- Output directory (write generated code here): {output_dir}

## gs_template Framework
The implementation must use the gs_template framework at: {config.gs_template_root}
This framework provides:
- GSTemplateModel base class with 9 extension points (model.py)
- Shared rendering helpers using gsplat (rendering.py)
- 30+ reusable loss functions (losses.py)
- Custom DataManagers for preprocessing (datamanagers.py)
- Shading utilities (shading.py)
- Custom optimizers (optimizers.py)
- 20+ implemented paper examples (methods/)

## Pipeline Steps

Execute these steps IN ORDER. Use the appropriate subagent for each step.

1. **Parse**: Use the `parser` agent to download (if needed) and extract the paper.
   Tell it to save results in the workspace directory: {workspace}

2. **Plan**: Use the `planner` agent to read the cleaned paper and the gs_template framework.
   Tell it to read from {workspace}/cleaned_paper.md and the framework files.
   Tell it to write the plan to {workspace}/dag_plan.json

3. **Read the plan**: After planning, read {workspace}/dag_plan.json yourself to get
   the method_name, complexity_tier, and understand the architecture.
   The `method_name` (snake_case) from the plan is used as BOTH the Python package
   directory name AND the module name. For example, if the planner outputs
   method_name="dashgaussian", then the package directory is `dashgaussian/` (NOT `gs_method/`).

4. **Generate Code**: Use the `coder` agent to generate the implementation.
   Tell it:
   - The workspace path: {workspace}
   - The output directory: {output_dir}
   - The method name (from the plan) — this is also the package directory name
   - To read the cleaned paper, DAG plan, framework templates, and example implementations

{optional_text}

## REQUIRED OUTPUT FILES — ALL MANDATORY
The generated implementation MUST contain ALL of these files in {{output_dir}}.
CRITICAL: The Python package directory MUST be named after the method (method_name_snake
from the DAG plan), NOT "gs_method". For example, if the method is "dashgaussian":
1. `dashgaussian/__init__.py` — Package init, exports Model + Config + MethodSpec
2. `dashgaussian/model.py` — Main model class extending GSTemplateModel
3. `dashgaussian/losses.py` — Loss functions (custom + imports from gs_template.losses)
4. `dashgaussian/rendering.py` — Rendering (custom or re-export from gs_template.rendering)
5. `dashgaussian/configs.py` — MethodSpecification + TrainerConfig for nerfstudio
6. `dashgaussian/datamanager.py` — DataManager config (custom or re-export)
7. `README.md` — Documentation
8. `pyproject.toml` — Package with nerfstudio entry points

All internal imports must use the method name as the package, e.g.:
  `from dashgaussian.model import DashGaussianModel`
  `from dashgaussian.configs import dashgaussian_method`
And pyproject.toml entry points must reference the method package:
  `dashgaussian = "dashgaussian.configs:dashgaussian_method"`

CRITICAL: Tell the coder agent to generate ALL 8 files. After the coder finishes,
verify ALL files exist before proceeding. If any are missing, call the coder again
to generate the missing files.

## Important Rules
- Always tell subagents the exact workspace path ({workspace}) and output path ({output_dir})
- The package directory name MUST match the method_name_snake from the DAG plan
- After each agent completes, read its output files to check results before proceeding
- After the coder step, list {{output_dir}}/{{method_name_snake}}/ to verify ALL required files are present
- If a step fails, report the error clearly
- The final output should be a complete, pip-installable nerfstudio method in {output_dir}
- The method must work with: ns-train <method-name-cli> --data <path>

## Final Report
After all steps complete, summarize:
- Method name (snake_case and kebab-case)
- Package directory name (same as snake_case method name)
- Complexity tier
- Output directory
- Files generated
- Extension points used
- Review/validation/test results
- Any issues encountered
"""


# ═══════════════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ═══════════════════════════════════════════════════════════════════

async def run_pipeline(
    *,
    arxiv: str | None = None,
    pdf_path: str | None = None,
    method_name: str | None = None,
    data_path: str | None = None,
    config: PipelineConfig | None = None,
) -> dict:
    """Run the full Splatify pipeline using Claude Agent SDK."""

    if config is None:
        config = PipelineConfig()
    config.ensure_dirs()

    # Create a unique workspace for this run
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    workspace = config.workspace_dir / run_id
    output_dir = config.generated_dir / run_id
    workspace.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_header("Splatify Multi-Agent Pipeline")
    print(f"  Run ID:    {C.BOLD}{run_id}{C.RESET}")
    print(f"  Workspace: {workspace}")
    print(f"  Output:    {output_dir}")
    if arxiv:
        print(f"  Paper:     arXiv:{arxiv}")
    elif pdf_path:
        print(f"  Paper:     {pdf_path}")
    print()

    # Build agent definitions
    agents = build_agent_definitions(config)
    log_info(f"Loaded {len(agents)} subagent definitions: {', '.join(agents.keys())}")

    # Build orchestrator prompt
    prompt = build_orchestrator_prompt(
        arxiv=arxiv,
        pdf_path=pdf_path,
        method_name=method_name,
        data_path=data_path or config.default_dataset,
        config=config,
        workspace=workspace,
        output_dir=output_dir,
    )

    # Save the prompt for debugging
    (workspace / "orchestrator_prompt.txt").write_text(prompt)
    log_info(f"Orchestrator prompt saved to {workspace}/orchestrator_prompt.txt")

    # Run the orchestrator
    start_time = time.time()
    result_text = ""
    session_id = None
    message_count = 0

    log_stage("Starting orchestrator agent...")

    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            cwd=str(config.base_dir),
            allowed_tools=["Read", "Write", "Bash", "Glob", "Grep", "Agent"],
            agents=agents,
            permission_mode="acceptEdits",
            max_turns=200,
            env=get_claude_auth_env(),
        ),
    ):
        message_count += 1
        info = handle_message(message, start_time)

        if info["type"] == "result":
            result_text = info.get("text", "")
        elif info["type"] == "system" and "session_id" in info:
            session_id = info["session_id"]

    # Summary
    elapsed = time.time() - start_time
    log_header("RUN SUMMARY")
    print(f"  Run ID:      {run_id}")
    print(f"  Duration:    {elapsed:.1f}s")
    print(f"  Messages:    {message_count}")
    print(f"  Session:     {session_id or 'N/A'}")
    print(f"  Output:      {output_dir}")
    print()

    # List generated files
    gen_files = sorted(f.relative_to(output_dir) for f in output_dir.rglob("*") if f.is_file())
    if gen_files:
        log_success(f"Generated {len(gen_files)} files:")
        for f in gen_files:
            print(f"    {f}")
    else:
        log_error("No files generated!")

    # Save results
    result = {
        "run_id": run_id,
        "workspace": str(workspace),
        "output_dir": str(output_dir),
        "session_id": session_id,
        "duration_s": round(elapsed, 1),
        "message_count": message_count,
        "generated_files": [str(f) for f in gen_files],
        "result": result_text,
    }
    (workspace / "result.json").write_text(
        json.dumps(result, indent=2, default=str)
    )

    return result


# ═══════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Splatify: Convert Gaussian Splatting papers to nerfstudio implementations using Claude agents"
    )
    parser.add_argument("--arxiv", type=str, help="arXiv URL or ID (e.g., 2308.12345)")
    parser.add_argument("--pdf", type=str, help="Path to local PDF file")
    parser.add_argument("--method-name", type=str, help="Override method name (snake_case)")
    parser.add_argument("--data", type=str, help="Path to training dataset")
    parser.add_argument("--dataset", type=str, default="mipnerf360",
                        choices=["mipnerf360", "llff"], help="Dataset type (default: mipnerf360)")
    parser.add_argument("--scenes", type=str, nargs="+", default=["garden"],
                        help="Scene(s) to train on (default: garden)")
    parser.add_argument("--max-iters", type=int, default=30000, help="Max training iterations")
    parser.add_argument("--num-views", type=int, default=-1,
                        help="Number of training views (-1 = all, e.g. 3,6,9,12,24)")

    # Pipeline toggles
    parser.add_argument("--no-review", action="store_true", help="Skip code review step")
    parser.add_argument("--no-validation", action="store_true", help="Skip novelty validation")
    parser.add_argument("--no-test", action="store_true", help="Skip smoke testing")
    parser.add_argument("--train", action="store_true", help="Enable full training")

    # Config
    parser.add_argument("--gs-template", type=str, help="Path to gs_template framework")

    args = parser.parse_args()

    if not args.arxiv and not args.pdf:
        parser.error("Must provide either --arxiv or --pdf")

    # Build config
    config = PipelineConfig()
    if args.gs_template:
        config.gs_template_root = Path(args.gs_template)
    config.enable_review = not args.no_review
    config.enable_validation = not args.no_validation
    config.enable_smoke_test = not args.no_test
    config.enable_training = args.train
    config.dataset = args.dataset
    config.scenes = args.scenes
    config.default_max_iters = args.max_iters
    config.num_views = args.num_views

    # Run
    async def _run():
        return await run_pipeline(
            arxiv=args.arxiv,
            pdf_path=args.pdf,
            method_name=args.method_name,
            data_path=args.data,
            config=config,
        )

    result = asyncio.run(_run())

    print(f"\n{C.BOLD}Output directory: {result['output_dir']}{C.RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
