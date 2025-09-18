from __future__ import annotations

import sys
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


def _resolve_run_tasks_path(start_dir: Path) -> Path:
    candidates = [
        start_dir / "run_tasks.py",
        start_dir / "CRMArena" / "run_tasks.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate run_tasks.py. Checked: "
        + ", ".join(str(c) for c in candidates)
    )


REPO_ROOT = Path(__file__).resolve().parent
RUN_TASKS_PATH = _resolve_run_tasks_path(REPO_ROOT)


@dataclass
class RunConfig:
    model: str = "google/gemini-2.5-flash-lite"
    llm_provider: str = "openrouter"
    task_category: str = "all"
    org_type: str = "b2b"
    task_limit: int = 5
    agent_strategy: str = "react"
    agent_eval_mode: str = "default"
    log_dir: str = "logs"
    reuse_results: bool = True
    interactive: bool = False
    privacy_aware_prompt: bool = False
    max_turns: int | None = None
    max_user_turns: int | None = None
    extra_args: Sequence[str] = field(default_factory=tuple)

    def to_cli_args(self) -> list[str]:
        args: list[str] = [
            "--model", self.model,
            "--llm_provider", self.llm_provider,
            "--task_category", self.task_category,
            "--org_type", self.org_type,
            "--task-limit", str(self.task_limit),
            "--agent_strategy", self.agent_strategy,
            "--agent_eval_mode", self.agent_eval_mode,
            "--log_dir", self.log_dir,
        ]
        if self.reuse_results:
            args.append("--reuse_results")
        if self.interactive:
            args.append("--interactive")
        args.extend(["--privacy_aware_prompt", "true" if self.privacy_aware_prompt else "false"])
        if self.max_turns is not None:
            args.extend(["--max_turns", str(self.max_turns)])
        if self.max_user_turns is not None:
            args.extend(["--max_user_turns", str(self.max_user_turns)])
        if self.extra_args:
            args.extend(self.extra_args)
        return args


def run_configurations(
    configs: Iterable[RunConfig],
    *,
    python_executable: str | None = None,
    run_tasks_path: Path | None = None,
    check: bool = True,
) -> None:
    python_cmd = python_executable or sys.executable
    script_path = str(run_tasks_path or RUN_TASKS_PATH)

    for cfg in configs:
        cmd = [python_cmd, script_path, *cfg.to_cli_args()]
        print(f"Running CRMArena with command: {' '.join(cmd)}")
        subprocess.run(cmd, check=check)


if __name__ == "__main__":
    demo_configs = [
        RunConfig(task_category="lead_qualification", task_limit=3),
        RunConfig(task_category="handle_time", task_limit=2, agent_strategy="tool_call", org_type="original"),
    ]
    run_configurations(demo_configs)
