#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

TASKS_JSON_PATH = Path(".taskmaster/tasks/tasks.json")
AGENTS_JSON_PATH = Path(".taskmaster/agents/agents.json")
REPORTS_DIR = Path(".taskmaster/reports/agents")


@dataclass
class Task:
    id: int
    title: str
    description: str
    details: str
    test_strategy: str
    priority: Optional[str]
    dependencies: List[int]
    status: str


@dataclass
class Agent:
    id: str
    name: str
    why: str
    assigned_task_ids: List[int]


def load_tasks() -> Dict[int, Task]:
    if not TASKS_JSON_PATH.exists():
        raise FileNotFoundError(f"Missing tasks file: {TASKS_JSON_PATH}")
    with TASKS_JSON_PATH.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    tasks_raw = raw.get("master", {}).get("tasks", [])
    tasks: Dict[int, Task] = {}
    for t in tasks_raw:
        try:
            tasks[t["id"]] = Task(
                id=t["id"],
                title=t.get("title", ""),
                description=t.get("description", ""),
                details=t.get("details", ""),
                test_strategy=t.get("testStrategy", ""),
                priority=t.get("priority"),
                dependencies=[int(d) for d in t.get("dependencies", [])],
                status=t.get("status", "todo"),
            )
        except Exception as e:
            print(f"Warning: failed to parse task entry {t}: {e}", file=sys.stderr)
    return tasks


def load_agents() -> List[Agent]:
    if not AGENTS_JSON_PATH.exists():
        raise FileNotFoundError(f"Missing agents file: {AGENTS_JSON_PATH}")
    with AGENTS_JSON_PATH.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    agents: List[Agent] = []
    for a in raw.get("agents", []):
        agents.append(
            Agent(
                id=a.get("id"),
                name=a.get("name"),
                why=a.get("why", ""),
                assigned_task_ids=[int(x) for x in a.get("assignedTaskIds", [])],
            )
        )
    return agents


def safe_slug(text: str) -> str:
    return "".join(c.lower() if c.isalnum() else "-" for c in text).strip("-").replace("--", "-")


def render_plan(agent: Agent, task: Task, all_tasks: Dict[int, Task]) -> str:
    dep_titles = []
    for dep_id in task.dependencies:
        dep_task = all_tasks.get(dep_id)
        if dep_task:
            dep_titles.append(f"{dep_id}: {dep_task.title}")
        else:
            dep_titles.append(str(dep_id))

    rationale = agent.why.strip() if agent.why else ""

    # Tailored focus notes per task domain
    domain_guidance: Dict[int, str] = {
        2: (
            "Focus: deterministic vectorization (landscapes, images, Betti curves); schema design in PostgreSQL/MongoDB; indexing & versioning; similarity search primitives.\n"
            "Key decisions: feature parametrization grids, normalization, diagram persistence format, DB indices (GIN/GiST), cosine vs. Wasserstein proxy."
        ),
        5: (
            "Focus: PyTorch layers over PDs (attention, hierarchical, TDA-GNN); autograd compatibility; batching; numerical stability.\n"
            "Key decisions: differentiable diagram encodings, memory layout, PyTorch Geometric integration, benchmarking datasets."
        ),
        6: (
            "Focus: data connectors (equities/crypto), sliding-window embeddings, unsupervised transition detection, stability metric, EWS.\n"
            "Key decisions: window sizes, normalization, distance metrics on PDs/landscapes, thresholding logic, backtesting protocol."
        ),
        7: (
            "Focus: stream ingestion, windowing, topological signatures, baseline modeling, anomaly thresholds, attack classification.\n"
            "Key decisions: Kafka/Flink integration contracts, feature latency budget, labeling strategy, evaluation metrics."
        ),
        8: (
            "Focus: distributed refactors (Spark/Flink), CUDA hotspots, K8s autoscaling, streaming throughput tuning, batch >20GB.\n"
            "Key decisions: partitioning strategies, kernel tiling, HPA signals, serialization formats (Avro/Protobuf)."
        ),
        9: (
            "Focus: authN (MFA), RBAC, audit trail, encryption in motion/at rest.\n"
            "Key decisions: provider selection (OIDC), scopes/roles, immutable logs, KMS configs."
        ),
        10: (
            "Focus: Mapper algorithm, UI polish, accessibility.\n"
            "Key decisions: filter functions, clustering params, visualization interactions, WCAG checks."
        ),
    }

    guidance = domain_guidance.get(task.id, "")

    plan = []
    plan.append(f"# Task {task.id}: {task.title}")
    plan.append("")
    plan.append(f"Agent: {agent.name} ({agent.id})")
    if rationale:
        plan.append(f"Why: {rationale}")
    plan.append("")
    plan.append("## Summary")
    plan.append(task.description.strip())
    plan.append("")
    if guidance:
        plan.append("## Focus & Key Decisions")
        plan.append(guidance)
        plan.append("")
    plan.append("## Dependencies")
    plan.append("- " + ("\n- ".join(dep_titles) if dep_titles else "None"))
    plan.append("")
    plan.append("## Deliverables")
    plan.append("- Architecture overview and data flow")
    plan.append("- Detailed spec and acceptance tests derived from 'Test Strategy'")
    plan.append("- Milestone plan with risks and mitigations")
    plan.append("- Open questions and research notes")
    plan.append("")
    plan.append("## Details")
    plan.append(task.details.strip())
    plan.append("")
    if task.test_strategy:
        plan.append("## Test Strategy")
        plan.append(task.test_strategy.strip())
        plan.append("")
    plan.append("## Milestones")
    plan.append("1. Requirements refinement and literature scan")
    plan.append("2. Prototype design and validation criteria")
    plan.append("3. Implementation plan (phased)")
    plan.append("4. Benchmarks and evaluation")
    plan.append("5. Handover/Integration steps")
    plan.append("")
    plan.append("## Open Questions")
    plan.append("- [ ] TBD")
    plan.append("")
    plan.append("## References")
    plan.append("- See `.taskmaster/docs/prd.txt` and templates for guidance.")
    return "\n".join(plan)


def write_plan(agent: Agent, task: Task, all_tasks: Dict[int, Task]) -> Path:
    agent_dir = REPORTS_DIR / f"{agent.id}_{safe_slug(agent.name)}" / f"task_{task.id:03d}_{safe_slug(task.title)}"
    agent_dir.mkdir(parents=True, exist_ok=True)
    plan_path = agent_dir / "plan.md"
    plan_content = render_plan(agent, task, all_tasks)
    with plan_path.open("w", encoding="utf-8") as f:
        f.write(plan_content)
    return plan_path


def generate_all(agents: List[Agent], tasks: Dict[int, Task]) -> List[Path]:
    written: List[Path] = []
    for agent in agents:
        for tid in agent.assigned_task_ids:
            task = tasks.get(tid)
            if not task:
                print(f"Warning: Task {tid} not found for {agent.name}", file=sys.stderr)
                continue
            path = write_plan(agent, task, tasks)
            written.append(path)
    return written


def get_mtime_safe(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Taskmaster Agents Runner")
    parser.add_argument("--watch", action="store_true", help="Watch for changes and regenerate plans")
    parser.add_argument("--interval", type=int, default=15, help="Watch interval seconds")
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks()
    agents = load_agents()
    written = generate_all(agents, tasks)
    print(f"Generated {len(written)} planning docs under {REPORTS_DIR}")

    if not args.watch:
        return 0

    last_tasks_mtime = get_mtime_safe(TASKS_JSON_PATH)
    last_agents_mtime = get_mtime_safe(AGENTS_JSON_PATH)

    try:
        while True:
            time.sleep(args.interval)
            changed = False
            t_m = get_mtime_safe(TASKS_JSON_PATH)
            a_m = get_mtime_safe(AGENTS_JSON_PATH)
            if t_m != last_tasks_mtime or a_m != last_agents_mtime:
                try:
                    tasks = load_tasks()
                    agents = load_agents()
                    written = generate_all(agents, tasks)
                    print(f"Regenerated {len(written)} planning docs (changes detected)")
                    last_tasks_mtime = t_m
                    last_agents_mtime = a_m
                    changed = True
                except Exception as e:
                    print(f"Error during regeneration: {e}", file=sys.stderr)
            if not changed:
                print("No changes detected. Watching...")
    except KeyboardInterrupt:
        print("Exiting watcher.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
