# Background Agents Playbook

This folder defines scopes, deliverables, and operating procedures for background agents executing research/planning tasks in parallel with core engineering.

## Agents and Ownership

- Agent 1 — Research & Planning Agent
  - Owns: Task 2, Task 5
  - Focus: TDA vectorization + learnable topology research, architecture, specs
- Agent 2 — Infrastructure Planning Agent
  - Owns: Task 6, Task 7
  - Focus: Domain research (finance, cybersecurity), data pipelines, modeling plans
- Agent 3 — Performance & Scaling Agent
  - Owns: Task 8, Task 9, Task 10
  - Focus: Perf/scale architecture, security hardening, UI/UX and Mapper plan

## Source of Truth for Tasks

- Tasks are defined in `.taskmaster/tasks/` and `tasks.json`.
- Agent plans must reference task IDs and respect dependencies.

## Working Agreements

- Cadence: Daily brief update (≤10 bullets) and a weekly synthesized brief.
- Artifacts: Each task must have an Architecture Decision Record (ADR), a measurable acceptance checklist, and risk log.
- Handoffs: Every plan includes a “Dev-Ready” section with interface contracts and test criteria.
- Traceability: Reference FR/ST IDs (e.g., FR-CORE-002, ST-301) where applicable.

## Files

- `agent_research_planning.md` — Agent 1 (Tasks 2, 5)
- `agent_infrastructure_planning.md` — Agent 2 (Tasks 6, 7)
- `agent_performance_scaling.md` — Agent 3 (Tasks 8, 9, 10)