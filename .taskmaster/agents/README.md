# Taskmaster Background Agents

This directory defines background agents and their task assignments.

- Config: `.taskmaster/agents/agents.json`
- Runner: `.taskmaster/scripts/agent_runner.py`
- Outputs: `.taskmaster/reports/agents/<agent_id>_<name>/task_<id>_/plan.md`

## Start agents

```bash
python3 .taskmaster/scripts/agent_runner.py --watch --interval 15
```

Run without `--watch` to generate plans once.