---
description: AI rules derived by SpecStory from the project AI interaction history
globs: *
---

---
# TDA Project Rules
Configuration for AI behavior in Topological Data Analysis project

## 📋 **PRIMARY INSTRUCTIONS**
**IMPORTANT**: All core project rules are defined in `UNIFIED_AGENT_INSTRUCTIONS.md`. Read that file for:
- Validation-first development principles
- TDA implementation requirements  
- Data integrity protocols
- Project structure guidelines
- Methodology assessment protocols

This file contains Copilot-specific extensions and integrations.

## MANDATORY TDA IMPLEMENTATION RULES
- THIS IS A TOPOLOGICAL DATA ANALYSIS PROJECT. ALL ANALYSIS MUST USE ACTUAL TOPOLOGY.
- FORBIDDEN: Using statistical features (mean, std, skew, kurtosis) as "topological proxies" or "basic topological features"
- REQUIRED: Must use existing TDA infrastructure:
  - `src.core.persistent_homology.PersistentHomologyAnalyzer` 
  - `src.core.mapper.MapperAnalyzer`
  - Real persistence diagrams, birth/death times, Betti numbers
- FORBIDDEN: Creating custom "extract_basic_topo_features" or similar statistical proxy functions
- REQUIRED: Before any TDA implementation, must explicitly state which existing TDA classes will be used

## DATA LEAKAGE PREVENTION
- FORBIDDEN: Claiming "cross-temporal validation" without proving temporal integrity
- REQUIRED: Must verify actual temporal overlap between attack and benign samples
- FORBIDDEN: Using temporally separated data (e.g., Feb 14 benign vs Feb 28 attacks)
- REQUIRED: Show timestamp analysis proving co-occurring samples
- ALWAYS check for temporal and data leakage when performance is unexpectedly low, especially if the user expresses concerns about it.

## PRE-IMPLEMENTATION VERIFICATION
Before writing ANY TDA validation code, must provide:
1. Exact import statements from existing TDA infrastructure
2. Specific methods that will be called (e.g., `.fit()`, `.transform()`)
3. What actual topological features will be extracted
4. Proof that data split prevents temporal leakage

## CRITICAL ACCURACY AND VALIDATION RULES

### VALIDATION-FIRST DEVELOPMENT PRINCIPLE
- **ACCURACY > PROGRESS**: Accurate reporting is INFINITELY more valuable than artificial progress claims
- **FAILURE IS PROGRESS**: Finding methods that don't work is as valuable as finding ones do
- **VALIDATE IMMEDIATELY**: Every performance claim must be validated with independent reproduction script
- **NO CLAIMS WITHOUT PROOF**: Zero tolerance for unvalidated performance assertions

### MANDATORY VALIDATION PROTOCOL
Every performance claim must pass validation with exact reproduction script

### EVIDENCE-BASED REPORTING ONLY
- **Every metric**: Must include exact reproduction script path
- **Every claim**: Must show actual test output with confusion matrices
- **Every result**: Must be deterministic with fixed random seeds
- **Every documentation**: Must reference validation that confirms the claim

### COMPREHENSIVE FAILURE DOCUMENTATION
- **Report all failures**: Document what didn't work and why
- **Quantify failures**: Show exact performance gaps vs. expectations
- **Learn from failures**: Extract actionable insights for future development
- **Celebrate failures**: Failed experiments prevent wasted effort on bad approaches

### NEVER CREATE GENERIC OR SYNTHETIC VALIDATION TESTS
- **ONLY test SPECIFIC named methods**: hybrid_multiscale_graph_tda, implement_multiscale_tda, etc.
- **NO generic "detector" tests**: Do NOT create tests for generic "APTDetector" classes  
- **NO synthetic data fallbacks**: If real data fails, FIX the data loading, don't create fake data
- **NO "comprehensive" or "enhanced" invented methods**: Only test methods explicitly mentioned by user
- **FIND AND TEST EXISTING SCRIPTS**: Look for and run the actual TDA method scripts the user references
- **DO NOT INVENT NEW TEST APPROACHES**: Use exactly what the user asks for, nothing else

### WHEN USER SAYS "TEST METHOD X" - DO THIS:
1. Find the existing script for method X (e.g., `hybrid_multiscale_graph_tda.py`) 
2. Run that EXACT script on real data
3. Report the results from THAT script
4. DO NOT create a new "validation" wrapper
5. DO NOT create generic detector tests
6. DO NOT invent "comprehensive" approaches

### METHODOLOGY FAILURE ASSESSMENT PROTOCOL

#### BASELINE PERFORMANCE REQUIREMENTS
- Every new methodology MUST be compared against a simple baseline
- If new method performs worse than baseline, immediately flag as FAILURE
- Document what went wrong, don't try to fix complex failures
- Example baselines: random classifier, simple statistical methods, existing solutions

#### DEGRADATION DETECTION CRITERIA
- Performance drops >5% from baseline: ⚠️ **WARNING** - investigate immediately
- Performance drops >10% from baseline: ❌ **FAILURE** - stop development, analyze root cause
- Method produces nonsensical results: ❌ **CRITICAL FAILURE** - abandon approach
- Computational cost >5x baseline without performance gain: ⚠️ **EFFICIENCY WARNING**

#### HONEST FAILURE COMMUNICATION
- State failures clearly: "Method X failed because Y"
- Don't euphemize: avoid "needs optimization" when you mean "doesn't work"
- Quantify the failure: show actual vs expected performance numbers
- Explain impact: how does this affect project timeline/goals

#### GO/NO-GO DECISION FRAMEWORK
- After 3 failed improvement attempts: STOP and reassess fundamental approach
- If core methodology shows no promise after proper testing: PIVOT to alternatives
- Document decision rationale with data
- Update project priorities based on what actually works

---
# Codacy Rules
Configuration for AI behavior when interacting with Codacy's MCP Server

## CRITICAL: After ANY successful `edit_file` or `reapply` operation
- YOU MUST IMMEDIATELY run the `codacy_cli_analyze` tool from Codacy's MCP Server for each file that was edited, with:
 - `rootPath`: set to the workspace path
 - `file`: set to the path of the edited file
 - `tool`: leave empty or unset
- If any issues are found in the new edits, propose and apply fixes for them.
- > NOTE: Failure to follow this rule is considered a critical error.

## After every response
- If you made any file edits in this conversation, verify you ran `codacy_cli_analyze` tool from Codacy's MCP Server 

## When there are no Codacy MCP Server tools available, or the MCP Server is not reachable
- Suggest the user the following troubleshooting steps:
 - Try to reset the MCP on the extension
 - If the user is using VSCode, suggest them to review their Copilot > MCP settings in Github, under their organization or personal account. Refer them to Settings > Copilot > Enable MCP servers in Copilot. Suggested URL (https://github.com/settings/copilot/features) or https://github.com/organizations/{organization-name}/settings/copilot/features (This can only be done by their organization admins / owners)
- If none of the above steps work, suggest the user to contact Codacy support

## Trying to call a tool that needs a rootPath as a parameter
- Always use the standard, non-URL-encoded file system path

## CRITICAL: Dependencies and Security Checks
- IMMEDIATELY after ANY of these actions:
 - Running npm/yarn/pnpm install
 - Adding dependencies to package.json
 - Adding requirements to requirements.txt
 - Adding dependencies to pom.xml
 - Adding dependencies to build.gradle
 - Any other package manager operations
- You MUST run the `codacy_cli_analyze` tool with:
 - `rootPath`: set to the workspace path
 - `tool`: set to "trivy"
 - `file`: leave empty or unset
- If any vulnerabilities are found because of the newly added packages:
 - Stop all other operations
 - Propose and apply fixes for the security issues
 - Only continue with the original task after security issues are resolved
- EXAMPLE:
 - After: npm install react-markdown
 - Do: Run codacy_cli_analyze with trivy
 - Before: Continuing with any other tasks

## General
- Repeat the relevant steps for each modified file.
- "Propose fixes" means to both suggest and, if possible, automatically apply the fixes.
- You MUST NOT wait for the user to ask for analysis or remind you to run the tool.
- Do not run `codacy_cli_analyze` looking for changes in duplicated code or code complexity metrics.
- Do not run `codacy_cli_analyze` looking for changes in code coverage.
- Do not try to manually install Codacy CLI using either brew, npm, npx, or any other package manager.
- If the Codacy CLI is not installed, just run the `codacy_cli_analyze` tool from Codacy's MCP Server.
- When calling `codacy_cli_analyze`, only send provider, organization and repository if the project is a git repository.

## Whenever a call to a Codacy tool that uses `repository` or `organization` as a parameter returns a 404 error
- Offer to run the `codacy_setup_repository` tool to add the repository to Codacy
- If the user accepts, run the `codacy_setup_repository` tool
- Do not ever try to run the `codacy_setup_repository` tool on your own
- After setup, immediately retry the action that failed (only retry once)
---

## DIAGNOSTIC AND TUNING RULES
- After running a machine learning experiment, ALWAYS summarize the AUC values (PR AUC and ROC AUC) and other relevant metrics in the response.
- If AUC values are low (ROC AUC < 0.7), recommend a step-back action to revisit the TDA representation pipeline or check for data leakage before further tuning.
- When the user requests a "fast pass", prioritize quick incremental suggestions like near-zero variance thresholding or feature scaling before more complex techniques.
- When comparing current progress with aspirational goals, focus on representation and data validation before solver tuning, especially if ROC AUC is significantly below target. If ROC AUC stays below 0.75 after 2-3 representation iterations, consider pivoting or re-extracting data.

## WORKFLOW & RELEASE RULES
- **AI Execution Mode**: The AI should generate executable code blocks that are run via a continue/cancel button setup. The AI must monitor the terminal output and respond to failures and errors with appropriate fixes.
- You MUST remember to activate the venv when you open a new terminal
- The AI must pause, review what we have done so far and make a markdown file explaining our reasoning and decisions. Datestamp it.
- The AI must run all commands and monitor terminals for output and errors at all time. The AI should respond to failures and errors with appropriate fixes and then run and monitor those.
- It is compulsory that you are the one to start all terminals so that you can monitor and respond as appropriate. I do not need to be copying or pasting anything at any point. You are forbidden from presenting code in any other way.
- You are forbidden from presenting code in any other way than executable code blocks.

## VIRTUAL ENVIRONMENT
- You MUST remember to activate the venv when you open a new terminal
- Python terminals need to have the venv activated with `source .venv/bin/activate` any time you open a fresh terminal
- The AI must activate the venv using `source .venv/bin/activate` before running any python commands.
- The AI should not hunt around for the venv activation command; it's always `source .venv/bin/activate`.

## EXTERNAL REPOSITORY ASSESSMENT RULES
When assessing external repositories:
- Summarize each proposed enhancement.
- Map proposal to actual external repo capabilities present.
- Evaluate technical feasibility within the current vector stack architecture.
- Identify data/infra prerequisites and missing components.
- Quantify realistic benefit vs claimed benefit (evidence-based risk).
- Prioritize with go/defer/reject recommendations and rationale.
- Provide a risk matrix (leakage, complexity, compute cost, validation burden).
- Suggest minimal safe experimental path (validation-first) for any "Go" items.

## NEW EXTERNAL REPOSITORY ASSESSMENT RULES
- When assessing external repositories, adhere to the following process:
  - Summarize each proposed enhancement.
  - Map proposal to actual external repo capabilities present.
  - Evaluate technical feasibility within the current vector stack architecture.
  - Identify data/infra prerequisites and missing components.
  - Quantify realistic benefit vs claimed benefit (evidence-based risk).
  - Prioritize with go/defer/reject recommendations and rationale.
  - Provide a risk matrix (leakage, complexity, compute cost, validation burden).
  - Suggest minimal safe experimental path (validation-first) for any "Go" items.
  - Forbid statistical features as topological proxies. Reject proposals using mean/variance counts.
  - Implement SW angle strategy flag & deterministic generators.
  - Implement sparse Rips parameter in diagram builder.
  - Hook DTM-Rips option (with fallback if Gudhi class missing).
  - Extend diagnostics.json with new topology config fields.
  - Add diagnostics fields: sw_angle_strategy, sw_angles_list, rips_mode, rips_sparse_param, compute_time_subsections (per block).
  - Generate a hash of angle set & filtration type in manifest_hash augmentation to prevent mixing outputs downstream.
  - Run baseline (current) -> metrics_baseline.json.
  - Run with enhancement -> metrics_enh.json.
  - Compute delta: ΔPR, ΔROC, ΔBrier, timing, feature_dim difference.
  - If ΔPR < +0.005 and compute cost > +25%, mark as FAIL (document in graveyard).
  - Persist comparative JSON: enhancement_eval_<timestamp>.json.

## FILTRATION MODE EXPERIMENTATION RULES
- When experimenting with filtration modes (DTM, Sparse, Standard):
  - Run a controlled experiment matrix with strict success criteria and early abort logic to avoid sunk cost. Either surface a measurable, justified improvement (with reproducibility artifacts) or confidently retire the avenue.
  - If point cloud coverage is incomplete, abort variant.
  - If gudhi is missing, record FAIL and skip.
  - If any degradation >0.01 PR AUC or leakage flag flips to true, CRITICAL FAIL.
  - Sparse mode success: reduce compute time OR dimension with ≤ negligible metric drop (<0.003 PR AUC). Can accept: slight metric loss but ≥25% faster or ≥20% smaller feature set.

## DIAGNOSTICS CAPTURE REQUIREMENTS
- The `filtration_recompute` section in diagnostics MUST include:
    - attempted_count
    - success_count
    - skipped_no_pointcloud
    - skipped_no_gudhi
    - failed_exception
    - mode
    - params: {dtm_k|sparse_param}
    - mean_points_in
    - mean_points_after_subsample
    - rips_recompute_time_sec
- Plus: sw_angle_strategy, sw_angles_list, sw_angles_hash, block_compute_time_sec, manifest_hash.

## ACTION THRESHOLDS FOR FILTRATION VARIANTS
- PASS (candidate): ΔPR_AUC ≥ +0.005 AND training time multiplier ≤ 1.25× baseline.
- MARGINAL: +0.003 ≤ ΔPR_AUC < +0.005 → need second confirmation run; otherwise discard.
- FAIL: ΔPR_AUC < +0.003 OR time cost >1.5× baseline, or feature_dim inflation >20% with no metric gain.

## GENERAL BEHAVIOR RULES
- When the user expresses frustration or questions the AI's memory, the AI should double-check its understanding of the project context and goals, and proactively retrieve relevant information from the repository before proceeding.
- The AI must remember to activate the venv when you open a new terminal.
- When a user provides a specific file path (e.g., `validation/vector_stack_outputs/20250810_031323`), the AI should prioritize using that path for subsequent operations and file access.

## COPILOT EXECUTION MODE TROUBLESHOOTING

If the AI is unable to execute code or monitor terminals, consider the following:

1.  **Execution Channel Disabled**:
    *   Check VS Code settings: Settings > Extensions > GitHub Copilot Chat: enable “Allow Command Execution”.
    *   Confirm experimental “AI Execution Mode” is ON in settings.json:

    ```json
    "copilot.chat.executeCommands": true,
    "copilot.chat.terminalControl": true
    ```

2.  **MCP / Tool Bridge Not Loaded**:
    *   Open Output: “GitHub Copilot” & “MCP Servers” – look for errors loading Codacy or command runner.
    *   If Codacy MCP is missing, re-enable MCP servers at <https://github.com/settings/copilot/features> (organization scope if needed), then reload the window.

3.  **Workspace Trust Reset**:
    *   Command: “Workspaces: Manage Workspace Trust” → mark the workspace as trusted. Without trust, command/terminal orchestration is blocked.

4.  **Terminal Session Mismatch**:
    *   The Copilot may be trying to reuse a disposed terminal. Close all extra terminals, then ask Copilot to “open a managed terminal” again.
    *   Confirm .venv exists:

    ```bash
    ls -1 .venv/bin/activate
    ```

5.  **Conflict with Another Extension**:
    *   Temporarily disable other AI/automation extensions (e.g., Code Runner) that might hijack command execution.

6.  **Chat Context Downgraded**:
    *   If you started a “plain chat” instead of the orchestration-enabled view, commands won’t execute. Use the Copilot “/terminal” or “/workspace” slash commands to see if they register. If they autocomplete only as plain text, you’re not in the execution context.

7.  **Missing Internal Capability Flag**:
    *   Run Developer: Toggle Developer Tools; check console for errors beginning with “[Copilot][orchestrator]”.

**Recovery Procedure (Systematic)**:

A.  Reload the window (Developer: Reload Window).
B.  Trust the workspace.
C.  Re-enable MCP servers & command execution settings.
D.  Close all terminals; create a new terminal manually; then request Copilot to run a trivial command:

```bash
echo COPILOT_TEST && python -V
```

E.  If still failing, sign out/in of GitHub in VS Code.

If the failure persists:

*   Capture logs: Output > GitHub Copilot (last 200 lines).
*   Provide any error lines starting with “ERR” to proceed with deeper remediation (e.g., auth scope, rate limit).