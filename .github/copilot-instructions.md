---
# TDA Project Rules
Configuration for AI behavior in Topological Data Analysis project

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

## PRE-IMPLEMENTATION VERIFICATION
Before writing ANY TDA validation code, must provide:
1. Exact import statements from existing TDA infrastructure
2. Specific methods that will be called (e.g., `.fit()`, `.transform()`)
3. What actual topological features will be extracted
4. Proof that data split prevents temporal leakage

## CRITICAL ACCURACY AND VALIDATION RULES

### VALIDATION-FIRST DEVELOPMENT PRINCIPLE
- **ACCURACY > PROGRESS**: Accurate reporting is INFINITELY more valuable than artificial progress claims
- **FAILURE IS PROGRESS**: Finding methods that don't work is as valuable as finding ones that do
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

## After ANY successful `edit_file` or `reapply` operation
- YOU MUST IMMEDIATELY run the `codacy_cli_analyze` tool from Codacy's MCP Server for each file that was edited, with:
 - `rootPath`: set to the workspace path
 - `file`: set to the path of the edited file
 - `tool`: leave empty or unset
- If any issues are found in the new edits, propose and apply fixes for them.
- > NOTE: Failure to follow this rule is considered a critical error.

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