# Gemini Project Guide: TDA Platform

## 📋 **PRIMARY INSTRUCTIONS**
**IMPORTANT**: All core project rules are defined in `UNIFIED_AGENT_INSTRUCTIONS.md`. Read that file for:
- Validation-first development principles
- TDA implementation requirements  
- Data integrity protocols
- Project structure guidelines
- Methodology assessment protocols

This file contains Gemini-specific extensions and workflow details.

## 🚨 Primary Directive: Validation First

This project operates under a strict **VALIDATION-FIRST DEVELOPMENT PRINCIPLE**. My primary goal is to ensure every claim is backed by reproducible, evidence-based validation. Accuracy is more important than perceived progress.

1.  **No Unvalidated Claims**: I will not state any performance metric (e.g., accuracy, F1-score, speed) without citing a specific, deterministic validation script that proves it.
2.  **Evidence is Mandatory**: All performance-related changes must be accompanied by output from the validation script (e.g., metrics, confusion matrices).
3.  **Failure is a Finding**: I will document and quantify failed experiments in `METHODOLOGY_GRAVEYARD.md` or other appropriate logs. Negative results are valuable.
4.  **Trust Validation, Not Comments**: I will trust the output of validation scripts over any performance claims made in code comments or older documentation.

## 📂 Key Project Files

-   **`TDA_AUTHORITATIVE_STATUS.md`**: This is the single source of truth for project status, priorities, and progress. I will read this at the start of every session and update it upon completing any task.
-   **`claude.md`**: The origin of the project's strict validation and methodology protocols. I will refer to it for detailed process guidelines.
-   **`README.md`**: Contains the public-facing project overview, strategic goals, and setup instructions.
-   **`validation/`**: The directory containing all validation scripts and results. This is where I will look for or create proof for any performance claims.

## ⚙️ Development Workflow & Commands

My core workflow will be:

1.  **Synchronize with Project Status**: Read `TDA_AUTHORITATIVE_STATUS.md` to determine the current priority.
2.  **Implement Changes**: Write or modify code to address the task.
3.  **Validate Rigorously**: If the task involves performance, I will:
    a.  Locate or create a validation script in `validation/`.
    b.  Ensure the script is deterministic (i.e., uses a fixed `random_state`).
    c.  Run the script and record the exact results.
4.  **Verify Quality**: Run the full test and quality suite.
5.  **Commit with Evidence**: Commit the changes with a clear message that includes the validated performance and the script used to prove it.
6.  **Update Status**: Update `TDA_AUTHORITATIVE_STATUS.md` to reflect the completed work.

---

### Essential Commands

-   **Activate Environment**: `source .venv/bin/activate`
-   **Install Dependencies**: `pip install -r requirements.txt`
-   **Install in Dev Mode**: `pip install -e .`
-   **Run All Tests**: `pytest tests/`
-   **Run Tests with Coverage**: `pytest --cov=src tests/`
-   **Format Code (Black)**: `black src/ tests/ examples/`

## 📝 Reporting Format

When reporting on my work, I will adhere to the following format:

-   **For successful, validated changes:**
    > "Implemented XYZ. Achieved **75.3% F1-score** (Validated by: `validation/validate_xyz.py` with seed=42)."

-   **For failed experiments:**
    > "Attempted ABC. Resulted in **48.1% F1-score**, a 22.5% regression from the baseline. Abandoning approach. See `METHODOLOGY_GRAVEYARD.md` for details."

## 🎯 Strategic Context

-   **Goal**: Build a TDA platform for Cybersecurity (APT detection, IoT) and Financial Risk (bubble detection, risk assessment).
-   **Core Advantage**: Mathematical interpretability and superior performance in high-dimensional spaces, which are key for regulatory compliance and complex pattern detection.
-   **Tech Stack**: Python, `scikit-tda`, `gudhi`, `ripser`, `scikit-learn`, `pytorch`.
