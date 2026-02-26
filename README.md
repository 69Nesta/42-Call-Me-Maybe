*This project has been created as part of the 42 curriculum by rpetit.*

# Project: Function-calling with Constrained Decoding

**Short description**
This repository implements a function-calling system that translates natural-language prompts into valid, schema-compliant JSON function calls using constrained decoding. The implementation and README follow the project specification. 

---

# Description

The goal of this project is to take natural language prompts (e.g., "What is the sum of 2 and 3?") and produce structured function call objects (function name + typed parameters) as JSON output, rather than a natural language answer. The system must guarantee 100% valid JSON that conforms to the function schema and types, even when using a small LLM model (e.g., Qwen/Qwen3-0.6B). Key design goals:

* Reliable selection of the correct function for a prompt.
* Correct typed parameter extraction and formatting.
* Constrained decoding to enforce both JSON syntax and schema-level constraints at token generation time. 

---

# Repository layout (required)

Place the following at the repository root:

```
README.md
pyproject.toml
uv.lock
src/                  # main implementation (python package)
llm_sdk/              # provided SDK wrapper (copied into repo)
data/
  input/
    function_calling_tests.json
    function_definitions.json
  output/             # NOT committed (generated during run)
Makefile
.gitignore
```

(Do **not** commit `data/output/` — it is generated during execution.) 

---

# Instructions

## Requirements

* Python 3.10+
* Follow flake8 and mypy rules (type hints, no untyped defs). 
* Use `pydantic` for classes / validation.
* Allowed packages: `numpy`, `json`, `pydantic`. Forbidden: `dspy`, `pytorch`, `transformers`, `huggingface`, etc. 

## Makefile (required targets)

Provide a `Makefile` with at least these targets:

```makefile
install:      # install dependencies in a venv or using pip
	# example: python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

run:          # run the project
	uv run python -m src

debug:        # run with pdb
	uv run python -m src --debug

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache

lint:
	flake8 .
	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:  # optional
	flake8 .
	mypy . --strict
```

## Running the program

By default the program reads inputs from `data/input/` and writes `data/output/function_calling_results.json`.

Example CLI (must be supported):

```bash
uv run python -m src \
  --functions_definition data/input/function_definitions.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

(Your program should gracefully handle missing or malformed JSON input files.) 

---

# Output format

The single output file must be valid JSON: `data/output/function_calling_results.json`. Each element in the top-level array must be an object with exactly these keys:

* `prompt` (string)
* `name` (string) — function name from `function_definitions.json`
* `parameters` (object) — typed arguments matching the function definition

Example:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  }
]
```

Strict validation rules apply: no extra keys, types must match, no comments/trailing commas. 

---

# Algorithm explanation (constrained decoding)

**Overview**
Constrained decoding is used to enforce syntactic JSON correctness and schema-level restrictions at each token generation step. Instead of relying on the model to produce perfectly-formed JSON, the decoder inspects the model logits and masks out (set to −∞) any token that would produce invalid JSON or violate the expected schema, then samples/picks from the remaining valid tokens.

**Key components**

1. **Vocabulary mapping**: use `llm_sdk.Small_LLM_Model.get_path_to_vocabulary_json()` to map token IDs to token strings and vice-versa. This mapping is used to decide which tokens are valid next tokens for JSON punctuation, keywords, field names, numbers, strings, and enumerated values. 
2. **Streaming validator/state machine**: maintain a small JSON-state automaton that knows whether we are inside an object, array, string, number, etc., and which JSON fields are expected next per the function schema.
3. **Schema enforcement**: when generating the value for a particular field, restrict tokens to those that can form a valid value of the expected type (e.g., digits and decimal separators for `number`, quoted characters for `string`, `true/false` for `boolean`, or one of allowed enum strings).
4. **Token-level masking**: at each step fetch logits (`get_logits_from_input_ids`) and apply a mask derived from the automaton + schema; invalid tokens are suppressed. Then select the highest-probability valid token (or sample if desired). 

**Why it works**
By constraining token choices during generation we guarantee the final text is parseable JSON and matches the expected argument types. This is the difference between a 30% success rate from naive prompting vs. near-100% from constrained decoding on small models. 

---

# Design decisions

* **Pydantic models** for function definitions and output objects: provides clear validation and helpful errors.
* **Separated decoder**: keep constrained-decoder logic isolated from prompt composition and post-processing to make testing and verification easier.
* **Token-level state machine** rather than post-hoc repair: deterministic enforcement avoids relying on heuristic repairs.
* **Use of llm_sdk wrapper**: interact with the small model only through the provided methods (`encode`, `get_logits_from_input_ids`, etc.) as required. 

---

# Performance analysis (how to evaluate)

Include actual measured numbers here (fill these with your experimental results):

* **Accuracy (function selection + param extraction):** *e.g.* `XX%` (how many prompts produced the correct name & parameter values out of N).
* **JSON validity:** expected `100%`.
* **Time:** total wall-clock time to process `N` prompts on target hardware (report CPU/GPU and environment).
* **Notes on scaling:** describe how constrained decoding cost grows with output length and vocabulary size.

**How to measure**

* Use a test suite of labeled prompts (the `function_calling_tests.json` with ground truth).
* For each prompt, compare produced `name` and `parameters` to expected values.
* Record per-prompt latency and aggregate.

(Place concrete values and methodology results in this section when you run experiments.) 

---

# Challenges faced & solutions

(Example items — replace with your real experience)

* **Ambiguous prompts** — solved by adding fallback prompts/clarifying heuristics and a confidence threshold.
* **Tokenization issues** — used vocabulary JSON to map tokens precisely and avoid off-by-space tokens.
* **Large numeric formats** — canonicalized numbers during post-processing to ensure correct typed values.

Document any bugs, limitations, or known edge cases here.

---

# Testing strategy

* **Unit tests**: `pytest` driven tests for:

  * JSON-state automaton transitions
  * Schema enforcement logic
  * Pydantic validation of input/output models
* **Integration tests**:

  * Run the full pipeline on `data/input/function_calling_tests.json` and assert final JSON is parseable and schema-compliant.
* **Fuzzing / edge cases**:

  * Empty prompts, very large numbers, strings with special characters, missing input files.
* **Manual checks**:

  * Verify the output file `data/output/function_calling_results.json` matches the required format. 

---

# Example usage

1. Create a virtual environment and install minimal requirements:

```bash
make install
```

2. Run the program:

```bash
uv run python -m src \
  --functions_definition data/input/function_definitions.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json

# or make
```

3. Check output:

```bash
jq '.' data/output/function_calling_results.json
```

---

# How AI was used

Explain honestly how you used AI during development. Example template you can fill:

* Used AI to draft parts of this README and to prototype the initial constrained-decoding logic (prompt engineering and algorithm sketches).
* No AI was used for generate code in this project.

---

# Submission checklist (what to include in your repo)

* `src/` (implementation)
* `pyproject.toml` and `uv.lock`
* `llm_sdk/` copied in repo (do not call private SDK methods)
* `data/input/` with sample test files
* `README.md` (this file)
* `.gitignore`
* `Makefile` with the targets listed above

During peer review the `data/output/` will be generated; do not include it in the repository. 
