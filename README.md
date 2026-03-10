*This project has been created as part of the 42 curriculum by rpetit.*

# Call Me Maybe

## Description

Call Me Maybe is a small command-line tool that demonstrates constrained decoding to map natural-language user prompts to pre-defined function calls. The program encodes available function names, queries a small local language model for next-token logits, and incrementally decodes the function name and its parameters (numbers or strings). The goal is to reliably identify which function should be called and extract typed parameters from an arbitrary user query.

The project is modular and uses pydantic models to validate inputs, an abstracted `Small_LLM_Model` (in `llm_sdk`) for encoding/decoding and logits access, and helper utilities for logging and progress reporting.

## Repository layout (relevant files)

- `src/` — main application code
  - `CallMeMaybe.py` — core constrained-decoding logic and orchestration
  - `FunctionDefinitions.py` — load and validate functions definition JSON
  - `CallingTests.py` — run batch tests from JSON
  - `OutputFile.py` — write prompts + extracted calls to output JSON
  - `Vocabulary.py` — map numeric tokens in the model vocabulary
  - `ArgsParser.py` — CLI argument definitions
  - `utils/` — utility modules
    - `Logger.py` — Simple logging utility with color support
    - `Color.py` — ANSI color codes for terminal output
    - `ProgressBar.py` — Simple progress bar
- `data/` — data files
  - `input/` — input JSON files for function definitions and test prompts
    - `functions_definition.json` — defines available functions, their parameters, and types
    - `function_calling_tests.json` — example prompts to test the program
  - `output/` — output JSON files with results
- `llm_sdk/` — local model wrapper (abstracted `Small_LLM_Model`)

## Instructions

Prerequisites
- Python 3.10+
- `uv` tool used by the included Makefile (see `Makefile`)
- The project expects a small local model that implements the `Small_LLM_Model` interface found in `llm_sdk`.

Common commands
Install dependencies:
```bash
make install
# or
uv sync
```

Run all built-in tests and save results to the default output file:
```bash
make run
```

Get help on CLI arguments and usage:
```bash
uv run python3 -m src --help
```

Run interactively and enter a single prompt:
```bash
uv run python3 -m src --interactive # or -I
```

Specify custom files:
```bash
uv run python3 -m src -i ./data/input/function_calling_tests.json -f ./data/input/functions_definition.json -o ./data/output/prompts_output.json
```

Lint and type-check:
```bash
make lint
# or 
make lint-strict
```

Configuration files
- `--input / -i` — path to JSON file with prompts to process (default `./data/input/function_calling_tests.json`)
- `--output / -o` — path to output JSON file (default `./data/output/prompts_output.json`)
- `--functions_definition / -f` — path to functions definition (default `./data/input/functions_definition.json`)

## Algorithm explanation (constrained decoding)

The constrained decoding approach implemented in `src/CallMeMaybe.py` works as follows:

1. Encoding available functions: all function names are encoded with the model tokenizer to obtain token-id sequences for each name (`FunctionDefinitions.get_names_inputs`).
2. Preprompt construction: build a preprompt that lists available functions and the user query; encode the preprompt and the user prompt.
3. Iterative name decoding: starting from the preprompt token ids, repeatedly query the model for next-token logits and restrict allowed next tokens to those that continue any function name (by comparing candidate token ids to the encoded function name prefixes). The decoder picks the highest-logit token that matches a valid continuation.
4. Confidence checks: the implementation inspects logits and compares them against a minimum confidence threshold (`MIN_CONFIDENCE_THRESHOLD`) in early iterations to avoid spurious matches; if confidence is too low, an error is raised.
5. Parameter extraction: once a function name is selected, parameters are extracted by appending parameter-specific prompts and decoding token-by-token. Numeric parameters are recognized using a vocabulary mapping of numeric tokens; string parameters rely on tokenization and terminator tokens to delimit strings.
6. Termination and output: once all parameters are found, the function call and parameter values are saved to the configured output JSON file.

Key implementation details:
- The model must expose tokenization (`encode`, `decode`) and raw logits for input token ids (`get_logits_from_input_ids`). This is provided by the `Small_LLM_Model` abstraction in `llm_sdk`.
- Vocabulary-based numeric detection is used to map model token ids to numeric values instead of relying purely on text parsing.
- The decoder limits the number of logits checked (configurable constants) to balance speed and robustness.

## Design decisions

- Modularity: each concern (parsing JSON, function definitions, vocabulary, output, progress) is isolated into its own module to simplify testing and maintenance.
- Pydantic models: pydantic is used for data validation and clear model definitions for functions, parameters, and runtime objects.
- Deterministic constrained decoding: instead of sampling, the tool greedily chooses the top logits constrained by valid function-name continuations to reduce hallucination when selecting a function.
- Vocabulary-assisted numeric parsing: numeric tokens are detected by a reverse mapping (`Vocabulary`) so numbers are extracted more reliably from token ids.
- Simple CLI: `ArgsParser` provides predictable flags and defaults so the tool is usable out of the box with the provided JSON fixtures.

## Performance analysis

- Accuracy: accuracy depends heavily on the underlying model quality and the quality of the function definitions and prompts. The constrained decoding approach reduces incorrect function selections by enforcing valid continuations, but ambiguous or out-of-vocabulary phrasing can still cause errors.
- Speed: querying logits iteratively is more expensive than a single forward pass. Performance is bounded by the model's `get_logits_from_input_ids` implementation and the number of decoding iterations (proportional to the longest function name and the number of parameter tokens). The code uses numpy to sort logits which is efficient for moderate vocabulary sizes.
- Reliability: the confidence threshold and limited logits checks help avoid spurious selections, but the current thresholds are heuristic and may need tuning for different models and tokenizers.

## Challenges faced

- Tokenization mismatch: function and parameter boundaries depend on the tokenizer; ensuring consistent token-level comparisons (prefix matching) was essential and required careful handling of encoded ids.
- Numeric detection: numbers can be split across tokens; building a reverse numeric vocabulary and mapping token ids back to numbers reduced parsing errors.
- Ambiguous or short function names: short or similar function names require extra care in decoding to avoid premature termination or misselection; this is mitigated by checking available continuations and confidence thresholds.
- Input robustness: parsing different JSON shapes and handling empty files required explicit error handling (see `JsonParser`, `OutputFile`, `FunctionDefinitions`).

## Testing strategy

- Automated tests: use `data/input/function_calling_tests.json` to provide a suite of prompts exercised by `src/CallingTests.py`. Running the program with the default input file will process all tests and write results to `data/output/prompts_output.json`.
- Validation: `FunctionDefinitions` uses pydantic validation to ensure function definitions are well-formed. Malformed definitions raise descriptive errors.
- Manual interactive tests: run `uv run python3 -m src --interactive` to try ad-hoc queries and inspect which function and parameters are returned.
- Additional unit tests (recommended): add unit tests for tokenizer behavior, `Vocabulary` numeric mapping, and the constrained token-selection logic.

## Example usage

Run all built-in tests and save results to the default output file:
```bash
make run
```

Run interactively and enter a single prompt:
```bash
uv run python3 -m src --interactive # or -I
```

Specify custom files:
```bash
uv run python3 -m src -i ./data/input/function_calling_tests.json -f ./data/input/functions_definition.json -o ./data/output/prompts_output.json
```

Lint and type-check:
```bash
make lint
# or 
make lint-strict
```

## Resources

- pydantic: https://docs.pydantic.dev/ — used for data modeling and validation
- numpy: https://numpy.org/ — used for efficient logits sorting and array operations
- Tokenization and decoding references: model-specific tokenizer docs (see your local `llm_sdk` for specifics)
- Articles on constrained decoding and guided decoding:
  - standard literature on prefix-constrained decoding and beam search variants

> Ai was use to create a draft of this README and for some question on libs (e.g. numpy and pydantic) 
