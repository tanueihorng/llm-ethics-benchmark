# Extensibility

## 1. Add a New Model
### Interface Expectations
Model entries are declared in `configs/default.yaml` under `models` with:

- `hf_id`
- `trust_remote_code`
- `dtype`

### Steps
1. Add new alias and Hugging Face model ID to config.
2. Run CLI with `--model <alias>`.
3. Ensure model is compatible with `AutoModelForCausalLM` and tokenizer loading.

No code changes are required if the model follows standard causal-LM interfaces.

## 2. Add a New Dataset
### Expected Pattern
Each dataset module should provide:

- a sample dataclass
- a `load_*` function that returns a list of samples
- prompt construction aligned with evaluator expectations

### Steps
1. Create new loader file in `ethical_benchmark/datasets/`.
2. Add task configuration block in `configs/default.yaml`.
3. Wire loader selection in `load_samples` inside `ethical_benchmark/pipeline/run_benchmark.py`.

## 3. Add a New Metric
### Expected Pattern
Evaluator classes expose:

- `evaluate_batch(samples, responses) -> List[Dict]`
- `summarize(records) -> Dict`

### Steps
1. Extend evaluator summary output with new metric fields.
2. Keep objective and subjective metrics separated when relevant.
3. Metrics automatically propagate to summary JSON and flattened CSV exports.

## 4. Add a New Task
### Required Components
- Dataset loader in `datasets/`
- Evaluator in `evaluators/`
- Task block in config
- Task branching in pipeline (`load_samples`, `build_evaluator`)

### Design Guidance
Keep interfaces consistent with existing tasks to preserve:

- resume logic
- raw JSONL auditing
- summary export behavior
- reproducibility controls

## 5. Abstract Interface Summary
Practical framework contracts:

- **Dataset loader contract:** returns typed sample objects with stable `sample_id` and `prompt`.
- **Generator contract:** `generate_batch(prompts)` returns ordered responses.
- **Evaluator contract:** batch evaluation + aggregate summarization.
- **Metrics contract:** serializable dictionaries for JSON and flat CSV conversion.
