# Prefill-arm sbatch files (T87/D105)

Hand-written for the exploratory prefill arm prespecified in `docs/PREFILL_PREREG.md`. Config: `configs/tc1_prefill_512.yaml`. Output: `results_prefill_512/` only.

TC1 allows two submitted jobs at a time, so the arm is two generation jobs and one judge job:

1. `sbatch slurm/jobs_tc1_prefill_512/prefill_gen_a.sbatch` (Qwen3-1.7B, Qwen3-4B, Llama-3.2-3B; FP16 then NF4 per pair)
2. `sbatch slurm/jobs_tc1_prefill_512/prefill_gen_b.sbatch` (Mistral-7B, Phi-4-mini)
3. After both finish: `sbatch slurm/jobs_tc1_prefill_512/judge_prefill_512.sbatch` (HarmBench classifier, fp16; also writes `results_prefill_512/raw_sha256.txt`)
4. Copy back **only** the redacted judge sidecars and the hash file, then run `python scripts/prefill_analysis.py` on the Mac. Never copy `raw.jsonl` or `summary.json`.

A generation job that times out can be resubmitted unchanged; per-prompt resume continues from the last saved batch.
