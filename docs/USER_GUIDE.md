# User Guide: Ethical Benchmarking Framework

**Welcome!** This guide will walk you through everything you need to know about using this framework for your Final Year Project (FYP). No advanced technical knowledge required!

---

## 📚 Table of Contents

1. [What Is This Project?](#what-is-this-project)
2. [Project Structure Explained](#project-structure-explained)
3. [Getting Started](#getting-started)
4. [Running Your First Benchmark](#running-your-first-benchmark)
5. [Understanding the Results](#understanding-the-results)
6. [Comparing Different Models](#comparing-different-models)
7. [Customizing the Benchmarks](#customizing-the-benchmarks)
8. [Troubleshooting](#troubleshooting)
9. [FAQ](#faq)

---

## What Is This Project?

### The Big Picture

Imagine you're testing how "ethical" different AI language models are. This framework helps you measure three important aspects:

1. **Toxicity** - Does the model generate harmful or offensive content?
2. **Social Bias** - Does the model show unfair bias against certain groups of people?
3. **Factuality** - Does the model tell the truth or spread misinformation?

### Why Sub-10B Models?

We focus on smaller models (under 10 billion parameters) because:
- They can run on regular computers (no need for expensive supercomputers!)
- They're more realistic for student projects and research labs
- They represent what many real-world applications actually use

---

## Project Structure Explained

Here's what each folder and file does. Think of it like organizing a kitchen:

### 📁 Top-Level Directories

```
ethical_benchmark/          ← The main "recipe book" (your code)
├── datasets/              ← Where we get test questions from
├── evaluators/            ← How we grade the AI's answers
├── metrics/               ← Math for calculating scores
├── models/                ← Code to load and run AI models
└── pipeline/              ← The main workflow coordinator

configs/                   ← Settings and preferences
└── default.yaml          ← Main configuration file (like a control panel)

docs/                      ← Documentation (guides like this one)
├── USER_GUIDE.md         ← You are here!
├── methodology.md        ← How the testing works
├── datasets.md           ← About the test questions
└── evaluation_metrics.md ← About the scoring system

results/                   ← Where all your results are saved
├── raw/                  ← Detailed record of every test
└── summary/              ← Summary tables and charts

tests/                     ← Automated checks to ensure everything works
```

### 📄 Important Individual Files

| File | What It Does | Do I Need to Touch It? |
|------|--------------|----------------------|
| `README.md` | Project overview and quick start | Read it first! |
| `requirements.txt` | List of needed software packages | No (used automatically) |
| `run_benchmark.py` | The main program you'll run | No (just run it!) |
| `configs/default.yaml` | All your settings in one place | **YES** - Customize here |
| `CITATION.bib` | How others cite your work | Update with your details |

---

## Getting Started

### Step 1: Check Your Computer

**Minimum Requirements:**
- A computer with at least 8GB RAM
- 20GB free disk space
- Internet connection (for first-time setup)

**Preferred:**
- 16GB+ RAM
- A NVIDIA GPU (optional, but makes things faster)

### Step 2: Install Python

You need Python 3.9 or newer. Check if you have it:

```bash
python --version
```

If you see something like `Python 3.11.5`, you're good! If not, download Python from [python.org](https://www.python.org/downloads/).

### Step 3: Set Up Your Workspace

Open your terminal/command prompt and navigate to the project folder:

```bash
cd /Users/tanueihorng/Documents/FYP
```

### Step 4: Create a Virtual Environment

This creates an isolated space for the project (like a separate room for your experiment):

```bash
python -m venv .venv
```

Activate it:
- **On Mac/Linux:** `source .venv/bin/activate`
- **On Windows:** `.venv\Scripts\activate`

You'll see `(.venv)` appear at the start of your command line. That means it's working!

### Step 5: Install Dependencies

This installs all the software packages the project needs:

```bash
pip install -r requirements.txt
```

⏱️ **This will take 5-15 minutes.** It's downloading and installing:
- PyTorch (for running AI models)
- Transformers (Hugging Face library)
- Datasets (for test questions)
- And several other tools

---

## Running Your First Benchmark

Let's test a small model on toxicity! This is the safest way to make sure everything works.

### Quick Test Run (5-10 minutes)

```bash
python run_benchmark.py \
  --model gemma-2b \
  --task toxicity \
  --max_samples 50 \
  --output_dir results
```

**What this command means:**
- `--model gemma-2b` → Use the Gemma 2B model (small and fast)
- `--task toxicity` → Test for toxic/harmful content
- `--max_samples 50` → Only test 50 examples (instead of the default 500)
- `--output_dir results` → Save results in the "results" folder

### What Happens During the Run?

1. **Downloading the model** (first time only, ~5GB)
   ```
   Loading model alias='gemma-2b' hf_id='google/gemma-2-2b-it'
   ```

2. **Loading test questions**
   ```
   Loaded 50 samples for task 'toxicity'
   ```

3. **Running the test** (you'll see a progress bar!)
   ```
   Generating [toxicity]: 100%|████████| 13/13 [00:45<00:00]
   ```

4. **Saving results**
   ```
   Raw records: results/raw/toxicity__gemma-2b.jsonl
   Summary JSON: results/summary/toxicity__gemma-2b.json
   ```

### Your First Results!

After the run completes, check:
- `results/summary/toxicity__gemma-2b.json` - Complete summary
- `results/summary/toxicity_summary.csv` - Spreadsheet-friendly format

---

## Understanding the Results

### The Summary JSON File

Open `results/summary/toxicity__gemma-2b.json`. Here's what you'll see:

```json
{
  "task": "toxicity",
  "model": {
    "alias": "gemma-2b",
    "hf_id": "google/gemma-2-2b-it"
  },
  "metrics": {
    "num_samples": 50,
    "mean_toxicity": 0.214,
    "pct_above_threshold": 18.4
  }
}
```

**What it means:**
- `mean_toxicity: 0.214` → Average toxicity score is 21.4% (lower is better!)
- `pct_above_threshold: 18.4` → 18.4% of responses were "toxic" (above 50% threshold)

### The Raw JSONL File

Open `results/raw/toxicity__gemma-2b.jsonl` in a text editor. Each line is one test:

```json
{"sample_id": "0", "prompt": "Hello", "response": "Hi there!", "toxicity_score": 0.05}
{"sample_id": "1", "prompt": "...", "response": "...", "toxicity_score": 0.89}
```

**Why this matters:**
- You can inspect every single response the model gave
- You can manually verify if the toxicity scores make sense
- Great for writing your FYP report with specific examples!

### The CSV Summary

Open `results/summary/toxicity_summary.csv` in Excel or Google Sheets.

Perfect for:
- Creating comparison tables for your report
- Making charts and graphs
- Sharing results with your supervisor

---

## Comparing Different Models

### Run Multiple Models

Test three different models to compare them:

```bash
# Small model (2B parameters)
python run_benchmark.py --model gemma-2b --task toxicity --max_samples 100

# Medium model (3B parameters)
python run_benchmark.py --model llama3.2-3b --task toxicity --max_samples 100

# Larger model (8B parameters)
python run_benchmark.py --model llama3-8b --task toxicity --max_samples 100
```

### The CSV Automatically Combines Results!

After running all three, open `results/summary/toxicity_summary.csv`:

| model_alias | num_records | mean_toxicity | pct_above_threshold |
|-------------|-------------|---------------|---------------------|
| gemma-2b    | 100         | 0.214         | 18.4               |
| llama3.2-3b | 100         | 0.198         | 15.2               |
| llama3-8b   | 100         | 0.187         | 12.8               |

**Interpretation:**
- Larger models tend to be safer (lower toxicity)
- You can now write: "Our results show that model size correlates with reduced toxic outputs..."

---

## Customizing the Benchmarks

### Change the Number of Test Samples

For **quick tests during development:**
```bash
--max_samples 50  # Takes ~5 minutes
```

For **your final FYP results:**
```bash
--max_samples 500  # Takes ~45 minutes, more reliable
```

For **thorough research:**
```bash
# Don't specify --max_samples (uses defaults: 500 for toxicity/bias, 300 for factuality)
```

### Run All Three Tasks

```bash
# Test toxicity
python run_benchmark.py --model gemma-2b --task toxicity

# Test social bias
python run_benchmark.py --model gemma-2b --task bias

# Test factuality
python run_benchmark.py --model gemma-2b --task factuality
```

### Edit the Configuration File

Open `configs/default.yaml` in a text editor. This is your control panel!

**Change the toxicity threshold:**
```yaml
tasks:
  toxicity:
    evaluation:
      threshold: 0.5  # Change to 0.3 for stricter, 0.7 for more lenient
```

**Change how many tokens the model generates:**
```yaml
decoding:
  max_new_tokens: 128  # Change to 64 for shorter responses, 256 for longer
```

**Change sampling behavior:**
```yaml
decoding:
  temperature: 0.0  # 0.0 = deterministic, 0.7 = more creative/random
```

### Add a New Model

1. Find the Hugging Face model ID (e.g., browsing [huggingface.co/models](https://huggingface.co/models))
2. Add it to `configs/default.yaml`:

```yaml
models:
  my-new-model:
    hf_id: organization/model-name
    trust_remote_code: false
    dtype: auto
```

3. Run it:
```bash
python run_benchmark.py --model my-new-model --task toxicity
```

---

## Troubleshooting

### Problem: "Out of Memory" Error

**Solution 1:** Use a smaller model
```bash
--model gemma-2b  # Instead of llama3-8b
```

**Solution 2:** Reduce batch size in `configs/default.yaml`:
```yaml
tasks:
  toxicity:
    batch_size: 2  # Change from 4 to 2
```

**Solution 3:** Force CPU mode (slower but uses less memory):
```bash
--device cpu
```

### Problem: Download Stuck or Very Slow

**Why:** Models are large (2-8GB). First download can take 20-60 minutes.

**Solution:** 
- Be patient, it only happens once
- Check your internet connection
- Models are cached, so subsequent runs are instant!

### Problem: "ModuleNotFoundError"

**Why:** Virtual environment not activated or packages not installed.

**Solution:**
```bash
# Activate virtual environment
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows

# Reinstall packages
pip install -r requirements.txt
```

### Problem: Results Look Strange

**Check these:**
1. Did the run complete successfully? (Check for error messages)
2. Are you using enough samples? (Try --max_samples 200 instead of 20)
3. Is the model downloading correctly? (Check ~/.cache/huggingface/)

### Problem: Tests Failing

Run the test suite to check your installation:
```bash
pytest tests/ -v
```

You should see: `95 passed` (or similar). If tests fail, your installation may have issues.

---

## FAQ

### How long does a full benchmark take?

**Quick estimate:**
- Toxicity (500 samples): 30-60 minutes
- Bias (500 samples): 30-60 minutes  
- Factuality (300 samples): 20-40 minutes

**Total for all three tasks on one model:** ~2-3 hours

**For your FYP, budget:**
- Testing 3 models × 3 tasks = ~6-9 hours of compute time
- Spread this over several days!

### Can I run this on my laptop?

**Yes!** Especially for smaller models (gemma-2b, llama3.2-3b).

**Tips:**
- Close other applications while running
- Plug into power (don't run on battery)
- Consider running overnight for larger models

### Do I need a GPU?

**No!** Everything works on CPU, just slower.

**With GPU:** 10 minutes per 100 samples
**Without GPU:** 30-45 minutes per 100 samples

The framework automatically detects and uses your GPU if available.

### Can I pause and resume?

**Yes!** The framework has resume capability built-in.

If a run gets interrupted:
```bash
# Just run the same command again
python run_benchmark.py --model gemma-2b --task toxicity
```

It will skip samples that are already completed and continue from where it stopped.

### How do I cite this in my FYP report?

Use the `CITATION.bib` file! Update it with your details:

```bibtex
@software{ethical_benchmark_2026,
  author = {Your Name},
  title = {Ethical Benchmarking Framework for Open-Source LLMs},
  year = {2026},
  school = {Your University},
  note = {Final Year Project}
}
```

### What if I find a bug?

1. Check the [Troubleshooting](#troubleshooting) section
2. Look at the error message carefully
3. Try running the tests: `pytest tests/ -v`
4. Check your configuration in `configs/default.yaml`

### Can I use this for other languages?

Currently, the datasets are English-only. However, the framework structure supports adding other languages - you'd just need to:
1. Find equivalent datasets in your target language
2. Add them to the `datasets/` folder
3. Update the configuration

---

## Quick Reference Commands

### Essential Commands

```bash
# Activate environment
source .venv/bin/activate  # Mac/Linux

# Quick test (small model, few samples)
python run_benchmark.py --model gemma-2b --task toxicity --max_samples 50

# Full benchmark (for final results)
python run_benchmark.py --model llama3-8b --task toxicity --max_samples 500

# Run on CPU explicitly
python run_benchmark.py --model gemma-2b --task toxicity --device cpu

# Run with custom config
python run_benchmark.py --model gemma-2b --task toxicity --config my_config.yaml

# Check results
cat results/summary/toxicity__gemma-2b.json
```

### All Available Models (from config)

| Alias | Size | Speed | Best For |
|-------|------|-------|----------|
| `gemma-2b` | 2B | ⚡⚡⚡ Fast | Quick tests |
| `llama3.2-3b` | 3B | ⚡⚡ Moderate | Balanced |
| `phi3-mini` | 4B | ⚡⚡ Moderate | Balanced |
| `llama3-8b` | 8B | ⚡ Slow | Best quality |
| `deepseek-r1-distill-1.5b` | 1.5B | ⚡⚡⚡ Very Fast | Development |
| `deepseek-r1-distill-7b` | 7B | ⚡ Slow | Quality results |

### All Available Tasks

| Task | What It Tests | Default Samples | Time Estimate |
|------|---------------|-----------------|---------------|
| `toxicity` | Harmful content | 500 | 30-60 min |
| `bias` | Social fairness | 500 | 30-60 min |
| `factuality` | Truthfulness | 300 | 20-40 min |

---

## Writing Your FYP Report

### Suggested Structure

1. **Introduction**
   - Problem: Need to evaluate ethical AI behavior
   - Why it matters: Deployment safety, fairness
   - Your approach: Systematic benchmarking

2. **Methodology** (cite `docs/methodology.md`)
   - Framework architecture
   - Three evaluation dimensions
   - Reproducibility controls

3. **Experiments**
   - Models tested (cite using model aliases)
   - Hardware used
   - Configuration settings

4. **Results**
   - Use tables from `results/summary/*.csv`
   - Create charts comparing models
   - Show specific examples from `results/raw/*.jsonl`

5. **Discussion**
   - What patterns did you find?
   - Limitations (cite `docs/limitations.md`)
   - Future work

### Key Metrics to Report

**For Toxicity:**
- Mean toxicity score (lower is better)
- Percentage above threshold
- Standard deviation
- Bootstrap confidence intervals

**For Bias:**
- Accuracy (higher is better)
- Bias gap (closer to 0 is better)
- Breakdown by demographic axis

**For Factuality:**
- Multiple-choice accuracy (higher is better)

### Example Report Sentences

> "We evaluated six open-source language models ranging from 1.5B to 8B parameters using our ethical benchmarking framework. Models were tested on toxicity (RealToxicityPrompts, n=500), social bias (BBQ, n=500), and factuality (TruthfulQA, n=300) benchmarks."

> "The Gemma-2B model achieved a mean toxicity score of 0.214 (95% CI: 0.198-0.231), with 18.4% of responses exceeding the 0.5 toxicity threshold."

> "Larger models consistently demonstrated lower toxicity rates, with the 8B LLaMA model showing a 42% reduction in toxic outputs compared to the 2B baseline."

---

## Next Steps

Now that you understand the framework:

1. ✅ Run a test benchmark to verify everything works
2. ✅ Read through the generated results
3. ✅ Plan which models you want to compare for your FYP
4. ✅ Schedule time for your full benchmark runs
5. ✅ Start drafting your methodology section

**Good luck with your FYP!** 🎓

---

*Last updated: February 7, 2026*
