# HSBLV

Experiment code and recorded results for **Heteroskedastic Signals in Budgeted LLM Verification: Structural Heterogeneity Limits Optimization Gains**.

This repository studies budgeted error auditing with heterogeneous uncertainty signals and verification-cost proxies. It includes global threshold policies, cost-stratified thresholding (CST), heterogeneity-gated allocation (HGA), optimization baselines, and exploratory routing/COMET analyses. Manuscript sources and paper PDFs are intentionally excluded.

## Quick Start

Use Python 3.10 or newer. Reanalysis of the supplied traces requires only NumPy and SciPy and runs on CPU, without model weights or an API key.

```bash
git clone https://github.com/Aestalon/HSBLV.git
cd HSBLV
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/run_hga_reanalysis.py
python scripts/check_release.py
```

New analysis files are written to ignored `outputs/`; the recorded reference results in `results/` remain unchanged. The analysis scripts also generate optional LaTeX table snippets locally; these generated files are ignored by Git.

The release checker validates the tracked file types, data schemas, original data/result SHA-256 hashes, and exact numerical agreement of the HGA rerun with the recorded results.

## Experiments

```bash
# Main policy experiments with recorded signals and labels
python src/main.py --data mbpp --model qwen3 --n-seeds 10
python src/main.py --data mbpp --model llama3 --n-seeds 10
python src/main.py --data gpt --model gpt-4o-mini --n-seeds 10
python src/main.py --data math --model qwen3 --n-seeds 10
python src/main.py --data math --model llama3 --n-seeds 10

# HGA calibration reanalysis (all five benchmark/model settings)
python scripts/run_hga_reanalysis.py

# Exploratory analyses, separate from the primary HGA evidence
python scripts/run_comet_reanalysis.py
python scripts/routing_reanalysis.py
```

The main experiment runner also accepts `--signal`, `--result`, `--output`, `--seed`, and `--mode`. Ten-seed main experiments use seeds `42 + 100*i`; HGA reanalysis uses seeds 42 through 51. These are separate protocols, so their output tables should not be conflated.

## Repository Layout

| Path | Contents |
| --- | --- |
| `src/` | Feature construction, auditing policies, evaluation, and controlled analyses |
| `scripts/run_hga_reanalysis.py` | HGA calibration-based reanalysis |
| `scripts/run_comet_reanalysis.py` | Exploratory COMET policy selection |
| `scripts/routing_reanalysis.py` | Exploratory model-routing analysis |
| `scripts/MBPP_*.py`, `scripts/MATH_*.py` | Original model-generation/evaluation workflows, with environment-based configuration |
| `data/` | Derived uncertainty/cost signals and recorded pass/fail labels |
| `results/` | Original recorded numerical summaries, including per-seed aggregates |
| `docs/REPRODUCIBILITY.md` | Protocol details, data scope, and practical limitations |

## Model Generation

Generation is optional and considerably more expensive than reanalysis. Install `requirements-generation.txt`, obtain the benchmark datasets and model access separately, and configure paths using environment variables. MBPP scripts expect a Hugging Face dataset saved with `save_to_disk`; MATH scripts expect the original directory of JSON problem files.

```bash
export MBPP_PATH=/path/to/mbpp
export MATH_PATH=/path/to/MATH
export MODEL_PATH=/path/to/model
python scripts/MBPP_Qwen3.py
```

For the API-based MBPP script, set `OPENAI_API_KEY`; optionally set `OPENAI_BASE_URL`. Its default model is `gpt-4o-mini`. No credentials are distributed. On PowerShell, use `$env:MBPP_PATH = '...'` and equivalent assignments for the other variables.

MBPP evaluation executes generated Python code and uses POSIX signal-based timeouts. Run generation/evaluation in an isolated Linux environment; do not run untrusted model output on a machine with sensitive files or credentials. Use a dedicated working directory for generation outputs, then explicitly copy the desired traces into `data/`.

## Interpretation and Licensing

Hit rate measures precision among audited items, rather than total errors found. Evaluation also records error coverage and budget use. Costs are normalized token-length proxies; they are not measured wall-clock verification costs. HGA is a calibration-based analysis, with a larger calibration prefix than the initial score fit; see the protocol notes before comparing methods.

Code is provided under the MIT license. Included data are derived numerical traces and evaluation labels, not a redistribution of the original benchmark problem sets or model weights. Benchmark and model licenses continue to apply to their respective original materials. No conference acceptance or publication status is asserted by this repository.
