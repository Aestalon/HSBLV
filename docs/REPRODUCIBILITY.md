# Reproducibility Notes

## Included Materials

The five settings are MBPP with Qwen3, Llama-3.1, and GPT-4o-mini, and MATH with Qwen3 and Llama-3.1. The released traces contain task identifiers, derived signals, cost proxies, and evaluation labels. They do not include benchmark problem statements, generated solution text, model weights, manuscript files, local logs, or credentials.

Files under `results/` are copied from the existing experiment records without changing their numerical values. In the legacy HGA/COMET JSON records, the key `MBPP-GPT-4o` refers to the GPT-4o-mini traces. The release scripts use the corrected display name `MBPP-GPT-4o-mini`.

Some MATH evaluation rows use legacy `fix_*` placeholder task identifiers. The original MATH analyses associate signals and labels by row position, and this release preserves that behavior and the original files. Placeholder IDs do not independently establish correspondence to a specific original problem. The release checker reports their count rather than silently replacing identifiers. Their provenance must be recovered from the original correction records before claiming task-ID-based verification of those rows.

## Protocols

- Signal features: negative log probability, negative minimum margin, and log-probability variance, min-max normalized with Gaussian noise of standard deviation 0.1.
- Feedback: independently flipped pass/fail labels with probability 0.10, using seed `seed + 1`.
- Evaluation cost: `0.1 + 0.9 * normalized_output_length`.
- Decision cost: `0.1 + 0.9 * normalized_input_length`.
- HGA budgets: 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, and 0.50.
- HGA score fitting: first `min(50, N)` items. HGA gate calibration: first `max(min(50, N), floor(0.5*N))` items.
- HGA seeds: 42 through 51. Reported standard deviations use `ddof=1`.
- Main runner: seeds `base_seed + 100*i`; score fitting uses `min(50, N//5)` items. This differs from the HGA reanalysis protocol.

The supplied traces fix the item order and model generations. Random seeds vary feature noise, feedback noise, and policy randomness; they do not represent independent model-generation runs. Normalization and calibration use the implementation's original offline protocol. The release does not establish a fully causal online preprocessing or calibration protocol.

## Metrics

`src/evaluation.py` defines hit rate, error coverage, information gain, corrected accuracy, and bootstrap summaries. The reanalysis scripts additionally report the number of selected items and fraction of the available evaluation-cost budget spent. Improvements in audited precision should be read alongside coverage and spending; they do not alone establish improved discovery utility at equal expenditure.

## Environment and Scope

Python 3.10+ is required for the type annotations. CPU reanalysis requires NumPy and SciPy. The dependency ranges are installation constraints, not a reconstruction of the original historical software environment. Optional generation requires PyTorch, Transformers, Datasets, and model/API access. The original hardware and complete historical package lockfile are not established by this release.

Original generation scripts retain the historical generation and answer-checking logic. In particular, the MATH string-normalization checker is not a general symbolic equivalence verifier. New API responses and stochastic GPU generations may differ from the included traces. Rerunning analysis on the fixed traces is the supported low-cost reproducibility path.

The release changes file locations, default output destinations, credential configuration, and the GPT-4o-mini display name. It does not modify the policy calculations or original recorded numerical results. Newly generated tables and JSON summaries go to `outputs/`.
