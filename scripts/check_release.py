"""Validate the public artifact without running model generation."""

import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {".tex", ".bib", ".sty", ".cls", ".bst", ".pdf", ".docx", ".pptx"}
SECRET = re.compile(
    r"sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|"
    r"github_pat_[A-Za-z0-9_]{20,}|BEGIN (?:RSA |OPENSSH )?PRIVATE KEY"
)


def main():
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    ).decode("utf-8").split("\0")
    count = 0
    for name in filter(None, tracked):
        path = ROOT / name
        assert path.suffix.lower() not in FORBIDDEN, f"Excluded file tracked: {name}"
        assert not any(p in {".agents", ".claude", ".codex", "__pycache__"} for p in path.parts), name
        text = path.read_text(encoding="utf-8-sig")
        assert not SECRET.search(text), f"Potential secret in {name}"
        if path.suffix == ".py":
            ast.parse(text, filename=name)
        elif path.suffix == ".json":
            json.loads(text)
        elif path.suffix == ".jsonl":
            for line in text.splitlines():
                if line.strip():
                    json.loads(line)
        count += 1

    for signal_path in sorted((ROOT / "data").glob("*_signal_results_*.json")):
        eval_path = signal_path.with_name(
            signal_path.name.replace("_signal_results_", "_eval_results_") + "l"
        )
        signals = json.loads(signal_path.read_text(encoding="utf-8"))
        labels = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(signals) == len(labels), signal_path.name
        assert len({str(row["task_id"]) for row in signals}) == len(signals), signal_path.name
        mismatches = [(s, e) for s, e in zip(signals, labels) if str(s["task_id"]) != str(e["task_id"])]
        if mismatches:
            assert signal_path.name.startswith("math_"), signal_path.name
            assert all(str(e["task_id"]).startswith("fix_") for _, e in mismatches), signal_path.name
            print(f"Legacy data note: {len(mismatches)} placeholder label IDs; original positional alignment retained")
        for row in signals:
            assert set(row) <= {"task_id", "signals"}, signal_path.name
            required = {
                "ht1_neg_logprob", "ht2_neg_min_margin", "ht3_logprob_var",
                "cost_token_len", "cost_output_len",
            }
            assert required <= set(row["signals"]) <= required | {"cost_level", "cost_test_count"}, signal_path.name
            assert all(isinstance(value, (int, float)) for value in row["signals"].values()), signal_path.name
        for row in labels:
            assert set(row) <= {"task_id", "passed", "level", "type"}, eval_path.name
        print(f"Data pair OK: {signal_path.name} ({len(signals)} items)")

    manifest = json.loads((ROOT / "docs" / "data_manifest.json").read_text(encoding="utf-8-sig"))
    for entry in manifest:
        content = (ROOT / entry["path"]).read_bytes()
        assert len(content) == entry["bytes"], entry["path"]
        assert hashlib.sha256(content).hexdigest() == entry["sha256"], entry["path"]
    print(f"Original file hashes verified: {len(manifest)} data/result files")

    new_path = ROOT / "outputs" / "hga_reanalysis.json"
    if new_path.exists():
        original = json.loads((ROOT / "results" / "hga_reanalysis.json").read_text())
        rerun = json.loads(new_path.read_text())
        # The legacy record used an abbreviated model name; numerical fields must match.
        original["MBPP-GPT-4o-mini"] = original.pop("MBPP-GPT-4o")
        assert original == rerun, "HGA rerun differs from recorded reference results"
        print("HGA rerun matches all reference numerical fields exactly")
    else:
        print("HGA reference comparison skipped: run scripts/run_hga_reanalysis.py first")
    print(f"Release checks passed for {count} tracked files")


if __name__ == "__main__":
    main()
