"""Step 2: reproduce the original analysis (imp_rewind/analyze.py, unchanged) on the
audited runs.

Two runs of analyze.py:
  results/original_method_pooled/   all included runs pooled (as analyze.py --runs)
  results/original_method_heldout/  discovery runs as --runs, the 8 new runs as
                                    --test_runs (the protocol analyze.py was built for)

analyze.py names runs by folder name. The original discovery runs are imp_rewind/runs/
seed1 and seed2; the repeated seed1/seed2 are excluded by the audit, so names do not clash.
"""

import os
import subprocess
import sys

from common_eval import IMP_REWIND, OUT, load_manifest


def main():
    man = load_manifest()
    inc = [e for e in man["runs"] if e["include"]]
    disc = [e["path"] for e in inc if e["discovery"]]
    new = [e["path"] for e in inc if not e["discovery"]]
    names = [os.path.basename(p) for p in disc + new]
    assert len(names) == len(set(names)), f"run names clash: {names}"
    analyze = os.path.join(IMP_REWIND, "analyze.py")
    jobs = [
        ("original_method_pooled", ["--runs", *disc, *new]),
        ("original_method_heldout", ["--runs", *disc, "--test_runs", *new]),
    ]
    for name, args in jobs:
        out = os.path.join(OUT, name)
        os.makedirs(out, exist_ok=True)
        cmd = [sys.executable, analyze, *args, "--out", out]
        print("$", " ".join(os.path.relpath(c, IMP_REWIND) if os.path.isabs(c) else c for c in cmd))
        res = subprocess.run(cmd, cwd=IMP_REWIND, capture_output=True, text=True)
        open(os.path.join(out, "analyze_stdout.txt"), "w", encoding="utf-8").write(res.stdout + res.stderr)
        if res.returncode:
            print(res.stdout[-2000:], res.stderr[-2000:])
            raise SystemExit(f"analyze.py failed for {name}")
        print(res.stdout.split("=== How well each rule predicts t*")[1][:2500] if "=== How well" in res.stdout else res.stdout[-2500:])


if __name__ == "__main__":
    main()
