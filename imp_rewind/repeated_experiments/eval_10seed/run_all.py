"""Run the whole 10-seed evaluation from scratch: audit -> original analysis -> formula
evaluation -> model checks -> figures -> report tables -> redefined t* -> re-evaluation.
Usage (from this folder): python run_all.py"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for step in ["01_audit_runs.py", "02_original_method.py", "03_formula_eval.py", "04_model_checks.py", "05_figures.py", "06_report_tables.py",
             "07_redefine_tstar.py", "08_reevaluate.py"]:
    print(f"\n######## {step}")
    r = subprocess.run([sys.executable, "-u", os.path.join(HERE, step)], cwd=HERE)
    if r.returncode:
        sys.exit(f"{step} failed")
