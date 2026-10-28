"""Step 1: audit every run directory before any analysis.

Checks (one row per run x check in audit/checks.csv):
  config        protocol settings equal the reference (common_eval.REF_CFG)
  dense_log     epochs 0..T each exactly once; LR column equals the schedule
                (row e holds the LR of epoch e-1 -> e); final_metrics == row T
  target        target = dense val acc - tol in config, rounds.csv and trials
  round1_start  round 1 starts from the dense model (before_val_acc == dense val acc);
                a mismatch means round 1 came from a different (e.g. smoke-test) run
  rounds_seq    rounds numbered 1..K without gaps; sparsity == 1 - (1 - p)^k
  chain         before_val_acc(k) == final_val_acc(k-1)
  trials        trial JSONs == rewind_trials.csv; recovered flag == (acc >= target);
                no duplicate (round, t); n_trials matches
  search        replaying the binary search on the recorded trial outcomes gives
                exactly the recorded trials and t*
  final         final_val_acc of a round == the trial at t* (or the pruned model if t* = T)
  summaries     round_XX/summary.json agrees with rounds.csv
  curves        every curve file covers epochs t+1..T once, last val acc == trial result
  completeness  chain ended with an unrecoverable round or after all rounds;
                any round directory without a summary is reported as incomplete
  log           wrapper log: smoke-test invocations into the same run_dir, tracebacks

Writes audit/checks.csv, audit/rounds_status.csv, audit/run_manifest.json and
audit/AUDIT.md. Raw results are only read, never modified.
"""

import json
import os
import re

import numpy as np
import pandas as pd

from common_eval import AUDIT, MANIFEST, REF_CFG, DISCOVERY_LABELS, load_run, lr_at_epoch, registry, read_csv

EPS = 1e-6


def binary_search_replay(outcome, after_acc, target, T, t_min, res):
    """Replays run_imp.py's search on recorded outcomes {t: recovered}.
    Returns (t_star or None, list of t's the search would have tried, ok)."""
    if after_acc >= target - EPS:
        return T, [], True
    tried = [t_min]
    if t_min not in outcome:
        return None, tried, False
    if not outcome[t_min]:
        return None, tried, True
    lo, hi = t_min, T
    while hi - lo > res:
        mid = (lo + hi) // 2
        tried.append(mid)
        if mid not in outcome:
            return None, tried, False
        if outcome[mid]:
            lo = mid
        else:
            hi = mid
    return lo, tried, True


def audit_run(label, path, group):
    run = load_run(label, path)
    checks = []
    round_status = {}

    def add(name, ok, detail=""):
        checks.append(dict(run=label, group=group, check=name,
                           status="PASS" if ok is True else ("WARN" if ok == "warn" else "FAIL"),
                           detail=detail))

    cfg = run["cfg"]
    # ---- config
    bad = [k for k, v in REF_CFG.items() if k in cfg and cfg[k] != v]
    missing = [k for k in REF_CFG if k not in cfg]
    add("config", not bad and not missing,
        "; ".join([f"{k}={cfg[k]} (expected {REF_CFG[k]})" for k in bad] +
                  [f"missing {k}" for k in missing]))

    # ---- dense log
    raw, dense, fin = run["dense_raw"], run["dense"], run["dense_final"]
    T = int(cfg.get("epochs", 80))
    eps_ = raw["epoch"].tolist() if len(raw) else []
    dup = len(eps_) - len(set(eps_))
    gaps = sorted(set(range(T + 1)) - set(eps_))
    lr_ok = True
    for _, row in dense.iterrows():
        e = int(row["epoch"])
        if e == 0:
            continue
        if abs(row["lr"] - lr_at_epoch(e - 1, cfg)) > 1e-12:
            lr_ok = False
    final_ok = len(dense) and abs(dense.loc[dense["epoch"] == T, "val_acc"].iloc[0] - fin.get("val_acc", np.nan)) < EPS
    add("dense_log", dup == 0 and not gaps and lr_ok and bool(final_ok),
        f"duplicate rows {dup}, missing epochs {gaps[:5]}, lr schedule ok {lr_ok}, final_metrics matches row {T}: {bool(final_ok)}")

    # ---- target
    a_dense = fin.get("val_acc", np.nan)
    target = cfg.get("target_val_acc", np.nan)
    r = run["rounds"]
    tgt_cfg_ok = abs(target - (a_dense - cfg.get("tol", 0.5))) < EPS
    bad_rows = r.loc[(r["target_val_acc"] - target).abs() > EPS, "round"].tolist() if len(r) else []
    add("target", tgt_cfg_ok and not bad_rows,
        f"config target {target} vs dense {a_dense} - tol; rounds with a different target: {bad_rows}")

    # ---- round 1 must start from the dense model
    contaminated = False
    if len(r):
        b1 = float(r.loc[r["round"] == 1, "before_val_acc"].iloc[0])
        contaminated = abs(b1 - a_dense) > EPS
        add("round1_start", not contaminated,
            f"round 1 before_val_acc {b1} vs dense val acc {a_dense}"
            + (" -> round 1 was produced from a different dense run (smoke test reused by resume)" if contaminated else ""))

    # ---- rounds sequence and sparsity
    if len(r):
        ks = r["round"].tolist()
        seq_ok = ks == list(range(1, len(ks) + 1))
        p = cfg.get("prune_frac", 0.2)
        sp_err = float(np.max(np.abs(r["sparsity"] - (1 - (1 - p) ** r["round"]))))
        add("rounds_seq", seq_ok and sp_err < 1e-3, f"rounds {ks}; max |sparsity - (1-p)^k| = {sp_err:.2e}")

    # ---- chain continuity
    if len(r) > 1:
        brk = []
        for k in r["round"].iloc[1:]:
            prev = r.loc[r["round"] == k - 1, "final_val_acc"].iloc[0]
            cur = r.loc[r["round"] == k, "before_val_acc"].iloc[0]
            if not (abs(prev - cur) < EPS):
                brk.append(int(k))
        add("chain", not brk, f"rounds whose before_val_acc != previous final_val_acc: {brk}")

    # ---- trials: JSON vs CSV, flags, duplicates
    tj, tc = run["trials_json"], run["trials_csv"]
    completed_rounds = set(r["round"].tolist()) if len(r) else set()
    details = []
    ok = True
    if len(tj):
        d = tj.duplicated(["round", "t"]).sum()
        if d:
            ok = False
            details.append(f"{d} duplicate trial JSONs")
        tj_done = tj[tj["round"].isin(completed_rounds)]
        if len(tc):
            m = tj_done.merge(tc, on=["round", "t"], how="outer", suffixes=("_j", "_c"), indicator=True)
            only = m[m["_merge"] != "both"]
            diff = m[(m["_merge"] == "both") & ((m["final_val_acc_j"] - m["final_val_acc_c"]).abs() > EPS)]
            if len(only) or len(diff):
                ok = False
                details.append(f"{len(only)} trials only in one of JSON/CSV, {len(diff)} value mismatches")
            flag = (tc["final_val_acc"] >= tc["target_val_acc"] - EPS).astype(int)
            if (flag != tc["recovered"]).any():
                ok = False
                details.append("recovered flag inconsistent with target")
        elif len(tj_done):
            ok = False
            details.append("trial JSONs exist but rewind_trials.csv is missing")
        for k in completed_rounds:
            n_rec = int(r.loc[r["round"] == k, "n_trials"].iloc[0])
            n_json = int((tj["round"] == k).sum())
            if n_rec != n_json:
                ok = False
                details.append(f"round {k}: n_trials {n_rec} but {n_json} trial JSONs")
        orphan = sorted(set(tj["round"]) - completed_rounds)
        if orphan:
            details.append(f"trial JSONs of rounds without a result row: {orphan}")
    add("trials", ok, "; ".join(details))

    # ---- binary search replay and final accuracy
    s_fail, f_fail = [], []
    res, t_min = int(cfg.get("resolution", 2)), int(cfg.get("t_min", 2))
    for _, row in r.iterrows():
        k = int(row["round"])
        tk = tj[tj["round"] == k] if len(tj) else pd.DataFrame(columns=["t", "final_val_acc"])
        outcome = {int(t): (a >= target - EPS) for t, a in zip(tk["t"], tk["final_val_acc"])}
        t_rep, tried, ok_rep = binary_search_replay(outcome, row["after_val_acc"], target, T, t_min, res)
        t_rec = None if pd.isna(row["t_star"]) else int(row["t_star"])
        if not ok_rep or t_rep != t_rec or sorted(tried) != sorted(outcome):
            s_fail.append(k)
        if t_rec is not None:
            if t_rec == T:
                want = row["after_val_acc"]
            else:
                hit = tk.loc[tk["t"] == t_rec, "final_val_acc"]
                want = hit.iloc[0] if len(hit) else np.nan
            if not abs(row["final_val_acc"] - want) <= EPS:
                f_fail.append(k)
    add("search", not s_fail, f"rounds where the replayed search disagrees: {s_fail}")
    add("final", not f_fail, f"rounds whose final_val_acc is not the t* trial: {f_fail}")

    # ---- summaries
    mism = []
    for _, row in r.iterrows():
        s = run["summaries"].get(int(row["round"]))
        if s is None:
            mism.append(f"{int(row['round'])}: missing")
            continue
        ts = s.get("t_star")
        if not ((ts is None and pd.isna(row["t_star"])) or (ts is not None and ts == row["t_star"])):
            mism.append(f"{int(row['round'])}: t_star")
        if abs(s["sparsity"] - row["sparsity"]) > 1e-9:
            mism.append(f"{int(row['round'])}: sparsity")
    add("summaries", not mism, "; ".join(mism))

    # ---- curves
    cbad = []
    for f in run["curves"]:
        k = int(re.search(r"round_(\d+)", f).group(1))
        t = int(re.search(r"curve_t(\d+)", f).group(1))
        c = read_csv(f)
        expect = list(range(t + 1, T + 1))
        has_json = len(tj) and ((tj["round"] == k) & (tj["t"] == t)).any()
        if not has_json:
            cbad.append(f"r{k} t{t}: partial curve without trial result ({len(c)} of {len(expect)} epochs)")
            continue
        if c["epoch"].tolist() != expect:
            cbad.append(f"r{k} t{t}: epochs {c['epoch'].min()}..{c['epoch'].max()} n={len(c)}")
            continue
        acc = float(tj.loc[(tj["round"] == k) & (tj["t"] == t), "final_val_acc"].iloc[0])
        if abs(c["val_acc"].iloc[-1] - acc) > 0.011:
            cbad.append(f"r{k} t{t}: last curve acc {c['val_acc'].iloc[-1]} vs trial {acc}")
    add("curves", "warn" if cbad else True, "; ".join(cbad))

    # ---- completeness
    last = r.iloc[-1] if len(r) else None
    incomplete = [k for k in run["round_dirs"] if k not in run["summaries"]]
    ended = last is not None and (pd.isna(last["t_star"]) or int(last["round"]) == int(cfg.get("rounds", 10)))
    add("completeness", ended and not incomplete,
        f"last round {None if last is None else int(last['round'])}, ended by "
        + ("unrecoverable round" if last is not None and pd.isna(last["t_star"]) else "round limit" if ended else "nothing (run interrupted)")
        + (f"; round dirs without summary (incomplete): {incomplete}" if incomplete else ""))

    # ---- wrapper log
    if run["log"] is not None:
        inv = re.findall(r"^\$ .*?(train_dense\.py|run_imp\.py)(.*)$", run["log"], flags=re.M)
        smoke = [f"{s}{a.strip()[:60]}" for s, a in inv if "--epochs 2" in a or "--rounds 1 " in a + " "]
        tbs = len(re.findall(r"^Traceback", run["log"], flags=re.M))
        res_from = re.findall(r"Resuming dense training from epoch (\d+)", run["log"])
        add("log", "warn" if (smoke or tbs) else True,
            f"smoke-test invocations into this run_dir: {len(smoke)}; tracebacks: {tbs}; "
            f"dense resumed from epochs {sorted(set(map(int, res_from)))}")
    else:
        add("log", True, "no wrapper log (original run)")

    # ---- round status
    for _, row in r.iterrows():
        k = int(row["round"])
        if contaminated:
            st = "contaminated"
        elif pd.isna(row["t_star"]):
            st = "unrecoverable"
        elif int(row["t_star"]) == T and int(row["n_trials"]) == 0:
            st = "no_retrain"
        else:
            st = "searched"
        round_status[k] = st
    for k in incomplete:
        round_status[k] = "contaminated" if contaminated else "incomplete"

    if contaminated:
        status, include = "excluded", False
        reason = ("IMP chain invalid: round 1 (and its mask and model) came from the 2-epoch smoke test "
                  "that was run into the same run_dir and reused by the resume logic.")
    else:
        include = not any(c["status"] == "FAIL" and c["check"] not in ("completeness",) for c in checks)
        status = "included" if include else "excluded"
        reason = ""
        if incomplete:
            reason = f"run interrupted: round(s) {incomplete} incomplete (t* censored), later rounds not run"
        if not include:
            reason = "failed checks: " + ", ".join(c["check"] for c in checks if c["status"] == "FAIL")
    entry = dict(label=label, path=path, group=group, include=include, status=status, reason=reason,
                 discovery=label in DISCOVERY_LABELS, dense_val_acc=a_dense, target_val_acc=target,
                 rounds={str(k): v for k, v in sorted(round_status.items())})
    return checks, entry


def main():
    os.makedirs(AUDIT, exist_ok=True)
    all_checks, entries = [], []
    for label, path, group in registry():
        c, e = audit_run(label, path, group)
        all_checks += c
        entries.append(e)
    checks = pd.DataFrame(all_checks)
    checks.to_csv(os.path.join(AUDIT, "checks.csv"), index=False)
    rs = pd.DataFrame([dict(run=e["label"], round=int(k), status=v, run_included=e["include"])
                       for e in entries for k, v in e["rounds"].items()])
    rs.to_csv(os.path.join(AUDIT, "rounds_status.csv"), index=False)
    manifest = dict(note="Generated by 01_audit_runs.py. Only runs with include=true and rounds with status "
                         "no_retrain/searched/unrecoverable enter the analysis; 'incomplete' rounds contribute "
                         "their finished trials to the accuracy fit only.",
                    runs=entries)
    json.dump(manifest, open(MANIFEST, "w"), indent=2)

    # human-readable summary
    piv = checks.pivot(index="run", columns="check", values="status")
    order = [e["label"] for e in entries]
    piv = piv.loc[order]
    lines = ["# Run audit", "", "Generated by `01_audit_runs.py`. PASS / WARN / FAIL per check; details in `checks.csv`.", ""]
    lines.append("| run | " + " | ".join(piv.columns) + " |")
    lines.append("|---|" + "---|" * len(piv.columns))
    for run_, row in piv.iterrows():
        lines.append(f"| {run_} | " + " | ".join(row.fillna("-")) + " |")
    lines += ["", "## Decision per run", ""]
    for e in entries:
        counts = pd.Series(list(e["rounds"].values())).value_counts().to_dict()
        lines.append(f"- **{e['label']}** ({e['group']}): {e['status']}. Rounds: {counts}. {e['reason']}")
    lines += ["", "## Non-PASS details", ""]
    for _, c in checks[checks["status"] != "PASS"].iterrows():
        lines.append(f"- {c['run']} / {c['check']} ({c['status']}): {c['detail']}")
    open(os.path.join(AUDIT, "AUDIT.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

    pd.set_option("display.width", 200)
    print(piv.to_string())
    print()
    for e in entries:
        print(f"{e['label']:12s} {e['status']:9s} rounds={dict(pd.Series(list(e['rounds'].values())).value_counts())} {e['reason']}")
    print(f"\nWrote {AUDIT}")


if __name__ == "__main__":
    main()
