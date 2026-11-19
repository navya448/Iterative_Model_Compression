"""Run every KD experiment in plan.json, unattended, in order.

    python run_all.py --smoke        # ~10-20 min end-to-end check of every dataset/model
    python run_all.py                # the real thing (days; see README for estimates)

What it does:
  * downloads each dataset once (prepare), then for every stage / setting / seed:
    the dense model (= the teacher), then one IMP chain per method (kd, ce, ...)
  * every job is a separate process with its own log in <runs_dir>/logs/
  * finished work is never redone: rerun the same command after a crash, a reboot
    or Ctrl-C and it continues where it stopped (down to the last finished epoch)
  * a failing job is retried (--retries, default 2) after a pause; if it still fails,
    jobs that need it are skipped and everything else keeps going
  * after every finished IMP chain the analysis is refreshed (analysis/summary.md)
  * progress is in <runs_dir>/status.json and <runs_dir>/logs/run_all.log

Options: --only <text> runs only jobs whose name contains the text (e.g. cifar100),
--dry_run lists the jobs, --gpu 1 picks a GPU, --gpu_fraction 0.5 shares the GPU.
"""

import argparse
import atexit
import json
import os
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
JOB = os.path.join(HERE, "kd_job.py")
ANALYZE = os.path.join(HERE, "analyze_kd.py")

# Harsh pruning and a tight target, so the smoke test really runs the t* search.
SMOKE = dict(epochs=6, rounds=3, t_min=1, resolution=1, prune_frac=0.5, rel_tol=0.01,
             train_limit=2000)


class Runner:
    def __init__(self, args):
        self.args = args
        self.logs = os.path.join(args.runs_dir, "logs")
        os.makedirs(self.logs, exist_ok=True)
        self.main_log = open(os.path.join(self.logs, "run_all.log"), "a", encoding="utf-8")
        self.status_path = os.path.join(args.runs_dir, "status.json")
        self.status = {}
        self.env = dict(os.environ, PYTHONUNBUFFERED="1")
        if args.gpu is not None:
            self.env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        if args.gpu_fraction is not None:
            self.env["IMP_GPU_FRACTION"] = str(args.gpu_fraction)

    def say(self, msg):
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        print(line, flush=True)
        self.main_log.write(line + "\n")
        self.main_log.flush()

    def save_status(self):
        tmp = self.status_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(dict(updated=f"{datetime.now():%Y-%m-%d %H:%M:%S}", jobs=self.status), f,
                      indent=2)
        os.replace(tmp, self.status_path)

    def run_process(self, cmd, log_path, prefix):
        """Run one command, stream its output to the console and its log. Returns exit code."""
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} {' '.join(cmd)}\n")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1, env=self.env, cwd=HERE,
                                    encoding="utf-8", errors="replace")
            try:
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                    if not self.args.quiet:
                        print(f"{prefix} {line}", end="", flush=True)
                return proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise

    def run_job(self, job):
        st = self.status.setdefault(job["id"], dict(state="pending", attempts=0))
        st["log"] = job["log"]
        if not job.get("always") and job["done"]():
            st["state"] = "done"
            self.save_status()
            self.say(f"SKIP (already done) {job['id']}")
            return True
        for attempt in range(1, self.args.retries + 2):
            st.update(state="running", attempts=st["attempts"] + 1,
                      started=f"{datetime.now():%Y-%m-%d %H:%M:%S}")
            self.save_status()
            self.say(f"START {job['id']} (attempt {attempt}) -> log {job['log']}")
            t0 = time.time()
            code = self.run_process(job["cmd"], job["log"], f"[{job['id']}]")
            minutes = (time.time() - t0) / 60
            st["minutes"] = round(st.get("minutes", 0) + minutes, 1)
            if code == 0 and (job.get("always") or job["done"]()):
                st.update(state="done", finished=f"{datetime.now():%Y-%m-%d %H:%M:%S}")
                self.save_status()
                self.say(f"DONE  {job['id']} in {minutes:.1f} min")
                return True
            st.update(state="failed", exit_code=code)
            self.save_status()
            self.say(f"FAIL  {job['id']} (exit {code}) after {minutes:.1f} min. "
                     f"Last lines of {job['log']}:\n" + tail(job["log"]))
            if attempt <= self.args.retries:
                self.say(f"      retrying in {self.args.retry_wait} s (resumes where it stopped)")
                time.sleep(self.args.retry_wait)
        return False

    def analysis(self):
        if self.args.no_analysis:
            return
        cmd = [sys.executable, ANALYZE, "--runs_dir", self.args.runs_dir,
               "--out", self.args.analysis_dir, "--n_boot", str(self.args.n_boot)]
        self.say("Refreshing analysis ...")
        log = os.path.join(self.logs, "analysis.log")
        try:
            code = self.run_process(cmd, log, "[analysis]")
            self.say("Analysis " + ("updated: " + os.path.join(self.args.analysis_dir, "summary.md")
                                    if code == 0 else f"FAILED (exit {code}), see {log}"))
        except KeyboardInterrupt:
            raise
        except Exception as e:                      # never stop the experiments for this
            self.say(f"Analysis could not run: {e}")


def tail(path, n=15):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""


def build_jobs(plan, args):
    d = dict(plan["defaults"])
    train_limit = None
    if args.smoke:
        d.update({k: v for k, v in SMOKE.items() if k != "train_limit"})
        train_limit = SMOKE["train_limit"]
    common = ["--data_dir", args.data_dir, "--runs_dir", args.runs_dir]
    if train_limit:
        common += ["--train_limit", str(train_limit)]
    logs = os.path.join(args.runs_dir, "logs")
    jobs, seen = [], set()

    def add(job):
        if job["id"] not in seen:
            seen.add(job["id"])
            jobs.append(job)

    datasets = []
    for stage in plan["stages"]:
        for ds, _ in stage["settings"]:
            if ds not in datasets:
                datasets.append(ds)
    for ds in datasets:
        add(dict(id=f"prepare:{ds}", kind="prepare", deps=[],
                 cmd=[sys.executable, JOB, "prepare", "--dataset", ds, "--data_dir", args.data_dir,
                      "--runs_dir", args.runs_dir],
                 log=os.path.join(logs, f"prepare_{ds}.log"), done=lambda: False, always=True))

    for stage in plan["stages"]:
        seeds = stage["seeds"][:1] if args.smoke else stage["seeds"]
        for seed in seeds:
            for ds, model in stage["settings"]:
                name = f"{ds}_{model}"
                sdir = os.path.join(args.runs_dir, name, f"seed{seed}")
                did = f"dense:{name}:seed{seed}"
                add(dict(id=did, kind="dense", deps=[f"prepare:{ds}"],
                         cmd=[sys.executable, JOB, "dense", "--dataset", ds, "--model", model,
                              "--seed", str(seed), "--epochs", str(d["epochs"]), *common],
                         log=os.path.join(logs, f"dense_{name}_seed{seed}.log"),
                         done=lambda p=os.path.join(sdir, "dense", "final_metrics.json"):
                         os.path.exists(p)))
                for meth in stage["methods"]:
                    add(dict(id=f"imp:{name}:seed{seed}:{meth}", kind="imp", deps=[did],
                             cmd=[sys.executable, JOB, "imp", "--dataset", ds, "--model", model,
                                  "--seed", str(seed), "--method", meth,
                                  "--rounds", str(d["rounds"]), "--prune_frac", str(d["prune_frac"]),
                                  "--rel_tol", str(d["rel_tol"]), "--resolution", str(d["resolution"]),
                                  "--t_min", str(d["t_min"]), "--probe_frac", str(d["probe_frac"]),
                                  "--kd_T", str(d["kd_T"]), "--kd_lambda", str(d["kd_lambda"]),
                                  *common],
                             log=os.path.join(logs, f"imp_{name}_seed{seed}_{meth}.log"),
                             done=lambda p=os.path.join(sdir, meth, "DONE.json"): os.path.exists(p)))
    if args.only:
        keep = {j["id"] for j in jobs if args.only in j["id"]}
        # keep what the selected jobs depend on
        for j in jobs:
            if j["id"] in keep:
                keep.update(j["deps"])
        for j in list(jobs):
            if j["id"] in keep:
                keep.update(j["deps"])
        jobs = [j for j in jobs if j["id"] in keep]
    return jobs


def take_lock(runs_dir):
    """Refuse to start a second run_all.py on the same runs_dir (POSIX only)."""
    path = os.path.join(runs_dir, ".run_all.lock")
    if os.name == "posix" and os.path.exists(path):
        try:
            pid = int(open(path).read().strip())
            os.kill(pid, 0)
            raise SystemExit(f"run_all.py is already running on {runs_dir} (pid {pid}).")
        except (ValueError, ProcessLookupError, PermissionError):
            pass
    with open(path, "w") as f:
        f.write(str(os.getpid()))
    atexit.register(lambda: os.path.exists(path) and os.remove(path))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", default=os.path.join(HERE, "plan.json"))
    p.add_argument("--data_dir", default=os.path.abspath(os.path.join(HERE, "..", "data")))
    p.add_argument("--runs_dir", default=None, help="default: runs/ (runs_smoke/ with --smoke)")
    p.add_argument("--analysis_dir", default=None,
                   help="default: analysis/ (analysis_smoke/ with --smoke)")
    p.add_argument("--smoke", action="store_true",
                   help="tiny version of every job (6 epochs, 3 rounds, 2000 images)")
    p.add_argument("--only", default=None, help="run only jobs whose name contains this text")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--retry_wait", type=int, default=60)
    p.add_argument("--gpu", default=None, help="sets CUDA_VISIBLE_DEVICES for the jobs")
    p.add_argument("--gpu_fraction", type=float, default=None,
                   help="share of the GPU to use, e.g. 0.5 on a shared machine")
    p.add_argument("--no_analysis", action="store_true")
    p.add_argument("--n_boot", type=int, default=None)
    p.add_argument("--quiet", action="store_true", help="don't echo job output to the console")
    args = p.parse_args()
    suffix = "_smoke" if args.smoke else ""
    args.runs_dir = os.path.abspath(args.runs_dir or os.path.join(HERE, "runs" + suffix))
    args.analysis_dir = os.path.abspath(args.analysis_dir or os.path.join(HERE, "analysis" + suffix))
    args.data_dir = os.path.abspath(args.data_dir)
    if args.n_boot is None:
        args.n_boot = 20 if args.smoke else 300

    with open(args.plan) as f:
        plan = json.load(f)
    jobs = build_jobs(plan, args)
    if args.dry_run:
        for j in jobs:
            print(f"{j['id']:45s} {'done' if j['done']() else 'todo':5s} {' '.join(j['cmd'][1:])}")
        return

    os.makedirs(args.runs_dir, exist_ok=True)
    take_lock(args.runs_dir)
    r = Runner(args)
    if os.path.exists(r.status_path):
        try:
            r.status = json.load(open(r.status_path)).get("jobs", {})
        except (OSError, ValueError):
            r.status = {}
    r.say(f"run_all.py: {len(jobs)} jobs, runs in {args.runs_dir}, data in {args.data_dir}"
          + (" [SMOKE TEST]" if args.smoke else ""))
    ok = {}
    t_start = time.time()
    try:
        for i, job in enumerate(jobs, 1):
            missing = [d for d in job["deps"] if not ok.get(d, False)]
            if missing:
                r.status[job["id"]] = dict(state="skipped", reason=f"needs {missing}")
                r.save_status()
                r.say(f"SKIP  {job['id']}: a job it needs failed ({', '.join(missing)})")
                ok[job["id"]] = False
                continue
            r.say(f"--- job {i}/{len(jobs)} ---")
            ok[job["id"]] = r.run_job(job)
            if job["kind"] == "imp" and ok[job["id"]]:
                r.analysis()
    except KeyboardInterrupt:
        for st in r.status.values():
            if st.get("state") == "running":
                st["state"] = "interrupted"
        r.save_status()
        r.say("Interrupted. Rerun the same command to continue where it stopped.")
        sys.exit(130)

    r.analysis()
    failed = [j for j, v in ok.items() if not v]
    hours = (time.time() - t_start) / 3600
    r.say(f"ALL FINISHED in {hours:.1f} h: {len(ok) - len(failed)} ok, {len(failed)} failed/skipped"
          + (": " + ", ".join(failed) if failed else ""))
    if failed:
        r.say("Fix the cause (see the logs) and rerun the same command; finished work is kept.")
        sys.exit(1)


if __name__ == "__main__":
    main()
