"""Overnight job queue (runs unattended).

Lanes:
  mps : Douban-Book jobs, at most 2 at a time, after the running Douban jobs (WAIT_DOUBAN pid) end.
  cpu : LastFM jobs on CPU (nice 10, 2 threads), one at a time, after the LastFM diagnostics
        (WAIT_LASTFM pid) end.
Every job runs as a subprocess with its own log in results/logs/overnight_<name>.log. A failing
job is logged and the queue continues; jobs that depend on a failed or skipped job are skipped.
Exit code 3 = |Omega| too large, 4 = condition not met (both mean "skip dependents").
Status lines go to results/overnight_status.log. At the end the summary script runs.

Usage: nohup .venv/bin/python -u scripts/overnight_queue.py --wait-douban PID --wait-lastfm PID &
       caffeinate -i -w <queue pid> &
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY = os.path.join(ROOT, ".venv/bin/python")
STATUS = os.path.join(ROOT, "results/overnight_status.log")
LOGS = os.path.join(ROOT, "results/logs")
REASON = {3: "|Omega| estimate above the 50M limit", 4: "condition not met", 5: "missing inputs"}


def status(msg):
    line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with open(STATUS, "a") as f:
        f.write(line + "\n")


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def run_exp(cfg, *sets, device="mps", extra=()):
    return [PY, "-u", "scripts/run_experiment.py", cfg, "--quiet", "--device", device, *extra, "--set", *sets]


DB = ["dataset=douban_book", "data_dir=data/processed/douban_book", "reg=0.0001", "eval_every=10"]
LF_INJ = ["dataset=lastfm_inject40", "data_dir=data/processed/lastfm_inject40"]


def jobs(smoke=False):
    """(name, lane, cmd, deps). With smoke=True every job is a 1-epoch LastFM seed-0 CPU run."""
    one = ["max_epochs=1", "eval_every=1", "min_epochs=0", "results_csv=results/smoke.csv"] if smoke else []
    db = (["dataset=lastfm", "data_dir=data/processed/lastfm"] if smoke else DB)
    dev = "cpu" if smoke else "mps"
    pre = "smoke_" if smoke else ""
    J = [
        # ---------------- Douban-Book (seed 0) ----------------
        ("db_prep", "mps",
         [PY, "-u", "scripts/prepare_pugib.py", "--dataset", "lastfm_smoke" if smoke else "douban_book",
          "--variants", "best", "--seeds", "0"], []),
        ("db_gbsr", "mps", run_exp("configs/gbsr/lastfm_gbsr.yaml", *db, f"name={pre}douban_gbsr", "beta=40", "sigma=2.5",
                                   "detach_mask=false", *one, device=dev, extra=["--no-tune", "--seeds", "0"]), []),
        ("db_pugib_anchor", "mps", run_exp("configs/pugib/pugib_s_anchor.yaml", *db, f"name={pre}douban_pugib_s_anchor",
                                           "beta=0.01", "kappa=0.05", "norm=prior", *one, device=dev,
                                           extra=["--no-tune", "--seeds", "0"]), ["db_prep"]),
        ("db_lgcns_omega025", "mps", run_exp("configs/pugib/lightgcn_s_warm.yaml", *db,
                                             f"name={pre}douban_lightgcn_s_warm_omega025", "omega_s=0.25", *one,
                                             device=dev, extra=["--seeds", "0"]), []),
        ("db_lgcnpu_s", "mps", run_exp("configs/pugib/lightgcn_pu_s.yaml", *db, f"name={pre}douban_lightgcn_pu_s", *one,
                                       device=dev, extra=["--seeds", "0"]), ["db_prep"]),
        ("db_pugib_unif", "mps", run_exp("configs/pugib/pugib_s_unif.yaml", *db, f"name={pre}douban_pugib_s_unif",
                                         "beta=0.01", "kappa=0.05", "norm=prior", *one, device=dev,
                                         extra=["--no-tune", "--seeds", "0"]), ["db_prep"]),
        # ---------------- LastFM on CPU (seed 0) ----------------
        ("lf_pugib_beta0", "cpu", run_exp("configs/pugib/pugib_s_anchor.yaml", f"name={pre}pugib_s_anchor_beta0",
                                          "beta=0", "kappa=0.1", "norm=prior", *one, device="cpu",
                                          extra=["--no-tune", "--seeds", "0"]), []),
        ("lf_pugib_beta0001", "cpu", run_exp("configs/pugib/pugib_s_anchor.yaml", f"name={pre}pugib_s_anchor_beta0001",
                                             "beta=0.001", "kappa=0.1", "norm=prior", *one, device="cpu",
                                             extra=["--no-tune", "--seeds", "0"]), []),
        ("lf_decide", "cpu", [PY, "-u", "scripts/check_corruption_drop.py"], []),
        ("lf_inj_prep", "cpu", ["bash", "-c",
                                f"{PY} -u scripts/make_corrupt_dataset.py --src lastfm --seed 0 "
                                f"--edges data/processed/lastfm/seed_0/corrupt/inject40.txt --name {pre}lastfm_inject40 && "
                                f"{PY} -u scripts/prepare_pugib.py --dataset {pre}lastfm_inject40 --variants best --seeds 0"],
         ["lf_decide"]),
        ("lf_inj_pugib", "cpu", run_exp("configs/pugib/pugib_s_anchor.yaml", f"dataset={pre}lastfm_inject40",
                                        f"data_dir=data/processed/{pre}lastfm_inject40", f"name={pre}pugib_inj40_anchor",
                                        "beta=0.001", "kappa=0.1", "norm=prior", *one, device="cpu",
                                        extra=["--no-tune", "--seeds", "0"]), ["lf_inj_prep"]),
        ("lf_inj_gbsr", "cpu", run_exp("configs/gbsr/lastfm_gbsr.yaml", f"dataset={pre}lastfm_inject40",
                                       f"data_dir=data/processed/{pre}lastfm_inject40", f"name={pre}gbsr_inj40",
                                       "detach_mask=false", *one, device="cpu", extra=["--no-tune", "--seeds", "0"]),
         ["lf_inj_prep"]),
        ("lf_inj_auroc", "cpu", [PY, "-u", "scripts/auroc_injected.py", "--dataset", f"{pre}lastfm_inject40",
                                 "--seed", "0", "--run", f"{pre}pugib_inj40_anchor", f"{pre}gbsr_inj40"],
         ["lf_inj_pugib", "lf_inj_gbsr"]),
    ]
    return J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-douban", type=int, default=0)
    ap.add_argument("--wait-lastfm", type=int, default=0)
    ap.add_argument("--max-mps", type=int, default=2)
    ap.add_argument("--smoke", action="store_true", help="run every job for 1 epoch on LastFM seed 0, CPU")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    os.chdir(ROOT)
    os.makedirs(LOGS, exist_ok=True)
    J = jobs(args.smoke)
    if args.only:
        J = [j for j in J if j[0] in args.only]
        keep = {j[0] for j in J}
        J = [(n, ln, c, [d for d in deps if d in keep]) for n, ln, c, deps in J]  # forced subset
    state = {name: "pending" for name, *_ in J}  # pending/running/ok/failed/skipped
    procs = {}
    status(f"QUEUE start pid={os.getpid()} smoke={args.smoke} jobs={[j[0] for j in J]}")
    gate = {"mps": lambda: not alive(args.wait_douban), "cpu": lambda: not alive(args.wait_lastfm)}
    if args.smoke:
        gate = {"mps": lambda: True, "cpu": lambda: True}
        args.max_mps = 1  # smoke: everything on CPU, sequential
    opened = {"mps": False, "cpu": False}
    while True:
        # collect finished jobs
        for name, (p, lane, logf, t0) in list(procs.items()):
            rc = p.poll()
            if rc is None:
                continue
            logf.close()
            dur = time.time() - t0
            if rc == 0:
                state[name] = "ok"
                status(f"FINISHED {name} ({dur / 60:.1f} min)")
            elif rc in REASON:
                state[name] = f"skip:{rc}"
                status(f"SKIPPED {name}: {REASON[rc]} (exit {rc}); dependents will be skipped")
            else:
                state[name] = "failed"
                tail = open(os.path.join(LOGS, f"overnight_{name}.log")).read().strip().splitlines()[-3:]
                status(f"FAILED {name} exit {rc} after {dur / 60:.1f} min: {' | '.join(tail)[-400:]}")
            del procs[name]
        # skip jobs whose dependencies did not succeed
        for name, lane, cmd, deps in J:
            if state[name] == "pending" and any(state[d] not in ("pending", "running", "ok") for d in deps):
                bad = [d for d in deps if state[d] not in ("pending", "running", "ok")]
                state[name] = "skipped"
                status(f"SKIPPED {name}: dependency {bad} did not succeed ({[state[d] for d in bad]})")
        # launch
        for lane in ("mps", "cpu"):
            if not gate[lane]():
                continue
            if not opened[lane]:
                opened[lane] = True
                status(f"LANE {lane} open")
            cap = args.max_mps if lane == "mps" else 1
            if args.smoke:  # one job at a time overall
                cap, running = 1, len(procs)
            else:
                running = sum(1 for _, (p, ln, *_r) in procs.items() if ln == lane)
            for name, ln, cmd, deps in J:
                if running >= cap:
                    break
                if ln != lane or state[name] != "pending" or any(state[d] != "ok" for d in deps):
                    continue
                if lane == "mps" and name != "db_prep" and not args.smoke:
                    ck = os.path.join(ROOT, "results/checkpoints/lightgcn_douban_book_seed0.pt")
                    if not os.path.exists(ck):
                        state[name] = "skipped"
                        status(f"SKIPPED {name}: Douban LightGCN checkpoint missing ({ck})")
                        continue
                env = dict(os.environ)
                full = cmd
                if lane == "cpu" or args.smoke:
                    env["OMP_NUM_THREADS"] = "2"
                    full = ["nice", "-n", "10"] + cmd
                logf = open(os.path.join(LOGS, f"overnight_{name}.log"), "w")
                p = subprocess.Popen(full, stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
                procs[name] = (p, lane, logf, time.time())
                state[name] = "running"
                running += 1
                status(f"STARTED {name} (pid {p.pid}) lane={lane}: {' '.join(cmd[2:] if cmd[0] == PY else cmd)[:300]}")
        if not procs and all(s != "pending" for s in state.values()):
            break
        time.sleep(5 if args.smoke else 30)
    status(f"QUEUE jobs done: {state}")
    if not args.smoke:
        rc = subprocess.call([PY, "scripts/overnight_summary.py"], stdout=open(os.path.join(LOGS, "overnight_summary.log"), "w"),
                             stderr=subprocess.STDOUT, cwd=ROOT)
        status(f"SUMMARY written (exit {rc}): results/overnight_summary.md")
    status("QUEUE end")


if __name__ == "__main__":
    main()
