"""Plan, status, pull and stop for the Modal run. Starts no container and no GPU.

  python3 scripts/cloud/modal_tools.py plan     # month spend + budget plan
  python3 scripts/cloud/modal_tools.py status   # ledger, progress per model, spend
  python3 scripts/cloud/modal_tools.py pull     # Volume runs/<model>/ -> results/cloud/<model>/
  python3 scripts/cloud/modal_tools.py stop     # stop every running coa-xmodel app

Spend this month = max(Modal billing report for the month, sum of the run ledger
on the Volume). The billing report lags by up to an hour; the ledger is written
every minute by each container with its elapsed time at the maximum billed rate,
so the max of the two is an upper estimate for this app.
"""
import datetime as dt
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import budget as B  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
LOCAL = os.path.join(ROOT, "results", "cloud")
APP_NAME = "coa-xmodel"
VOL_NAME = "coa-xmodel-results"


def _vol():
    import modal
    return modal.Volume.from_name(VOL_NAME, create_if_missing=True)


def _read(vol, path):
    return b"".join(vol.read_file(path))


def ledger():
    vol = _vol()
    try:
        ents = vol.listdir("/ledger")
    except Exception:  # noqa: BLE001
        return []
    out = []
    for e in ents:
        if e.path.endswith(".json"):
            try:
                out.append(json.loads(_read(vol, e.path)))
            except Exception:  # noqa: BLE001
                pass
    return out


def _report(start, end=None, resolution="d"):
    try:  # modal >= 1.4: Workspace.billing.report
        import modal
        return list(modal.Workspace.from_context().billing.report(start=start, end=end, resolution=resolution))
    except (AttributeError, TypeError):
        import modal.billing as mb
        return list(mb.workspace_billing_report(start=start, end=end, resolution=resolution))


def billing_month_usd():
    now = dt.datetime.now(dt.timezone.utc)
    start = dt.datetime(now.year, now.month, 1, tzinfo=dt.timezone.utc)
    # daily rows up to today, hourly rows for today (hourly reports span at most 7 days)
    today = dt.datetime(now.year, now.month, now.day, tzinfo=dt.timezone.utc)
    rows = _report(start, today, "d") if today > start else []
    rows += _report(today, None, "h")
    tot = 0.0
    for r in rows:
        c = r.get("cost") if isinstance(r, dict) else getattr(r, "cost", 0)
        tot += float(c or 0)
    return tot


def spent_estimate():
    led = sum(float(x.get("usd", 0)) for x in ledger()
              if str(x.get("run_id", ""))[:6] == dt.datetime.now(dt.timezone.utc).strftime("%Y%m"))
    try:
        bill = billing_month_usd()
        src = f"billing ${bill:.2f}, ledger ${led:.2f}"
    except Exception as e:  # noqa: BLE001
        bill = None
        src = f"billing unavailable ({type(e).__name__}), ledger ${led:.2f}"
    return {"usd": max(led, bill or 0.0), "billing": bill, "ledger": led, "source": src}


def pull(quiet=False):
    """Copy every file under runs/ whose size differs locally."""
    vol = _vol()
    try:
        ents = vol.listdir("/runs", recursive=True)
    except Exception:  # noqa: BLE001
        return 0
    n = 0
    for e in ents:
        if int(getattr(e.type, "value", e.type)) != 1:  # FILE
            continue
        rel = e.path.split("runs/", 1)[1]
        dst = os.path.join(LOCAL, rel)
        if os.path.exists(dst) and os.path.getsize(dst) == e.size:
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".part"
        with open(tmp, "wb") as f:
            for chunk in vol.read_file(e.path):
                f.write(chunk)
        os.replace(tmp, dst)
        n += 1
    if not quiet:
        print(f"pulled {n} files to {LOCAL}")
    return n


def status():
    sp = spent_estimate()
    print(f"spend this month (upper est.): ${sp['usd']:.2f} [{sp['source']}] cap ${B.BUDGET_USD:.2f}")
    for x in sorted(ledger(), key=lambda r: r.get("start", 0)):
        print(f"  ledger {x['run_id']} {x['key']:16s} {x['status']:14s} "
              f"{x['elapsed_s'] / 3600:.2f} h  ${x['usd']:.2f}")
    pull(quiet=True)
    if os.path.isdir(LOCAL):
        for m in sorted(os.listdir(LOCAL)):
            p = os.path.join(LOCAL, m, "progress.json")
            if os.path.exists(p):
                d = json.load(open(p))
                print(f"  {m:16s} {d['status']:14s} units {d['units_in_files']}/{d['units_total']} "
                      f"seeds_complete {d['seeds_complete']} tok/s {d['out_tok_per_s']} "
                      f"cap_rate {d['hit_cap_rate']} eta_s {d['eta_s']} ({d['updated']})")


def stop():
    out = subprocess.run([sys.executable, "-m", "modal", "app", "list", "--json"],
                         capture_output=True, text=True).stdout
    try:
        apps = json.loads(out)
    except json.JSONDecodeError:
        print(out)
        return
    for a in apps:
        desc = a.get("Description") or a.get("description") or a.get("Name") or ""
        state = (a.get("State") or a.get("state") or "").lower()
        aid = a.get("App ID") or a.get("app_id")
        if APP_NAME in desc and "stopped" not in state:
            print("stopping", aid, desc, state)
            subprocess.run([sys.executable, "-m", "modal", "app", "stop", aid], check=False)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "plan":
        sp = spent_estimate()
        print(sp["source"])
        print(json.dumps(B.report(sp["usd"]), indent=1))
    elif cmd == "status":
        status()
    elif cmd == "pull":
        pull()
    elif cmd == "stop":
        stop()
    else:
        raise SystemExit(__doc__)
