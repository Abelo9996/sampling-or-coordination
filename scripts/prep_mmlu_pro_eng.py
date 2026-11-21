"""Build tasks_data/mmlu_pro_engineering.jsonl from the public MMLU-Pro test
split (TIGER-Lab/MMLU-Pro, Wang et al. 2024, arXiv:2406.01574).

Steps: download the test parquet at a pinned Hugging Face revision into a
separate download directory, keep rows with category == "engineering", sort by
question_id, draw a simple random sample of n (default 150) with
random.Random(seed) (default seed 0), and write the sample sorted by
question_id. A sidecar .meta.json records the revision, the parquet SHA-256,
the pool size, and the option-count and source histograms, so the sample can
be re-derived and audited. The raw parquet is not written into the repo.

Requires pyarrow (the macOS system Python /usr/bin/python3 has it; the
analysis .venv does not). Downloaded data is untrusted: run with -I.

  /usr/bin/python3 -I scripts/prep_mmlu_pro_eng.py --download-dir <new empty dir>
"""
import argparse
import collections
import hashlib
import json
import os
import random
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "TIGER-Lab/MMLU-Pro"
# Pinned dataset revision (HF commit sha, read from the HF API on 2026-10-11).
REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
PARQUET = "data/test-00000-of-00001.parquet"
LETTERS = "ABCDEFGHIJ"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(revision, ddir):
    os.makedirs(ddir, exist_ok=True)
    dst = os.path.join(ddir, f"mmlu_pro_test_{revision[:12]}.parquet")
    if not os.path.exists(dst):
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{PARQUET}"
        print("downloading", url, flush=True)
        tmp = dst + ".part"
        with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
        os.replace(tmp, dst)
    return dst


def load_rows(path):
    try:
        import pyarrow.parquet as pq
    except ImportError:
        # -I drops the user site-packages, where pyarrow often lives; re-add
        # only that (trusted) directory, never the cwd or the download dir.
        import site
        sys.path.append(site.getusersitepackages())
        import pyarrow.parquet as pq
    cols = ["question_id", "question", "options", "answer", "answer_index",
            "category", "src"]
    return pq.read_table(path, columns=cols).to_pylist()


def clean(r):
    """Validate one row and return the record written to the jsonl."""
    opts = [str(o) for o in r["options"]]
    if any(o.strip() in ("", "N/A") for o in opts):
        raise ValueError(f"{r['question_id']}: empty or N/A option")
    k = len(opts)
    ai = int(r["answer_index"])
    if not (1 <= k <= len(LETTERS)) or not (0 <= ai < k):
        raise ValueError(f"{r['question_id']}: k={k} answer_index={ai}")
    if r["answer"] != LETTERS[ai]:
        raise ValueError(f"{r['question_id']}: answer {r['answer']} != index {ai}")
    return {"question_id": int(r["question_id"]), "question": r["question"],
            "options": opts, "answer": r["answer"], "answer_index": ai,
            "category": r["category"], "src": r["src"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-dir", required=True,
                    help="new, empty directory for the raw parquet (outside the repo)")
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--category", default="engineering")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(
        ROOT, "tasks_data", "mmlu_pro_engineering.jsonl"))
    args = ap.parse_args()

    path = download(args.revision, os.path.abspath(args.download_dir))
    digest = sha256(path)
    rows = [r for r in load_rows(path) if r["category"] == args.category]
    pool = sorted((clean(r) for r in rows), key=lambda d: d["question_id"])
    if len(pool) < args.n:
        sys.exit(f"pool has {len(pool)} < n={args.n} items")
    sample = random.Random(args.seed).sample(pool, args.n)
    sample.sort(key=lambda d: d["question_id"])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for d in sample:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def hist(xs):
        return dict(sorted(collections.Counter(xs).items()))

    meta = {
        "dataset": REPO, "revision": args.revision, "file": PARQUET,
        "parquet_sha256": digest, "category": args.category,
        "pool_size": len(pool), "n": args.n, "seed": args.seed,
        "sampler": "random.Random(seed).sample(pool sorted by question_id, n); "
                   "output sorted by question_id",
        "pool_option_counts": hist(len(d["options"]) for d in pool),
        "sample_option_counts": hist(len(d["options"]) for d in sample),
        "sample_gold_letters": hist(d["answer"] for d in sample),
        "sample_src": hist(d["src"] for d in sample),
        "out_sha256": sha256(args.out),
    }
    mpath = os.path.splitext(args.out)[0] + ".meta.json"
    with open(mpath, "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print(json.dumps({k: meta[k] for k in ("pool_size", "n", "seed",
                                           "sample_option_counts",
                                           "parquet_sha256")}, indent=1))
    print("wrote", args.out, "and", mpath)


if __name__ == "__main__":
    main()
