"""Build scripts/cloud/kaggle_nonthinking.ipynb: the free fallback for the two
non-thinking models on a Kaggle 2x T4 session (fp16, T4 has no bf16).

The notebook embeds the needed code and data as a base64 tar.gz (same file list
as the Modal image), so it needs no repository access. Rebuild after any change
to the harness or the driver:
  python3 scripts/cloud/make_kaggle_notebook.py
"""
import base64
import io
import json
import os
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "kaggle_nonthinking.ipynb")
FILES = ["harness/__init__.py", "harness/client.py", "harness/vllm_client.py",
         "harness/openai_client.py", "harness/tasks.py", "harness/topologies.py",
         "harness/runner.py", "scripts/run_power_campaign.py", "scripts/run_rev3_campaign.py",
         "scripts/rev3_workflows.py", "scripts/extra_workflows.py",
         "scripts/cloud/run_cloud_campaign.py", "scripts/cloud/container_job.py",
         "tasks_data/gsm8k_test.jsonl", "tasks_data/math500.jsonl"]
VLLM_VERSION = "0.9.2"


def payload():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for f in FILES:
            ti = tf.gettarinfo(os.path.join(ROOT, f), arcname=f)
            ti.mtime, ti.uid, ti.gid, ti.uname, ti.gname = 0, 0, 0, "", ""
            with open(os.path.join(ROOT, f), "rb") as fh:
                tf.addfile(ti, fh)
    return base64.b64encode(buf.getvalue()).decode()


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


def build():
    b64 = payload()
    cells = [
        md("# Cross-model replication: non-thinking models on 2x T4 (free fallback)\n\n"
           "Runs the paper's main experiment (GSM8K first 500 test items and MATH-500 levels 1 to 3, "
           "238 items; SA, SC k=8, debate N3 R1, GE R2, HIERFIX M3; seeds 1 to 3; caps 640 and 1024) "
           "on **microsoft/Phi-4-mini-instruct** (GPU 0) and **Qwen/Qwen2.5-3B-Instruct** (GPU 1) in "
           "parallel, unquantized in fp16 (T4 has no bf16), with vLLM " + VLLM_VERSION + ".\n\n"
           "Settings before running: Accelerator **GPU T4 x2**, Internet **on** (needs a phone-verified "
           "account). No token is needed (both models are ungated). Run all cells. Expected time about "
           "3 to 5 h; the session limit is 12 h. Records are written as they complete to "
           "`/kaggle/working/results/<model>/` (same schema as the Modal and Ollama runs) and zipped "
           "at the end; download `results_nonthinking.zip` from the Output tab, unzip into "
           "`results/cloud/` of the repository.\n\n"
           "Resume after an interrupted session: upload the previous zip as a dataset, unzip it to "
           "`/kaggle/working/results/` before the run cell; finished units are skipped."),
        code("# Kaggle's image is Python 3.13; vLLM " + VLLM_VERSION + " needs Python <3.13, so use a 3.11 venv via uv.\n"
             "!pip install -q uv\n"
             "!uv venv -q --python 3.11 /kaggle/py311\n"
             "!uv pip install -q --python /kaggle/py311/bin/python vllm==" + VLLM_VERSION + " transformers==4.53.2 numpy\n"
             "!/kaggle/py311/bin/python -c \"import vllm, sys; print('vllm', vllm.__version__, sys.version)\"\n"
             "!nvidia-smi --query-gpu=name,memory.total --format=csv"),
        code("import base64, io, tarfile, os\n"
             "PAYLOAD = '" + b64 + "'\n"
             "os.makedirs('/kaggle/working/coa', exist_ok=True)\n"
             "tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)), mode='r:gz').extractall('/kaggle/working/coa')\n"
             "print(sorted(os.listdir('/kaggle/working/coa')))"),
        code("import subprocess, time, os, sys\n"
             "HARD = time.time() + 11.3 * 3600  # inside the 12 h session limit\n"
             "JOBS = [('phi4-mini', 'microsoft/Phi-4-mini-instruct', '0', 8000),\n"
             "        ('qwen25-3b', 'Qwen/Qwen2.5-3B-Instruct', '1', 8001)]\n"
             "procs = {}\n"
             "for key, hf, gpu, port in JOBS:\n"
             "    out = f'/kaggle/working/results/{key}'\n"
             "    os.makedirs(out, exist_ok=True)\n"
             "    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, VLLM_USE_V1='0', COA_NO_PREFIX_CACHE='1')\n"
             "    cmd = ['/kaggle/py311/bin/python', '-u', '/kaggle/working/coa/scripts/cloud/container_job.py',\n"
             "           '--hf-id', hf, '--thinking', 'off', '--out-dir', out, '--dtype', 'half',\n"
             "           '--max-model-len', '8192', '--port', str(port), '--max-num-seqs', '64',\n"
             "           '--concurrency', '96', '--host', 'kaggle-t4', '--hard-deadline', str(HARD)]\n"
             "    procs[key] = subprocess.Popen(cmd, env=env, stdout=open(f'{out}/job_stdout.log', 'a'),\n"
             "                                  stderr=subprocess.STDOUT)\n"
             "    print('started', key, procs[key].pid)"),
        code("# Monitor: rerun this cell to refresh (the jobs keep running).\n"
             "import json, glob\n"
             "for p in sorted(glob.glob('/kaggle/working/results/*/progress.json')):\n"
             "    d = json.load(open(p))\n"
             "    print(p.split('/')[-2], d['status'], 'units', d['units_in_files'], '/', d['units_total'],\n"
             "          'tok/s', d['out_tok_per_s'], 'cap_rate', d['hit_cap_rate'], 'eta_s', d['eta_s'])\n"
             "print({k: p.poll() for k, p in procs.items()})"),
        code("# Wait for both jobs, then zip the results for download.\n"
             "for k, p in procs.items():\n"
             "    print(k, 'exit', p.wait())\n"
             "!cd /kaggle/working && zip -qr results_nonthinking.zip results && ls -la results_nonthinking.zip"),
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                      "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    with open(OUT, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")


if __name__ == "__main__":
    build()
