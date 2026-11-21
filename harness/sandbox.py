"""Sandboxed execution of untrusted (model-generated) Python programs.

Backends (paper/eaai/CODE_ENERGY_EXPERIMENT.md, section 5):
  docker   PC runs: `docker run --rm --network none --memory 512m --cpus 1
           --pids-limit 64 --read-only ... python:3.11-slim timeout T python -I`
  macos    Mac smoke: subprocess under `sandbox-exec` denying all network
           access and all writes under /Users, rlimits, timeout.
  none     refuses to run anything (raises); use for canonical-only checks
           with `allow_unsandboxed=True` only on trusted code.

Select with AGENTS_SANDBOX=docker|macos (default: macos on Darwin if
sandbox-exec exists, else docker if available, else refuse).
"""
import os
import platform
import shutil
import subprocess
import tempfile
import uuid

DOCKER_IMAGE = os.environ.get("AGENTS_SANDBOX_IMAGE", "python:3.11-slim")
TIMEOUT_S = 10
MACOS_PROFILE = ('(version 1)(allow default)(deny network*)'
                 '(deny file-write* (subpath "/Users"))')


def default_backend():
    env = os.environ.get("AGENTS_SANDBOX", "").strip().lower()
    if env:
        return env
    if platform.system() == "Darwin" and shutil.which("sandbox-exec"):
        return "macos"
    if shutil.which("docker"):
        return "docker"
    return "none"


def docker_cmd(workdir, name, timeout=TIMEOUT_S):
    return ["docker", "run", "--rm", "--name", name, "--network", "none",
            "--memory", "512m", "--memory-swap", "512m", "--cpus", "1",
            "--pids-limit", "64", "--read-only", "--tmpfs", "/tmp:size=16m",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", "65534:65534", "-v", f"{workdir}:/w:ro", "-w", "/tmp",
            DOCKER_IMAGE, "timeout", str(int(timeout)), "python", "-I",
            "/w/prog.py"]


def _rlimits():  # child-side limits for the macOS backend
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT_S, TIMEOUT_S + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))


def run_program(code, timeout=TIMEOUT_S, backend=None):
    """Run `code` as a standalone Python program. Returns
    {"ok": exit status 0, "returncode", "timeout": bool, "stdout", "stderr"}
    (stdout/stderr truncated to 2000 characters)."""
    backend = backend or default_backend()
    workdir = tempfile.mkdtemp(prefix="agents_sbx_")
    os.chmod(workdir, 0o755)
    path = os.path.join(workdir, "prog.py")
    with open(path, "w") as f:
        f.write(code)
    os.chmod(path, 0o644)
    name = None
    try:
        if backend == "docker":
            name = "agents-sbx-" + uuid.uuid4().hex[:12]
            cmd, kw, host_to = docker_cmd(workdir, name, timeout), {}, timeout + 20
        elif backend == "macos":
            if not shutil.which("sandbox-exec"):
                raise RuntimeError("sandbox-exec unavailable; refusing to run")
            cmd = ["sandbox-exec", "-p", MACOS_PROFILE, "python3", "-I", path]
            kw, host_to = {"preexec_fn": _rlimits, "cwd": workdir}, timeout
        else:
            raise RuntimeError(f"no sandbox backend ({backend!r}); refusing to run")
        try:
            p = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=host_to, stdin=subprocess.DEVNULL, **kw)
            rc, out, err, to = p.returncode, p.stdout, p.stderr, False
            if backend == "docker" and rc == 124:   # GNU timeout inside
                to = True
        except subprocess.TimeoutExpired as e:
            rc, to = -9, True
            out = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            err = "timeout"
            if name:
                subprocess.run(["docker", "kill", name], capture_output=True)
        return {"ok": rc == 0 and not to, "returncode": rc, "timeout": to,
                "stdout": out[-2000:], "stderr": err[-2000:]}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
