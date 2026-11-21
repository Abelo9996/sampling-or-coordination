"""GPU energy attribution per model call from `nvidia-smi` power samples.

A background thread runs
  nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -lms 200
and stores (host time, watts) pairs. `joules(t0, t1)` integrates the samples
over [t0, t1] by the trapezoid rule, with the power at the interval ends
interpolated linearly between neighboring samples. Calls must not overlap (the
code-task runner issues one request at a time), otherwise energy would be
double-counted. Design: paper/eaai/CODE_ENERGY_EXPERIMENT.md, section 6.
"""
import bisect
import subprocess
import threading
import time

NVSMI = ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits",
         "-lms", "200"]


def parse_watts(line):
    """'52.31' or '52.31 W' (first GPU on multi-GPU lines) -> float or None."""
    tok = line.strip().split(",")[0].strip().split(" ")[0]
    try:
        return float(tok)
    except ValueError:
        return None   # '[N/A]' or a header


def integrate(ts, ws, t0, t1):
    """Trapezoidal integral of the piecewise-linear power trace over [t0, t1]."""
    if t1 <= t0 or len(ts) < 2 or t0 < ts[0] or t1 > ts[-1]:
        return None

    def at(t):
        i = bisect.bisect_right(ts, t)
        if i <= 0:
            return ws[0]
        if i >= len(ts):
            return ws[-1]
        a, b = ts[i - 1], ts[i]
        return ws[i - 1] + (ws[i] - ws[i - 1]) * (t - a) / (b - a) if b > a else ws[i]

    i0, i1 = bisect.bisect_right(ts, t0), bisect.bisect_left(ts, t1)
    pts = [(t0, at(t0))] + [(ts[i], ws[i]) for i in range(i0, i1)] + [(t1, at(t1))]
    return sum((pts[j + 1][0] - pts[j][0]) * (pts[j + 1][1] + pts[j][1]) / 2
               for j in range(len(pts) - 1))


class PowerSampler:
    def __init__(self, cmd=None):
        self.cmd = list(cmd or NVSMI)
        self.ts, self.ws = [], []
        self._lock = threading.Lock()
        self._proc = None
        self._thr = None

    def start(self):
        self._proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL, text=True,
                                      bufsize=1)
        self._thr = threading.Thread(target=self._read, daemon=True)
        self._thr.start()
        return self

    def _read(self):
        for line in self._proc.stdout:
            w = parse_watts(line)
            if w is not None:
                with self._lock:
                    self.ts.append(time.time())
                    self.ws.append(w)

    def stop(self):
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def wait_past(self, t, timeout=2.0):
        """Block until a sample later than t exists (so [.., t] is covered)."""
        end = time.time() + timeout
        while time.time() < end:
            with self._lock:
                if self.ts and self.ts[-1] >= t:
                    return True
            time.sleep(0.05)
        return False

    def joules(self, t0, t1):
        self.wait_past(t1)
        with self._lock:
            ts, ws = list(self.ts), list(self.ws)
        return integrate(ts, ws, t0, t1)

    def mean_watts(self, t0, t1):
        j = self.joules(t0, t1)
        return None if j is None else j / (t1 - t0)


class NvmlSampler(PowerSampler):
    """Reads the driver's own board-power sample buffer through NVML
    (nvmlDeviceGetSamples, NVML_TOTAL_POWER_SAMPLES; about one sample per 20 ms,
    timestamps in host-epoch microseconds). Used where `power.draw` reports
    [N/A] (driver 610.88 under WSL on the RTX 4060 PC) although the same
    buffer backs the 'Power Samples' block of `nvidia-smi -q -d POWER`. The
    buffer holds about 2.4 s, so it is drained every 0.5 s."""

    POLL_S = 0.5

    def __init__(self):
        super().__init__(cmd=[])
        import ctypes

        class _Sample(ctypes.Structure):
            _fields_ = [("ts", ctypes.c_ulonglong), ("val", ctypes.c_ulonglong)]
        self._ct, self._Sample = ctypes, _Sample
        self._lib = ctypes.CDLL("libnvidia-ml.so.1")
        if self._lib.nvmlInit_v2() != 0:
            raise RuntimeError("nvmlInit failed")
        self._h = ctypes.c_void_p()
        if self._lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self._h)) != 0:
            raise RuntimeError("no NVML device 0")
        self._last = 0
        self._stop = threading.Event()

    def _drain(self):
        ct = self._ct
        vt, n = ct.c_int(), ct.c_uint(0)
        if self._lib.nvmlDeviceGetSamples(self._h, 0, ct.c_ulonglong(self._last),
                                          ct.byref(vt), ct.byref(n), None) != 0 or not n.value:
            return 0
        buf = (self._Sample * n.value)()
        if self._lib.nvmlDeviceGetSamples(self._h, 0, ct.c_ulonglong(self._last),
                                          ct.byref(vt), ct.byref(n), buf) != 0:
            return 0
        new = sorted((buf[i].ts, buf[i].val & 0xffffffff) for i in range(n.value)
                     if buf[i].ts > self._last)
        with self._lock:
            for ts_us, mw in new:
                self.ts.append(ts_us / 1e6)
                self.ws.append(mw / 1000.0)
        if new:
            self._last = new[-1][0]
        return len(new)

    def start(self):
        if not self._drain() and not self._drain():
            raise RuntimeError("NVML returned no power samples")
        self._thr = threading.Thread(target=self._read, daemon=True)
        self._thr.start()
        return self

    def _read(self):
        while not self._stop.is_set():
            self._drain()
            time.sleep(self.POLL_S)

    def stop(self):
        self._stop.set()


def make_sampler():
    """nvidia-smi power.draw when it reports watts, else the NVML sample buffer."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=power.draw",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        out = ""
    if parse_watts(out or "") is not None:
        return PowerSampler()
    return NvmlSampler()
