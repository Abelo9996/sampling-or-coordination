"""Cost-matched dominance test and break-even price ratio (pure numpy).

The two procedures of "Sampling or Coordination?" as reusable functions.

cost_matched_dominance
    Compares a workflow A (e.g. multi-agent debate) with every cheaper configuration B of a
    reference family (e.g. single call and self-consistency). For each B whose mean cost is
    strictly lower than A's, it computes the accuracy advantage of B over A with a seed-aware
    hierarchical paired bootstrap (items resampled with replacement, then seeds within each
    resampled item). Output:
        R  robustly dominated: some cheaper B has a 95% interval of its advantage above zero;
        C  cheaper, accuracy not distinguishable: there is a cheaper B, none is robust, and not
           every cheaper B is detectably less accurate;
        N  not dominated: no cheaper B, or every cheaper B is detectably less accurate.
    The rule leans toward finding domination (R needs one cheaper comparator, N needs all).

break_even_price_ratio
    For an output:input price ratio r, the price-weighted cost of a configuration is
    input tokens + r x output tokens per query. The break-even price ratio of a (e.g. SC k=8)
    against b (e.g. debate) is r* = (in_b - in_a) / (out_a - out_b): a is cheaper than b iff
    r < r*. If a generates no more output tokens than b, r* is infinite (a is cheaper at every
    ratio when it also reads fewer input tokens). The 95% interval resamples items, then seeds
    within each item, independently for a and b.

Both use numpy.random.default_rng(seed) with the paper's draw order, so with B=10000, seed=0
and the released logs they return the paper's intervals (tests/test_dominance.py checks this).
The bootstrap of cost_matched_dominance reproduces harness.stats.hier_paired_test draw for draw
(it emulates numpy's bounded-integer stream in bulk, so it is about 100 times faster).

Inputs are per item (rows, the same item order for every configuration) and per seed (columns).
To reproduce the paper's draws, order the rows as the paper's scripts do: sorted item ids for
cost_matched_dominance (harness.stats sorts them), the task loader's order for
break_even_price_ratio.
A configuration run once (e.g. a greedy call) is passed with its value repeated over the seed
axis, as in the paper; this changes only which random numbers are drawn, not the estimate.

    from soc.dominance import cost_matched_dominance, break_even_price_ratio
    v = cost_matched_dominance(acc_debate, tok_debate, {"SC_k4": acc_sc4}, {"SC_k4": tok_sc4})
    r = break_even_price_ratio(in_sc8, out_sc8, in_debate, out_debate)   # r["point"], r["ci95"]
"""
import numpy as np

__all__ = ["cost_matched_dominance", "break_even_price_ratio", "hier_paired_bootstrap"]

_M32 = np.uint64(0xFFFFFFFF)


class _Bounded32:
    """numpy Generator.integers(0, m, size) for m <= 2**32, consumed as one stream.

    Generator.integers draws one 32-bit word per value (Lemire's method, with a rejection that
    redraws when the low word is below (2**32 - m) % m); PCG64 hands out 32-bit words as the low
    then the high half of each 64-bit output, carried across calls. Drawing the words in bulk
    and applying the same rule gives the same values as a sequence of integers() calls.
    """

    def __init__(self, seed):
        self.bg = np.random.default_rng(seed).bit_generator
        self.buf = np.empty(0, np.uint64)

    def _words(self, n):
        while self.buf.size < n:
            raw = self.bg.random_raw(max(n - self.buf.size, 1 << 16) // 2 + 1)
            w = np.empty(2 * raw.size, np.uint64)
            w[0::2] = raw & _M32
            w[1::2] = raw >> np.uint64(32)
            self.buf = np.concatenate([self.buf, w])

    def draw(self, ranges):
        """One value in [0, m) for each m in `ranges` (uint64 array), in order."""
        n = ranges.size
        thr = (np.uint64(1 << 32) - ranges) % ranges
        out = np.empty(n, np.uint64)
        done = 0
        while done < n:
            self._words(n - done)
            r = ranges[done:]
            m = self.buf[: n - done] * r
            bad = np.flatnonzero((m & _M32) < thr[done:])
            stop = bad[0] if bad.size else n - done
            out[done: done + stop] = m[:stop] >> np.uint64(32)
            self.buf = self.buf[stop + (1 if bad.size else 0):]  # a rejected word is discarded
            done += stop
        return out.astype(np.int64)


def _as2d(x):
    x = np.asarray(x, float)
    return x[:, None] if x.ndim == 1 else x


def hier_paired_bootstrap(a, b, B=10000, seed=0):
    """Seed-aware hierarchical paired bootstrap of mean(a) - mean(b), in points (x100).

    a, b: arrays (n_items, n_seeds) of 0/1 correctness (a 1-D array is one seed).
    Returns dict: diff, p (two-sided: 2 x the smaller tail at zero), ci95, boot (B draws).
    Same draws and arithmetic as harness.stats.hier_paired_test(a, b, B, seed).
    """
    A, Bm = _as2d(a), _as2d(b)
    n, sa = A.shape
    sb = Bm.shape[1]
    assert Bm.shape[0] == n, "a and b must have the same items"
    obs = float(np.mean(A.mean(axis=1) - Bm.mean(axis=1)))
    per = n + n * (sa + sb)
    blk = np.concatenate([np.full(sa, sa), np.full(sb, sb)]).astype(np.uint64)
    pattern = np.concatenate([np.full(n, n, np.uint64), np.tile(blk, n)])
    pattern = np.where(pattern == 1, 0, pattern)  # integers(0, 1) draws nothing
    keep = pattern > 0
    gen = _Bounded32(seed)
    boot = np.empty(B)
    chunk = max(1, 2_000_000 // per)
    for t0 in range(0, B, chunk):
        c = min(chunk, B - t0)
        full = np.zeros((c, per), np.int64)
        full[:, keep] = gen.draw(np.tile(pattern[keep], c)).reshape(c, -1)
        idx = full[:, :n]
        sub = full[:, n:].reshape(c, n, sa + sb)
        ia, ib = sub[:, :, :sa], sub[:, :, sa:]
        ai = A[idx[..., None], ia].sum(axis=-1) / sa
        bi = Bm[idx[..., None], ib].sum(axis=-1) / sb
        d = ai - bi
        s = np.zeros(c)
        for j in range(n):  # left-to-right sum, as the reference loop
            s += d[:, j]
        boot[t0: t0 + c] = s / n
    p = min(2.0 * min((boot <= 0).mean(), (boot >= 0).mean()), 1.0)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"diff": 100 * obs, "p": float(p), "ci95": [100 * float(lo), 100 * float(hi)], "boot": 100 * boot}


def _mean_cost(c):
    c = np.asarray(c, float)
    return float(c) if c.ndim == 0 else float(np.mean(c.mean(axis=1) if c.ndim == 2 else c))


def cost_matched_dominance(acc_a, cost_a, acc_b_configs, cost_b_configs, B=10000, seed=0, decimals=2):
    """Cost-matched dominance test of workflow A against reference configurations B.

    acc_a: (n_items, n_seeds) 0/1 correctness of A. cost_a: A's cost per item (n_items,),
    per item and seed (n_items, n_seeds), or its mean cost (scalar), in any unit (total tokens,
    output tokens, price-weighted tokens, joules). acc_b_configs / cost_b_configs: dicts
    name -> the same for each reference configuration.
    Intervals are rounded to `decimals` before the rule is applied, as in the paper (None: no
    rounding).
    Returns dict: verdict ('R', 'C' or 'N'), cheaper (names, cheapest first), robust_by,
    dominator (cheapest robust dominator or None), and per cheaper name: adv (B minus A,
    points), ci95, p (unadjusted), cost_diff (B minus A, mean).
    """
    ca = _mean_cost(cost_a)
    costs = {k: _mean_cost(cost_b_configs[k]) for k in acc_b_configs}
    cheaper = sorted([k for k in acc_b_configs if costs[k] < ca], key=lambda k: costs[k])
    comp = {}
    for k in cheaper:
        t = hier_paired_bootstrap(acc_a, acc_b_configs[k], B=B, seed=seed)  # A minus B, as the paper
        lo, hi = t["ci95"]
        if decimals is not None:
            lo, hi = round(lo, decimals), round(hi, decimals)
        comp[k] = {"adv": -t["diff"], "ci95": [-hi, -lo], "p": t["p"], "cost_diff": costs[k] - ca}
    robust = [k for k in cheaper if comp[k]["ci95"][0] > 0]
    if robust:
        v = "R"
    elif cheaper and all(comp[k]["ci95"][1] < 0 for k in cheaper):
        v = "N"
    elif cheaper:
        v = "C"
    else:
        v = "N"
    return {"verdict": v, "cheaper": cheaper, "robust_by": robust,
            "dominator": robust[0] if robust else None, "comparisons": comp}


def break_even_price_ratio(in_a, out_a, in_b, out_b, B=10000, seed=0):
    """Break-even output:input price ratio r* of configuration a against configuration b.

    in_*, out_*: input and output tokens per query, arrays (n_items, n_seeds) (1-D: one seed).
    a is cheaper than b in price-weighted tokens (input + r x output) iff r < r*.
    Returns dict: point (float, or inf), ci95 ([lo, hi], 'inf' where the percentile interval
    reaches infinity), share_infinite (share of replicates with r* = inf),
    d_in (in_b - in_a, mean), d_out (out_a - out_b, mean).
    Same draws as scripts/analyze_rev3_costs.crossover.
    """
    ia_, oa_, ib_, ob_ = (_as2d(x) for x in (in_a, out_a, in_b, out_b))
    a = np.stack([ia_, oa_], axis=-1)
    b = np.stack([ib_, ob_], axis=-1)
    n, sa, sb = a.shape[0], a.shape[1], b.shape[1]
    assert b.shape[0] == n, "a and b must have the same items"

    def rstar(din, dout):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(dout > 0, din / dout, np.inf)

    ma, mb = a.mean(axis=(0, 1)), b.mean(axis=(0, 1))
    point = float(rstar(mb[0] - ma[0], ma[1] - mb[1]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    ia = rng.integers(0, sa, size=(B, n, sa))
    ib = rng.integers(0, sb, size=(B, n, sb))
    ma_ = a[idx[..., None], ia].mean(axis=(1, 2))
    mb_ = b[idx[..., None], ib].mean(axis=(1, 2))
    rs = rstar(mb_[:, 0] - ma_[:, 0], ma_[:, 1] - mb_[:, 1])
    share_inf = float(np.mean(np.isinf(rs)))
    if share_inf > 0.025:
        ci = [float(np.percentile(rs, 2.5)) if share_inf < 0.975 else "inf", "inf"]
    else:
        lo, hi = np.percentile(rs, [2.5, 97.5])
        ci = [float(lo), float(hi)]
    return {"point": point, "ci95": ci, "share_infinite": share_inf,
            "d_in": float(mb[0] - ma[0]), "d_out": float(ma[1] - mb[1])}
