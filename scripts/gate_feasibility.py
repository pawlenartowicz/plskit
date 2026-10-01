#!/usr/bin/env python3
"""Feasibility check for the Gram-route decision gates. Not a CI gate.

A numpy transcription of the n-space PLS1 kernel's gate recursion
(`dual_route::multi_k::pls1_nspace_kernel` in plskit-rs): the
per-component first-order rounding bounds E_w(a) and E_t(a), including the
history term for errors carried from earlier components, the `‖G‖₂` bound
from a fixed-start power iteration, and the five decision gates. It runs
the recursion on ordinary data (uniform X, a signal outcome and 200 of its
permutations) at the block shapes the route serves and reports, per shape
and k, the fraction of replicates that would fall back to the primal
kernel. K_DUAL_MAX is the largest k (at most 10) such that every k up to it
falls back on at most 1% of the replicates at every shape; a block with
fewer than k + 1 rows does not constrain k (eligibility already excludes
it).

Needs numpy: run it in the plskit-py development environment (the one
`pip install . -v` sets up, which installs numpy with the package). CI does
not run it.

Usage (from plskit/):
    python3 scripts/gate_feasibility.py nspace              # table, K_DUAL_MAX
    python3 scripts/gate_feasibility.py nspace --reference  # Rust test constants
    python3 scripts/gate_feasibility.py gram_p                  # Gram-p table, K_GRAM_MAX
    python3 scripts/gate_feasibility.py gram_p --reference      # Rust test block (gram_p)
    python3 scripts/gate_feasibility.py gram_p --ratio-min 1e-5 # after a RATIO_MIN raise

For gram_p the exit code is 0 when K_GRAM_MAX >= 2 and every tall fixture's
pin k is at most K_GRAM_MAX; 2 when K_GRAM_MAX < 2 (the gram_p route would not
be worth enabling); 3 when a tall fixture's pin k exceeds K_GRAM_MAX;
1 when the reference design does not resolve.

Exit status: 0 when k = 2 is feasible on every fixture shape; 2 when it is
not, or when an ordinary shape already drives K_DUAL_MAX below 2
(the n-space route would not be worth enabling); 1 when the reference
design does not resolve; argparse's 2 on usage errors.

The constants below mirror plskit-rs and change together with it;
HISTORY_COEF in particular is `multi_k::HISTORY_COEF`.
"""

import argparse
import sys

import numpy as np

# The gates treat a NaN or an infinity as a failure, as the kernel does, so
# floating-point warnings carry no information here (and some BLAS builds
# raise spurious ones from matmul).
np.seterr(all="ignore")

EPS = float(np.finfo(float).eps)

# Mirrors of plskit-rs constants (change together).
HISTORY_COEF = 8.0  # multi_k::HISTORY_COEF
STEP_WN = 6  # multi_k::STEP_WN
STEP_TT = 10  # multi_k::STEP_TT
POWER_MAX_ITERS = 50  # multi_k::POWER_MAX_ITERS
POWER_REL_TOL = 1e-3  # multi_k::POWER_REL_TOL
RESOLVE_BAND = 4.0  # dual_route::RESOLVE_BAND
BOUND_BAND = RESOLVE_BAND  # multi_k::BOUND_BAND
ABS_BAND = 100.0  # dual_route::ABS_BAND
NIPALS_ABS_FLOOR = 1e-14  # fit::NIPALS_ABS_FLOOR

K_CAP = 10
MAX_FALLBACK_RATE = 0.01
REPLICATES = 200
REFERENCE_K = 4

# (label, block rows, p, fixture). The block is the matrix G is built from:
# the whole standardized X at perm_null, the largest training fold at
# raw_perm, the training half at split_exact.
SHAPES = [
    ("40x2000", 40, 2000, False),
    ("100x20000", 100, 20000, False),
    ("300x4000", 300, 4000, False),
    ("1000x4000", 1000, 4000, False),
    ("fixture perm_null wide n60 d3000", 60, 3000, True),
    ("fixture raw_perm wide k2 fold 48x3000", 48, 3000, True),
    ("fixture split_exact wide k2 half 30x3000", 30, 3000, True),
]


def standardize(x):
    """Columns centered and scaled to unit ddof-0 SD (scale 1 when zero)."""
    m = x.mean(axis=0)
    s = x.std(axis=0)
    s = np.where(s == 0.0, 1.0, s)
    return (x - m) / s


def w_rel_floor(n, d, x_fro, y_norm):
    """fit::w_rel_floor."""
    return max(n, d) * EPS * x_fro * y_norm


def gram_norm_bound(g, x_fro):
    """multi_k::gram_norm_bound."""
    n = g.shape[0]
    fro2 = x_fro * x_fro
    if n == 0:
        return fro2
    v = np.arange(1, n + 1, dtype=float)
    v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(POWER_MAX_ITERS):
        w = g @ v
        lam_new = float(v @ w)
        norm_w = float(np.linalg.norm(w))
        if not norm_w > 0.0:
            lam = lam_new
            break
        done = abs(lam_new - lam) <= POWER_REL_TOL * lam_new
        lam = lam_new
        v = w / norm_w
        if done:
            break
    est = 2.0 * lam
    return est if 0.0 < est < fro2 else fro2


def base_bounds(a, n_tr, p, x_fro):
    """multi_k::nspace_base_bounds (a is 1-based)."""
    fro2 = x_fro * x_fro
    hist = (a - 1) * (n_tr + 2)
    cw = p + 2 * n_tr + STEP_WN * hist
    ct = 2 * p + 3 * n_tr + STEP_TT * hist
    return cw * EPS * fro2, ct * EPS * fro2 * fro2


def kernel_gates(g, x_fro, g2, p, z, k):
    """The kernel's gate recursion for one outcome z.

    Returns (outcome, trace): outcome is "resolved", "zero" or "unresolved";
    trace holds (wn2, e_w, e_t, rho) for every component reached.
    """
    n = g.shape[0]
    if not np.any(z != 0.0):
        return "zero", []
    if not 1 <= k < n:
        return "unresolved", []
    z_norm = float(np.linalg.norm(z))
    zz = z_norm * z_norm
    w_floor = w_rel_floor(n, p, x_fro, z_norm)
    y = z.copy()
    ts, vs, trace = [], [], []
    rho_sum = 0.0
    for a in range(1, k + 1):
        bw, bt = base_bounds(a, n, p, x_fro)
        e_w = bw + HISTORY_COEF * rho_sum * g2
        e_t = bt + HISTORY_COEF * rho_sum * g2 * g2
        r = y.copy()
        for t_i, v_i in reversed(list(zip(ts, vs))):
            r = r - t_i * float(v_i @ r)
        h = g @ r
        wn2 = float(r @ h)
        wn = float(np.sqrt(max(wn2, 0.0)))
        rho = e_w * zz / wn2 if wn2 != 0.0 else np.inf
        trace.append((wn2, e_w, e_t, rho))
        keep_w = (
            wn2 >= BOUND_BAND * e_w * zz
            and wn >= RESOLVE_BAND * w_floor
            and wn >= ABS_BAND * NIPALS_ABS_FLOOR
        )
        if not keep_w:
            return "unresolved", trace
        rho_sum += rho
        gv = h / wn
        t = gv.copy()
        for t_i, v_i in zip(ts, vs):
            t = t - t_i * float(v_i @ t)
        tt = float(t @ t)
        keep_t = tt * wn2 >= BOUND_BAND * e_t * zz and tt >= ABS_BAND * NIPALS_ABS_FLOOR
        if not keep_t:
            return "unresolved", trace
        v = t / tt
        q = float(y @ t) / tt
        y = y - q * t
        ts.append(t)
        vs.append(v)
    return "resolved", trace


def fallback_rates(n, p, seed):
    """Fallback fraction per k over the observed outcome and REPLICATES
    permutations of it, on uniform X with a signal outcome."""
    rng = np.random.default_rng(seed)
    x = standardize(rng.uniform(-1.0, 1.0, (n, p)))
    y = x[:, 0] + rng.uniform(-1.0, 1.0, n)
    z0 = (y - y.mean()) / y.std()
    g = x @ x.T
    x_fro = float(np.linalg.norm(x))
    g2 = gram_norm_bound(g, x_fro)
    outcomes = [z0] + [z0[rng.permutation(n)] for _ in range(REPLICATES)]
    rates = {}
    for k in range(1, min(K_CAP, n - 1) + 1):
        fails = sum(1 for z in outcomes if kernel_gates(g, x_fro, g2, p, z, k)[0] == "unresolved")
        rates[k] = fails / len(outcomes)
    return rates, g2 / (x_fro * x_fro)


def reference_design():
    """The closed-form design that the Rust test
    `history_bounds_match_gate_feasibility_script` rebuilds."""
    n, p = 24, 300
    i = np.arange(n, dtype=float)[:, None]
    j = np.arange(p, dtype=float)[None, :]
    xs = standardize(np.sin((i + 1.0) * (j + 1.0) * 0.37) + 0.5 * np.cos(1.1 * i - 0.7 * j))
    ii = np.arange(n, dtype=float)
    y = np.cos(0.9 * ii + 0.3) + 0.5 * xs[:, 0]
    return xs, (y - y.mean()) / y.std()


def print_reference():
    xs, z = reference_design()
    g = xs @ xs.T
    x_fro = float(np.linalg.norm(xs))
    g2 = gram_norm_bound(g, x_fro)
    outcome, trace = kernel_gates(g, x_fro, g2, xs.shape[1], z, REFERENCE_K)
    if outcome != "resolved":
        print(f"the reference design did not resolve at k = {REFERENCE_K}", file=sys.stderr)
        return 1
    print("/// Printed by `python3 scripts/gate_feasibility.py nspace --reference`.")
    print(f"const SCRIPT_G2: f64 = {g2!r};")
    print("/// `(E_w(a), E_t(a), rho_a)` for `a = 1..=4`.")
    print("const SCRIPT_REFERENCE: [(f64, f64, f64); 4] = [")
    for _, e_w, e_t, rho in trace:
        print(f"    ({e_w!r}, {e_t!r}, {rho!r}),")
    print("];")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Gram-route gate feasibility check (not a CI gate).")
    ap.add_argument("route", choices=["nspace", "gram_p"])
    ap.add_argument("--reference", action="store_true", help="print the Rust test constants")
    ap.add_argument("--ratio-min", type=float, default=GRAM_P_RATIO_MIN)
    args = ap.parse_args()
    if args.route == "gram_p":
        return gram_p_main(args)
    if args.reference:
        return print_reference()
    print(f"HISTORY_COEF = {HISTORY_COEF}; {REPLICATES + 1} outcomes per cell; fallback fraction")
    print(f"{'block':<42} {'g2/fro2':>8}  " + " ".join(f"k={k:<5}" for k in range(1, K_CAP + 1)))
    k_dual_max = K_CAP
    k2_on_fixtures = True
    for seed, (label, n, p, fixture) in enumerate(SHAPES, start=1):
        rates, g2_rel = fallback_rates(n, p, seed)
        cells = " ".join(f"{rates[k]:<7.3f}" if k in rates else f"{'-':<7}" for k in range(1, K_CAP + 1))
        print(f"{label:<42} {g2_rel:>8.3f}  {cells}")
        for k in range(1, K_CAP + 1):
            if k in rates and rates[k] > MAX_FALLBACK_RATE:
                k_dual_max = min(k_dual_max, k - 1)
                break
        if fixture and rates.get(2, 1.0) > MAX_FALLBACK_RATE:
            k2_on_fixtures = False
    print(f"K_DUAL_MAX = {k_dual_max}")
    if not k2_on_fixtures or k_dual_max < 2:
        print("STOP: k = 2 is not feasible on a fixture shape, or an ordinary shape already drives K_DUAL_MAX below 2; report before writing route code.")
        return 2
    return 0


# ---------------------------------------------------------------------------
# p-space Gram backend: a transcription of `gram_p::GramBackend`'s
# decision-gate recursion inside `fit::pls1_component_loop`. Its history is
# carried by the running bounds (`rot_prop`, `d_s`), not by HISTORY_COEF,
# which is the coarser n-space rule (`multi_k::HISTORY_COEF`).

U = EPS / 2.0  # gram_p::U
GRAM_P_RATIO_MIN = 1e-6  # gram_p::RATIO_MIN (change together)
GRAM_P_POWER_MAX_ITERS = 50  # gram_p::POWER_MAX_ITERS
GRAM_P_POWER_REL_TOL = 1e-3  # gram_p::POWER_REL_TOL
GRAM_P_K_CAP = 20

# (label, block rows, p, keep, weighted, tall-fixture pin k or None). The block is
# what C is built from: the full data at perm_null, the largest training
# fold at raw_perm, the training half at split_exact.
GRAM_P_SHAPES = [
    ("tall fixture perm_null n2000 d50", 2000, 50, None, False, 3),
    ("tall fixture perm_null weighted n2000 d50", 2000, 50, None, True, 2),
    ("tall fixture raw_perm fold n_tr1600 d50", 1600, 50, None, False, 2),
    ("tall fixture split_exact half n_train1000 d200 keep10", 1000, 200, 10, False, 1),
    ("memprobe n10000 d500", 10000, 500, None, False, None),
    ("ordinary n500 d20", 500, 20, None, False, None),
    # n much larger than p, the regime the route targets (the calibration
    # sweep's n_much_larger_than_p family).
    ("ordinary n5000 d8", 5000, 8, None, False, None),
]


def c2_norm_estimate(c, x_fro):
    """gram_p::c2_norm_estimate: min(2·λ̂, ‖Xs‖_F²), λ̂ the Rayleigh quotient
    of a power iteration on C from the all-ones start / √p, at most
    GRAM_P_POWER_MAX_ITERS products, stopping once λ̂ moves by at most
    GRAM_P_POWER_REL_TOL of itself. ‖Xs‖_F² when λ̂ is not positive."""
    p = c.shape[0]
    fro2 = x_fro * x_fro
    v = np.full(p, 1.0 / np.sqrt(p))
    lam = 0.0
    for _ in range(GRAM_P_POWER_MAX_ITERS):
        w = c @ v
        lam_new = float(v @ w)
        norm_w = float(np.linalg.norm(w))
        if not norm_w > 0.0:
            lam = lam_new
            break
        v = w * (1.0 / norm_w)
        done = abs(lam_new - lam) <= GRAM_P_POWER_REL_TOL * lam_new
        lam = lam_new
        if done:
            break
    if not lam > 0.0:
        return fro2
    return min(2.0 * lam, fro2)


def selected_norm_and_gap(s, keep):
    """gram_p::selected_norm_and_gap."""
    p = s.shape[0]
    if keep is None or keep >= p:
        return float(np.linalg.norm(s)), np.inf
    mags = np.sort(np.abs(s))[::-1]
    return float(np.sqrt(np.sum(mags[:keep] ** 2))), float(mags[keep - 1] - mags[keep])


def select(s, keep):
    """fit::hard_select_keep: zero all but the keep largest |s|."""
    w = s.copy()
    if keep is not None and keep < w.shape[0]:
        idx = np.argsort(-np.abs(w), kind="stable")
        w[idx[keep:]] = 0.0
    return w


def gram_p_gates(c, x_fro, c2n, n, s0, ys_norm, k, keep, ratio_min):
    """The Gram backend's gates for one replicate, up to k components.

    Returns (fail, trace). fail is None when the replicate resolves (every
    component 1..k clears every gate, or the loop stops at a = 1 on the
    X backend's own floor test), else (a, gate) with gate one of SNorm,
    KeepGap, Tt, Ratio, LateStop. trace holds (tt, d_tt, d_p) per component
    that reached gate_tt.
    """
    p = c.shape[0]
    fro2 = x_fro * x_fro
    w_floor = w_rel_floor(n, p, x_fro, ys_norm)
    floor = max(NIPALS_ABS_FLOOR, w_floor)
    s = s0.copy()
    r_cols, p_cols, trace = [], [], []
    d_s = d_w = r_fro2 = p_fro2 = rot_prop = s1_norm = 0.0
    for a in range(1, k + 1):
        sel_norm, gap = selected_norm_and_gap(s, keep)
        s_norm = float(np.linalg.norm(s))
        if a == 1:
            s1_norm = s_norm
            d_w = 0.0
        else:
            d_w = 2.0 * d_s / sel_norm + (p + 2.0) * U
            band = RESOLVE_BAND * 2.0 * d_s
            s_ok = (
                sel_norm >= band
                and sel_norm >= RESOLVE_BAND * w_floor
                and sel_norm >= ABS_BAND * NIPALS_ABS_FLOOR
            )
            if not (s_ok and gap >= band):
                return (a, "SNorm" if not s_ok else "KeepGap"), trace
        w = select(s, keep)
        nv = float(np.linalg.norm(w))
        if not nv >= floor:
            return (None if a == 1 else (a, "LateStop")), trace
        w = w * (1.0 / nv)
        if r_cols:
            r_mat, p_mat = np.array(r_cols).T, np.array(p_cols).T
            r = w - r_mat @ (p_mat.T @ w)
        else:
            r = w.copy()
        cr = c @ r
        tt = float(r @ cr)
        r_norm = float(np.linalg.norm(r))
        if a == 1:
            d_r = 0.0
        else:
            rho = np.sqrt(r_fro2) * np.sqrt(p_fro2)
            d_r = d_w * (1.0 + rho) + rot_prop + (p + a + 2.0) * U * (1.0 + rho)
        l_tt = (n + 2.0 * p + 4.0) * U * fro2 * r_norm * r_norm
        d_tt = l_tt + c2n * d_r * (2.0 * r_norm + d_r)
        p_norm = float(np.linalg.norm(cr)) / tt
        l_p = (n + p + 2.0) * U * fro2 * r_norm / tt + 2.0 * U * p_norm
        d_p = l_p + c2n * d_r / tt + p_norm * d_tt / tt
        trace.append((tt, d_tt, d_p))
        decided = tt >= RESOLVE_BAND * 2.0 * d_tt and tt >= ABS_BAND * NIPALS_ABS_FLOOR
        if not decided:
            return (a, "Tt"), trace
        if not tt >= ratio_min * fro2 * r_norm * r_norm:
            return (a, "Ratio"), trace
        inv_tt = 1.0 / tt
        pv = cr * inv_tt
        q = float(s @ w) * inv_tt
        qt = abs(q * tt)
        l_qt = (2.0 * n + p + 3.0) * U * ys_norm * x_fro * r_norm + (p + 3.0) * U * s_norm
        d_qt = d_s + s_norm * d_w + s1_norm * d_r + l_qt
        d_s += qt * d_p + p_norm * d_qt + 2.0 * U * (s_norm + qt * p_norm)
        r_fro2 += r_norm * r_norm
        p_fro2 += p_norm * p_norm
        rot_prop += d_r * p_norm + r_norm * d_p
        r_cols.append(r)
        p_cols.append(pv)
        s = s - (q * tt) * pv
    return None, trace


def weighted_standardized(x, y, w):
    """Weighted standardization under mean-one weights (ddof 0), and √w:
    `linalg::standardize_weighted`, `standardize1_weighted` and
    `fit::fit_row_scale` of the normalized weights."""
    w = w * (w.shape[0] / w.sum())
    m = (w[:, None] * x).sum(axis=0) / w.sum()
    s = np.sqrt((w[:, None] * (x - m) ** 2).sum(axis=0) / w.sum())
    s = np.where(s == 0.0, 1.0, s)
    my = float((w * y).sum() / w.sum())
    sy = float(np.sqrt((w * (y - my) ** 2).sum() / w.sum()))
    return (x - m) / s, (y - my) / sy, np.sqrt(w)


def gram_p_first_failures(n, p, keep, weighted, seed, n_rep, k_cap, ratio_min):
    """First failing component per replicate (None when resolved at k_cap)
    over a signal outcome and n_rep permutations of it, on uniform X."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, (n, p))
    y = x @ (1.0 / (np.arange(p) + 1.0)) + rng.uniform(-1.0, 1.0, n)
    perms = [np.arange(n)] + [rng.permutation(n) for _ in range(n_rep)]
    if weighted:
        w = np.where(np.arange(n) % 9 == 0, 0.0, 0.5 + (np.arange(n) % 5) * 0.3)
        xs_std, ys_std, sq = weighted_standardized(x, y, w)
        xs = xs_std * sq[:, None]
    else:
        xs = standardize(x)
        ys_std = standardize(y[:, None])[:, 0]
        sq = np.ones(n)
    c = xs.T @ xs
    x_fro = float(np.linalg.norm(xs))
    c2n = c2_norm_estimate(c, x_fro)
    fails = []
    for perm in perms:
        # The standardized y is permuted, then √w-scaled with the unpermuted
        # weights (perm_null's order; √w = 1 when unweighted).
        ys = ys_std[perm] * sq
        fail, _ = gram_p_gates(
            c, x_fro, c2n, n, xs.T @ ys, float(np.linalg.norm(ys)), k_cap, keep, ratio_min
        )
        fails.append(fail)
    return fails, c2n / (x_fro * x_fro)


def gram_p_reference_design():
    """The closed-form design the Rust test
    `gram_p::tests_kernel::bounds_match_gate_feasibility_script` rebuilds."""
    n, p = 400, 12
    i = np.arange(n, dtype=float)[:, None]
    j = np.arange(p, dtype=float)[None, :]
    x = np.sin((i + 1.0) * (j + 1.0) * 0.37) + 0.5 * np.cos(1.1 * i - 0.7 * j)
    xs = standardize(x)
    ii = np.arange(n, dtype=float)
    y = np.cos(0.9 * ii + 0.3) + 0.5 * xs[:, 0]
    ys = (y - y.mean()) / y.std()
    return xs, ys


def gram_p_print_reference(ratio_min):
    xs, ys = gram_p_reference_design()
    n = xs.shape[0]
    c = xs.T @ xs
    x_fro = float(np.linalg.norm(xs))
    c2n = c2_norm_estimate(c, x_fro)
    fail, trace = gram_p_gates(
        c, x_fro, c2n, n, xs.T @ ys, float(np.linalg.norm(ys)), 3, None, ratio_min
    )
    if fail is not None or len(trace) != 3:
        print(f"reference design did not resolve at k = 3: {fail}", file=sys.stderr)
        return 1
    print("/// Printed by `python3 scripts/gate_feasibility.py gram_p --reference`:")
    print("/// `(tt_a, δtt_a, δp_a)` for a = 1..=3 on the reference design, with the")
    print("/// ‖C‖₂ estimate.")
    print(f"const SCRIPT_C2_NORM: f64 = {float(c2n)!r};")
    print("const SCRIPT_REFERENCE: &[(f64, f64, f64)] = &[")
    for tt, d_tt, d_p in trace:
        print(f"    ({float(tt)!r}, {float(d_tt)!r}, {float(d_p)!r}),")
    print("];")
    return 0


def gram_p_main(args):
    """`gram_p` route: fallback table, K_GRAM_MAX, the tall-fixture pin check."""
    if args.reference:
        return gram_p_print_reference(args.ratio_min)
    n_rep = REPLICATES
    print(f"RATIO_MIN = {args.ratio_min:g}, replicates per cell = {n_rep + 1}")
    print(f"{'shape':<46} {'c2/fro2':>8}  first failures (a, gate): count")
    k_gram_max = GRAM_P_K_CAP
    pins = []
    for seed, (label, n, p, keep, weighted, pin_k) in enumerate(GRAM_P_SHAPES, start=101):
        k_cap = min(GRAM_P_K_CAP, p)
        fails, c2_rel = gram_p_first_failures(
            n, p, keep, weighted, seed, n_rep, k_cap, args.ratio_min
        )
        total = len(fails)
        counts = {}
        for f in fails:
            if f is not None:
                counts[f] = counts.get(f, 0) + 1
        summary = ", ".join(f"{a}{g}:{m}" for (a, g), m in sorted(counts.items())) or "none"
        print(f"{label:<46} {c2_rel:>8.4f}  {summary}")
        rates = []
        for k in range(1, k_cap + 1):
            rates.append(sum(1 for f in fails if f is not None and f[0] <= k) / total)
        print(f"{'':<46} {'':>8}  rate by k: " + " ".join(f"{r:.3f}" for r in rates))
        k_ok = 0
        for k, rate in enumerate(rates, start=1):
            if rate > MAX_FALLBACK_RATE:
                break
            k_ok = k
        if k_ok == k_cap:
            # p < GRAM_P_K_CAP caps k by the shape, not by the gates.
            k_ok = GRAM_P_K_CAP
        k_gram_max = min(k_gram_max, k_ok)
        if pin_k is not None:
            pins.append((label, pin_k))
    print(f"K_GRAM_MAX = {k_gram_max}")
    over = [(label, k) for label, k in pins if k > k_gram_max]
    for label, k in over:
        print(f"PIN ABOVE K_GRAM_MAX: {label} runs at k = {k}")
    if k_gram_max < 2:
        print("STOP: K_GRAM_MAX < 2; report before writing any gram_p code.")
        return 2
    if over:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
