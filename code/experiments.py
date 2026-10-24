"""Reproduces every numerical result and figure in the paper.

Usage: python experiments.py [exp1 exp2 exp3 exp4 exp5]   (default: all)
Results are written to ../results/*.json and figures to ../figures/*.pdf.
"""
import itertools
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import networkx as nx

from agentnet import (make_graph, laplacian, directed_versions, influence_matrix,
                      driver_count_no_memory, driver_count_with_memory,
                      exact_driver_count, uncontrolled_modes, energy_stats,
                      pick_pinned, run_proportional, run_sdre, mu_extremes,
                      harmonic_capture)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
os.makedirs(RES, exist_ok=True)
N = 64
TOPOS = ["random", "small-world", "scale-free"]


def save(name, obj):
    with open(os.path.join(RES, name + ".json"), "w") as f:
        json.dump(obj, f, indent=1, default=float)


# ---------------------------------------------------------------- Exp 1
def exp1():
    """Driver agents: structural (without / with agent memory) and exact (identical agents)."""
    rows = []
    for name, dg in directed_versions(N, seed=0).items():
        a = influence_matrix(dg)
        L = np.diag(a.sum(1)) - a
        A = np.eye(len(a)) - 0.05 * L
        rows.append(dict(graph=name, n=dg.number_of_nodes(), edges=dg.number_of_edges(),
                         nd_no_memory=driver_count_no_memory(dg),
                         nd_memory=driver_count_with_memory(dg),
                         nd_exact_identical=exact_driver_count(A)))
    # exact counts over seeds for undirected debate graphs
    seeds = {}
    for kind in ["ring", "small-world", "random", "scale-free", "star", "complete"]:
        vals = []
        for s in range(10):
            g = make_graph(kind, N, s)
            L, _ = laplacian(g)
            vals.append(exact_driver_count(np.eye(N) - 0.05 * L))
        seeds[kind] = vals
    save("exp1_drivers", dict(rows=rows, exact_over_seeds=seeds))
    for r in rows:
        print(r)
    print(seeds)


# ---------------------------------------------------------------- Exp 2
def _exp2_one(args):
    kind, seed, rule, k, T = args
    g = make_graph(kind, N, seed)
    L, _ = laplacian(g)
    lamN = np.linalg.eigvalsh(L)[-1]
    c = min(0.05, 1.5 / lamN)
    A = np.eye(N) - c * L
    P = pick_pinned(g, L, k, rule, c=c, gain=0.5, seed=seed)
    B = np.eye(N)[:, P]
    lmin, avg = energy_stats(A, B, T)
    lmax = np.linalg.eigvalsh(sum(np.linalg.matrix_power(A, t) @ B @ B.T
                                  @ np.linalg.matrix_power(A, t) for t in range(T)))[-1]
    return dict(kind=kind, seed=seed, rule=rule, k=k, T=T, c=c,
                lmin=lmin, lmax=lmax, cond=lmax / max(lmin, 1e-300),
                avg_energy=avg, bound=N / (k * T),
                unreached=uncontrolled_modes(A, P))


def exp2():
    """Control energy of driving the collective, as a function of drivers k."""
    ks = [1, 2, 4, 8, 16, 32, 48, 64]
    jobs = [(kind, s, rule, k, 30) for kind in TOPOS + ["star"] for s in range(4)
            for rule in ["random", "degree", "spectral-greedy"] for k in ks]
    with Pool(4) as p:
        out = p.map(_exp2_one, jobs)
    save("exp2_energy", out)


# ---------------------------------------------------------------- Exp 3
ALPHA, GAIN, UMAX = 0.02, 0.5, 0.5


def _exp3_one(args):
    kind, seed, rule, k, c, ctrl = args
    g = make_graph(kind, N, seed)
    L, Aj = laplacian(g)
    P = pick_pinned(g, L, k, rule, c=0.05, gain=GAIN, seed=seed)
    rng = np.random.default_rng(100 + seed)
    e0 = 1.0 + 0.5 * rng.standard_normal(N)
    if ctrl == "proportional":
        rounds, energy, _ = run_proportional(e0, Aj, ALPHA, c, GAIN, P, T=1000, umax=UMAX)
    else:
        rounds, energy, _ = run_sdre(e0, Aj, ALPHA, c, P, q=1.0, r=1.0, T=1000, umax=UMAX)
    mu1, muN = mu_extremes(L, c, GAIN, P)
    return dict(kind=kind, seed=seed, rule=rule, k=k, c=c, ctrl=ctrl, rounds=rounds,
                energy=energy, mu1=mu1, muN=muN, linear_ok=bool(mu1 > ALPHA and muN < 2 + ALPHA))


def exp3():
    """SDRE versus proportional pinning under bounded influence and bounded verifier input."""
    cs = [0.01, 0.02, 0.04, 0.08, 0.16]
    jobs = [(kind, s, rule, k, c, ctrl) for kind in TOPOS for s in range(2)
            for rule in ["degree", "spectral-greedy"] for k in [4] for c in cs
            for ctrl in ["proportional", "sdre"]]
    with Pool(4) as p:
        out = p.map(_exp3_one, jobs, chunksize=1)
    save("exp3_sdre", out)


# ---------------------------------------------------------------- Exp 4
def guaranteed_radius(L, c, gain, P, alpha):
    """Largest z with alpha < mu_1(c w(z) L + g P), w(z) = tanh(z)/z (Theorem 3)."""
    lo, hi = 1e-6, 50.0
    def ok(z):
        w = np.tanh(z) / z
        return mu_extremes(L, c, gain, P, w=w)[0] > alpha
    if not ok(lo):
        return 0.0
    if ok(hi):
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


A4 = 0.01


def _exp4_one(args):
    kind, seed, sigma, rep, c, k = args
    g = make_graph(kind, N, seed)
    L, Aj = laplacian(g)
    c = min(c, 1.2 / np.linalg.eigvalsh(L)[-1])
    P = pick_pinned(g, L, k, "degree")
    rng = np.random.default_rng(1000 * seed + rep)
    xi = 1.0 + 0.5 * rng.standard_normal(N)
    e0 = sigma * xi
    rounds, _, _ = run_proportional(e0, Aj, A4, c, GAIN, P, T=4000)
    zstar = guaranteed_radius(L, c, GAIN, P, A4)
    deg = np.array([d for _, d in sorted(g.degree)])
    unp = np.setdiff1d(np.arange(N), P)
    outer = float(np.min(c * deg[unp] / A4))
    return dict(kind=kind, seed=seed, sigma=sigma, rep=rep, c=c, k=k,
                converged=rounds is not None, zstar=zstar, outer=outer,
                runaway=bool(np.any(np.abs(e0[unp]) > c * deg[unp] / A4)),
                guaranteed=bool(np.sqrt(2) * np.linalg.norm(e0) <= zstar),
                e0_inf=float(np.max(np.abs(e0))), e0_2=float(np.linalg.norm(e0)))


def exp4():
    """Region of attraction under bounded influence (Theorem 3)."""
    sigmas = list(np.round(np.logspace(-1, 1.6, 14), 3))
    jobs = [(kind, s, sig, r, 0.08, 8) for kind in TOPOS for s in range(2)
            for sig in sigmas for r in range(10)]
    with Pool(4) as p:
        out = p.map(_exp4_one, jobs, chunksize=4)
    save("exp4_region", out)


# ---------------------------------------------------------------- Exp 5
def worst_capture(L, c, gain, P, f):
    worst = 0.0
    for comp in itertools.combinations(P, f):
        h = harmonic_capture(L, c, gain, P, list(comp))
        honest = [i for i in range(len(h)) if i not in comp]
        worst = max(worst, float(np.mean(h[honest] > 0.5)))
    return worst


def resilient_greedy(g, L, k, c, gain):
    """Add verifiers one at a time, minimising worst-case single-compromise capture,
    ties broken by mu_1(cL + gP)."""
    chosen = []
    n = L.shape[0]
    for _ in range(k):
        best, key = None, None
        for v in range(n):
            if v in chosen:
                continue
            P = chosen + [v]
            wc = worst_capture(L, c, gain, P, 1) if len(P) > 1 else 1.0
            mu1 = mu_extremes(L, c, gain, P)[0]
            kk = (wc, -mu1)
            if key is None or kk < key:
                best, key = v, kk
        chosen.append(best)
    return chosen


def _exp5_one(args):
    kind, seed, rule, f, delta, k = args
    g = make_graph(kind, N, seed)
    L, Aj = laplacian(g)
    c = min(0.05, 1.2 / np.linalg.eigvalsh(L)[-1])
    if rule == "resilient-greedy":
        P = resilient_greedy(g, L, k, c, GAIN)
    else:
        P = pick_pinned(g, L, k, rule, c=c, gain=GAIN, seed=seed)
    # worst case over compromised subsets of size f, linear prediction
    best = None
    for comp in itertools.combinations(P, f):
        h = harmonic_capture(L, c, GAIN, P, list(comp))
        honest = [i for i in range(N) if i not in comp]
        cap = float(np.mean(h[honest] > 0.5))
        if best is None or cap > best[0]:
            best = (cap, list(comp), h)
    cap_lin, comp, h = best
    # nonlinear simulation of that worst case
    off = np.zeros(N)
    off[comp] = delta
    rng = np.random.default_rng(seed)
    e0 = 0.1 * rng.standard_normal(N)
    _, _, ef = run_proportional(e0, Aj, 0.0, c, GAIN, P, phi=np.tanh, T=4000, offsets=off)
    honest = [i for i in range(N) if i not in comp]
    cap_sim = float(np.mean(ef[honest] > delta / 2))
    _, _, efl = run_proportional(e0, Aj, 0.0, c, GAIN, P, phi=lambda z: z, T=4000, offsets=off)
    cap_sim_lin = float(np.mean(efl[honest] > delta / 2))
    hmax = float(np.max(h[honest]))
    return dict(kind=kind, seed=seed, rule=rule, f=f, delta=delta, k=k, c=c, pinned=[int(x) for x in P],
                compromised=[int(x) for x in comp], capture_pred=cap_lin,
                capture_sim_tanh=cap_sim, capture_sim_linear=cap_sim_lin, hmax=hmax,
                err_lin=float(np.max(np.abs(efl / delta - h))))


def exp5():
    """Compromised verifiers: harmonic-measure prediction and resilient placement."""
    jobs = [(kind, s, rule, f, d, k) for kind in TOPOS for s in range(4)
            for rule in ["degree", "spectral-greedy", "resilient-greedy"]
            for f in [1, 2] for d in [1.0, 6.0] for k in [4, 8]]
    with Pool(4) as p:
        out = p.map(_exp5_one, jobs, chunksize=2)
    save("exp5_compromised", out)




# ---------------------------------------------------------------- Exp 6
# Performance-based pinning controllability (after Ghaffari & Arebi's centrality method):
# minimum number of driver (verifier) agents, chosen by a centrality criterion, such that
# every agent reaches the verified answer within T_d rounds (5% residual error).
PB = dict(c=0.05, gain=0.5, alpha=0.01, Td=40, eps=0.05)
RULES = ["betweenness", "closeness", "eigenvector", "degree", "random"]


def _exp6_one(args):
    from agentnet import (make_graph_density, centrality_order, min_drivers_linear,
                          greedy_order, contraction)
    kind, n, d, seed = args
    g = make_graph_density(kind, n, d, seed)
    L, _ = laplacian(g)
    lam = np.linalg.eigvalsh(L)
    rt = PB["eps"] ** (1.0 / PB["Td"])
    out = dict(kind=kind, n=n, d=d, seed=seed, lam2=lam[1], lamN=lam[-1],
               overcoupled=bool(PB["c"] * lam[-1] > 2 + PB["alpha"]))
    orders = {r: centrality_order(g, r, seed) for r in RULES}
    orders["greedy"] = greedy_order(L, PB["c"], PB["gain"], PB["alpha"])
    for r, o in orders.items():
        k = min_drivers_linear(L, o, PB["c"], PB["gain"], PB["alpha"], rt)
        out["k_" + r] = k
    # bound of Corollary 1 for this target: mu_1 >= 1 + alpha - rho_target and mu_1 <= g k / N
    out["k_bound"] = int(np.floor(n * (1 + PB["alpha"] - rt) / PB["gain"])) + 1
    return out


def exp6():
    jobs = [(kind, n, d, s) for kind in TOPOS for n in [20, 40, 80, 160]
            for d in [4, 8, 12, 16] for s in range(5) if d < n]
    with Pool(4) as p:
        out = p.map(_exp6_one, jobs, chunksize=1)
    save("exp6_mdn", out)


# ---------------------------------------------------------------- Exp 7
def _exp7_one(args):
    from agentnet import make_graph_density, centrality_order, settling_rounds, min_drivers_linear
    kind, n, d, seed, rule, sigma = args
    g = make_graph_density(kind, n, d, seed)
    L, Aj = laplacian(g)
    order = centrality_order(g, rule, seed)
    rng = np.random.default_rng(seed + 7)
    e0s = [sigma * (1 + 0.5 * rng.standard_normal(n)) for _ in range(5)]
    rt = PB["eps"] ** (1.0 / PB["Td"])
    k_lin = min_drivers_linear(L, order, PB["c"], PB["gain"], PB["alpha"], rt)
    k_sim = None
    for k in range(1, n + 1):
        ok = all(settling_rounds(e0, Aj, PB["alpha"], PB["c"], PB["gain"], order[:k],
                                 phi=np.tanh, T=PB["Td"], eps=PB["eps"]) is not None for e0 in e0s)
        if ok:
            k_sim = k
            break
    return dict(kind=kind, n=n, d=d, seed=seed, rule=rule, sigma=sigma, k_lin=k_lin, k_sim=k_sim)


def exp7():
    jobs = [(kind, n, d, s, r, sig) for kind in TOPOS for n in [40, 80] for d in [8, 16]
            for s in range(3) for r in ["betweenness", "closeness", "eigenvector"]
            for sig in [0.5, 2.0, 4.0]]
    with Pool(4) as p:
        out = p.map(_exp7_one, jobs, chunksize=2)
    save("exp7_mdn_bounded", out)


# ---------------------------------------------------------------- Exp 8
def _exp8_one(args):
    from agentnet import make_graph_density, centrality_order, settling_controlled
    kind, n, d, seed, rule, sigma = args
    g = make_graph_density(kind, n, d, seed)
    L, Aj = laplacian(g)
    order = centrality_order(g, rule, seed)
    rng = np.random.default_rng(seed + 11)
    e0s = [sigma * (1 + 0.5 * rng.standard_normal(n)) for _ in range(3)]
    out = dict(kind=kind, n=n, d=d, seed=seed, rule=rule, sigma=sigma)
    variants = {"P": [("P", 0.5)], "P-tuned": [("P", gg) for gg in (0.25, 0.5, 1.0, 1.5)],
                "SDRE": [("SDRE", None)]}
    for name, opts in variants.items():
        out["k_" + name], out["energy_" + name] = None, None
        for k in range(1, n + 1):
            done = False
            for ctrl, gg in opts:
                res = [settling_controlled(e0, Aj, PB["alpha"], PB["c"], order[:k], ctrl,
                                           gain=gg if gg else 0.5, T=PB["Td"], eps=PB["eps"])
                       for e0 in e0s]
                if all(r is not None for r, _ in res):
                    out["k_" + name] = k
                    out["energy_" + name] = float(np.mean([E for _, E in res]))
                    out["gain_" + name] = gg
                    done = True
                    break
            if done:
                break
    return out


def exp8():
    """Minimum driver agents for the performance target: proportional versus SDRE pinning."""
    jobs = [(kind, 40, 8, s, r, sig) for kind in TOPOS for s in range(3)
            for r in ["betweenness", "closeness", "eigenvector"] for sig in [1.0, 3.0]]
    with Pool(4) as p:
        out = p.map(_exp8_one, jobs, chunksize=1)
    save("exp8_sdre_mda", out)


if __name__ == "__main__":
    which = sys.argv[1:] or ["exp1", "exp2", "exp4", "exp5", "exp6", "exp7", "exp8"]
    for w in which:
        print("running", w, flush=True)
        globals()[w]()
