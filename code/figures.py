"""Builds the figures and LaTeX tables of the paper from ../results/*.json."""
import collections
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
TAB = os.path.join(HERE, "..", "sections", "generated")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
COL = {"random": "#2a6fdb", "small-world": "#e07b39", "scale-free": "#2e9e6b", "star": "#8b5cc4",
       "degree": "#2a6fdb", "spectral-greedy": "#2e9e6b", "resilient-greedy": "#c43d5a"}
TOPOS = ["random", "small-world", "scale-free"]


def load(name):
    with open(os.path.join(RES, name + ".json")) as f:
        return json.load(f)


def table_drivers():
    d = load("exp1_drivers")
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Harness topology & $N$ & Links & No memory & With memory & Identical agents \\",
             r"\midrule"]
    for r in d["rows"]:
        lines.append(f"{r['graph']} & {r['n']} & {r['edges']} & {r['nd_no_memory']} & "
                     f"{r['nd_memory']} & {r['nd_exact_identical']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_drivers.tex"), "w").write("\n".join(lines) + "\n")


def fig_energy():
    d = load("exp2_energy")
    agg = collections.defaultdict(list)
    for r in d:
        agg[(r["kind"], r["rule"], r["k"])].append(r)
    ks = sorted({r["k"] for r in d})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax = axes[0]
    summary = {}
    for kind in TOPOS:
        xs, ys = [], []
        for k in ks:
            rs = agg[(kind, "random", k)]
            ok = [r for r in rs if r["cond"] < 1e14]
            if len(ok) == len(rs):
                xs.append(k)
                ys.append(np.median([r["avg_energy"] for r in rs]))
        ax.plot(xs, ys, "o-", color=COL[kind], label=kind, ms=3.5)
        summary[kind] = dict(zip(xs, ys))
    kk = np.array(ks, float)
    ax.plot(kk, 64 / (kk * 30), "k--", lw=1, label=r"floor $N/(kT)$")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("driver agents $k$ (of 64)")
    ax.set_ylabel(r"average energy $\mathrm{tr}(W_T^{-1})/N$")
    ax.set_title("(a) energy to steer the collective, $T=30$")
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    for (kind, rule, lab, col, ls) in [("scale-free", "random", "scale-free, random drivers", COL["scale-free"], "-"),
                                       ("scale-free", "degree", "scale-free, hub drivers", COL["scale-free"], "--"),
                                       ("random", "random", "random, random drivers", COL["random"], "-"),
                                       ("random", "degree", "random, hub drivers", COL["random"], "--"),
                                       ("star", "random", "star (any drivers)", COL["star"], "-")]:
        ys = [np.median([np.log10(max(r["cond"], 1.0)) for r in agg[(kind, rule, k)]]) for k in ks]
        ax.plot(ks, np.minimum(ys, 17), "o" + ls, color=col, label=lab, ms=3)
    ax.axhline(16, color="k", lw=0.8, ls=":")
    ax.text(1, 14.6, "double-precision limit", fontsize=7, ha="left")
    ax.set_ylim(0, 19)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("driver agents $k$ (of 64)")
    ax.set_ylabel(r"$\log_{10}$ condition number of $W_T$")
    ax.set_title("(b) numerical controllability")
    ax.legend(frameon=False, fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_energy.pdf"))
    # numbers for the text
    first_ok = {}
    for kind in TOPOS + ["star"]:
        for rule in ["random", "degree", "spectral-greedy"]:
            for k in ks:
                rs = agg[(kind, rule, k)]
                if all(r["cond"] < 1e14 for r in rs):
                    first_ok[f"{kind}/{rule}"] = k
                    break
    unreached_star = {k: float(np.median([r["unreached"] for r in agg[("star", "degree", k)]]))
                      for k in ks}
    ratio = {kind: {k: float(np.median([r["avg_energy"] / r["bound"] for r in agg[(kind, "spectral-greedy", k)]]))
                    for k in ks if all(r["cond"] < 1e14 for r in agg[(kind, "spectral-greedy", k)])}
             for kind in TOPOS}
    return dict(energy=summary, first_numerically_controllable=first_ok,
                star_unreached_modes=unreached_star, energy_over_floor=ratio)


def fig_region():
    d = load("exp4_region")
    agg = collections.defaultdict(list)
    for r in d:
        agg[(r["kind"], r["seed"])].append(r)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ax = axes[0]
    out = {}
    marker = {0: "o", 1: "s"}
    for (kind, seed), rs in sorted(agg.items()):
        by = collections.defaultdict(list)
        for r in rs:
            by[r["sigma"]].append(r)
        sig = sorted(by)
        conv = [np.mean([r["converged"] for r in by[s]]) for s in sig]
        run = [np.mean([r["runaway"] for r in by[s]]) for s in sig]
        ax.plot(sig, conv, marker[seed] + "-", color=COL[kind], ms=3, lw=1,
                label=f"{kind} {seed + 1}")
        axes[1].plot(np.array(sig) / rs[0]["zstar"], conv, marker[seed] + "-", color=COL[kind], ms=3, lw=1)
        s50 = max([s for s, cv in zip(sig, conv) if cv >= 0.5], default=None)
        out[f"{kind}/{seed}"] = dict(zstar=rs[0]["zstar"], outer=rs[0]["outer"], c=rs[0]["c"],
                                     sigma50=s50,
                                     sigma_guaranteed=max([s for s in sig if all(r["guaranteed"] for r in by[s])], default=None),
                                     fail_with_runaway=float(np.mean([r["runaway"] for r in rs if not r["converged"]])) if any(not r["converged"] for r in rs) else None,
                                     conv_given_no_runaway=float(np.mean([r["converged"] for r in rs if not r["runaway"]])))
    ax.set_xscale("log")
    ax.set_xlabel(r"initial bias scale $\sigma$")
    ax.set_ylabel("share of runs that converge")
    ax.set_title("(a) recovery from an initial bias")
    ax.legend(frameon=False, fontsize=6, ncol=1, loc="lower left")
    axes[1].set_xscale("log")
    axes[1].axvline(1.0, color="k", lw=0.8, ls=":")
    axes[1].set_xlabel(r"$\sigma / \bar z^\ast$")
    axes[1].set_title(r"(b) rescaled by $\bar z^\ast$ of Theorem 2")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_region.pdf"))
    return out


def table_capture():
    d = load("exp5_compromised")
    agg = collections.defaultdict(list)
    for r in d:
        agg[(r["kind"], r["k"], r["f"], r["rule"], r["delta"])].append(r)
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r" & & & \multicolumn{3}{c}{Honest agents captured, worst case (\%)} & \\",
             r"\cmidrule(lr){4-6}",
             r"Topology & Placement & $f$ & Theorem~\ref{thm:capture} & linear sim. & bounded, $\delta=6$ & $\max_i h_i$ \\",
             r"\midrule"]
    out = {}
    for kind in TOPOS:
        for k in [4, 8]:
            for f in [1, 2]:
                for rule in ["degree", "spectral-greedy", "resilient-greedy"]:
                    v1 = agg[(kind, k, f, rule, 1.0)]
                    v6 = agg[(kind, k, f, rule, 6.0)]
                    pred = 100 * np.mean([r["capture_pred"] for r in v1])
                    sl = 100 * np.mean([r["capture_sim_linear"] for r in v1])
                    st6 = 100 * np.mean([r["capture_sim_tanh"] for r in v6])
                    st1 = 100 * np.mean([r["capture_sim_tanh"] for r in v1])
                    hm = np.mean([r["hmax"] for r in v1])
                    out[f"{kind}/k{k}/f{f}/{rule}"] = dict(pred=pred, sim_lin=sl, sim_tanh1=st1, sim_tanh6=st6, hmax=hm,
                                                          maxerr=max(r["err_lin"] for r in v1 + v6))
                    if k == 8:
                        lines.append(f"{kind if (f == 1 and rule == 'degree') else ''} & {rule} & {f} & "
                                     f"{pred:.1f} & {sl:.1f} & {st6:.1f} & {hm:.2f} \\\\")
        if kind != TOPOS[-1]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_capture.tex"), "w").write("\n".join(lines) + "\n")
    return out


def fig_capture():
    d = load("exp5_compromised")
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.7))
    for kind in TOPOS:
        rs = [r for r in d if r["kind"] == kind and r["delta"] == 6.0]
        ax.scatter([100 * r["capture_pred"] for r in rs], [100 * r["capture_sim_tanh"] for r in rs],
                   s=10, color=COL[kind], label=kind, alpha=0.7)
    ax.plot([0, 100], [0, 100], "k--", lw=0.8)
    ax.set_xlabel("captured, Theorem 3 (%)")
    ax.set_ylabel(r"captured, bounded model, $\delta=6$ (%)")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_capture.pdf"))


def table_sdre():
    d = load("exp8_sdre_mda")
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r" & & \multicolumn{3}{c}{Driver agents (\% of $N$)} & \multicolumn{3}{c}{Energy per run} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
             r"Bias & Topology & P & P-tuned & SDRE & P & P-tuned & SDRE \\", r"\midrule"]
    out = {}
    for sig in [1.0, 3.0]:
        for kind in TOPOS + ["all"]:
            v = [r for r in d if r["sigma"] == sig and (kind == "all" or r["kind"] == kind)]
            ks = [100 * np.mean([r["k_" + c] for r in v]) / r0["n"] for c in ("P", "P-tuned", "SDRE") for r0 in v[:1]]
            es = [np.mean([r["energy_" + c] for r in v]) for c in ("P", "P-tuned", "SDRE")]
            out[f"{kind}_{sig}"] = dict(k=ks, energy=es)
            lab = f"$\\sigma={sig:.0f}$" if kind == TOPOS[0] else ""
            lines.append(f"{lab} & {kind} & " + " & ".join(f"{x:.0f}" for x in ks) + " & "
                         + " & ".join(f"{x:.1f}" for x in es) + r" \\")
        if sig == 1.0:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_sdre.tex"), "w").write("\n".join(lines) + "\n")
    return out


def fig_mda():
    d = load("exp6_mdn")
    rules = [("betweenness", "BC", "#2a6fdb", "-"), ("closeness", "CC", "#e07b39", "-"),
             ("eigenvector", "EC", "#2e9e6b", "-"), ("degree", "degree", "#888888", "--"),
             ("random", "random", "#bbbbbb", "--"), ("greedy", "spectral-greedy", "k", ":")]
    agg = collections.defaultdict(list)
    for r in d:
        agg[(r["kind"], r["d"], r["n"])].append(r)
    ns = [20, 40, 80, 160]
    dens = [4, 8, 16]
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.6), sharex=True, sharey=True)
    summary = {}
    for i, kind in enumerate(TOPOS):
        for j, dm in enumerate(dens):
            ax = axes[i, j]
            for key, lab, col, ls in rules:
                xs, ys = [], []
                for n in ns:
                    v = agg[(kind, dm, n)]
                    ks = [r["k_" + key] for r in v if r["k_" + key] is not None]
                    if len(ks) >= 3:
                        xs.append(n)
                        ys.append(100 * np.mean(ks) / n)
                    summary[f"{kind}/d{dm}/n{n}/{key}"] = (100 * np.mean(ks) / n if ks else None, len(ks), len(v))
                ax.plot(xs, ys, "o" + ls, color=col, ms=3, lw=1.2 if ls == "-" else 1,
                        label=lab)
            bnd = [100 * np.mean([r["k_bound"] for r in agg[(kind, dm, n)]]) / n for n in ns]
            ax.plot(ns, bnd, color="#c43d5a", lw=0.8, ls="-.", label="bound (Cor. 4a)")
            infeas = [n for n in ns if sum(r["k_betweenness"] is not None for r in agg[(kind, dm, n)]) < 3]
            for n in infeas:
                ax.text(n, 8, "x", ha="center", color="#c43d5a", fontsize=9)
            ax.set_xscale("log", base=2)
            ax.set_xticks(ns)
            ax.set_xticklabels([str(n) for n in ns])
            ax.set_ylim(0, 105)
            if i == 0:
                ax.set_title(f"mean degree {dm}")
            if j == 0:
                ax.set_ylabel(f"{kind}\nminimum driver agents (%)")
            if i == 2:
                ax.set_xlabel("agents $N$")
    axes[0, 0].legend(frameon=False, fontsize=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_mda.pdf"))
    return summary


def table_density():
    d = load("exp6_mdn")
    agg = collections.defaultdict(list)
    for r in d:
        agg[(r["kind"], r["d"], r["n"])].append(r)
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Topology & Criterion & $\bar d=4$ & $\bar d=8$ & $\bar d=12$ & $\bar d=16$ \\", r"\midrule"]
    for kind in TOPOS:
        for key, lab in [("betweenness", "betweenness"), ("closeness", "closeness"),
                         ("eigenvector", "eigenvector"), ("greedy", "spectral-greedy")]:
            cells = []
            for dm in [4, 8, 12, 16]:
                v = agg[(kind, dm, 80)]
                ks = [r["k_" + key] for r in v if r["k_" + key] is not None]
                cells.append(f"{100 * np.mean(ks) / 80:.0f}" + ("" if len(ks) == len(v) else f"$^{{{len(ks)}/{len(v)}}}$")
                             if ks else r"--")
            lines.append(f"{kind if key == 'betweenness' else ''} & {lab} & " + " & ".join(cells) + r" \\")
        if kind != TOPOS[-1]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_density.tex"), "w").write("\n".join(lines) + "\n")


def fig_llm():
    path = os.path.join(RES, "llm_exp.jsonl")
    if not os.path.exists(path):
        return None
    R = [json.loads(l) for l in open(path)]
    rcol = {"betweenness": "#2a6fdb", "eigenvector": "#2e9e6b", "random": "#999999", "none": "#000000"}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
    for ax, kind in zip(axes, TOPOS):
        for rule in ["betweenness", "eigenvector", "random", "none"]:
            for k, ls in [(6, "-"), (3, "--")]:
                if rule == "none" and k == 3:
                    continue
                v = [r for r in R if r["kind"] == kind and r["rule"] == rule and r["k"] == (0 if rule == "none" else k)]
                curves = []
                for r in v:
                    e = np.linalg.norm(np.array(r["traj"]) - r["ref"], axis=1)
                    curves.append(e / e[0])
                lab = "no drivers" if rule == "none" else f"{rule}, k={k}"
                ax.plot(np.mean(curves, axis=0), ls if rule != "none" else ":", color=rcol[rule], lw=1.3 if ls == "-" else 1,
                        label=lab)
        ax.axhline(0.1, color="#999999", lw=0.6)
        ax.set_title(kind)
        ax.set_xlabel("round")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel(r"$\|e(t)\|/\|e(0)\|$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, frameon=False, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(os.path.join(FIG, "fig_llm.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    table_drivers()
    nums = dict(energy=fig_energy(), region=fig_region(), capture=table_capture())
    fig_capture()
    nums["mda"] = fig_mda()
    table_density()
    nums["sdre"] = table_sdre()
    fig_llm()
    with open(os.path.join(RES, "summary_numbers.json"), "w") as f:
        json.dump(nums, f, indent=1, default=float)
    print("done")
