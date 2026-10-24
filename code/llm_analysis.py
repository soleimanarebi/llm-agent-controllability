"""Analysis of the real-LLM pinning experiment (results/llm_exp.jsonl)."""
import collections
import json
import os

import numpy as np

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def load():
    return [json.loads(l) for l in open(os.path.join(RES, "llm_exp.jsonl"))]


def neighbours(r):
    nb = [[] for _ in range(r["n"])]
    for i, j in r["edges"]:
        nb[i].append(j)
        nb[j].append(i)
    return nb


def fit_update(R):
    """Least squares for dx_i = c*(mean of neighbours - x_i) + g*[driver]*(ref - x_i) + b."""
    X, y = [], []
    for r in R:
        t, nb, D = np.array(r["traj"]), neighbours(r), set(r["drivers"])
        for s in range(len(t) - 1):
            for i in range(r["n"]):
                X.append([np.mean(t[s, nb[i]]) - t[s, i], (r["ref"] - t[s, i]) * (i in D), 1.0])
                y.append(t[s + 1, i] - t[s, i])
    X, y = np.array(X), np.array(y)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - np.sum((y - X @ coef) ** 2) / np.sum((y - y.mean()) ** 2)
    return dict(c=coef[0], g=coef[1], b=coef[2], r2=r2, n_obs=len(y))


def predict(r, c, g, b):
    t0, nb, D = np.array(r["traj"][0], float), neighbours(r), set(r["drivers"])
    x = t0.copy()
    for _ in range(len(r["traj"]) - 1):
        x = np.clip(x + np.array([c * (np.mean(x[nb[i]]) - x[i]) + g * (i in D) * (r["ref"] - x[i]) + b
                                  for i in range(r["n"])]), 0, 100)
    return x


def summary():
    R = load()
    fit = fit_update(R)
    cells = collections.defaultdict(list)
    obs, pred = [], []
    for r in R:
        t = np.array(r["traj"])
        ratio = np.linalg.norm(t[-1] - r["ref"]) / np.linalg.norm(t[0] - r["ref"])
        xp = predict(r, fit["c"], fit["g"], fit["b"])
        pr = np.linalg.norm(xp - r["ref"]) / np.linalg.norm(t[0] - r["ref"])
        if r["rule"] != "none":
            obs.append(ratio)
            pred.append(pr)
        cells[(r["kind"], r["rule"], r["k"])].append(dict(err=np.abs(t[-1] - r["ref"]).mean(), ratio=ratio,
                                                        settled=r["settle"] is not None, pred=pr))
    out = {"fit": fit, "corr_pred_obs": float(np.corrcoef(obs, pred)[0, 1]), "cells": {}}
    for key, v in cells.items():
        out["cells"]["|".join(map(str, key))] = dict(
            err=float(np.mean([x["err"] for x in v])), ratio=float(np.mean([x["ratio"] for x in v])),
            pred=float(np.mean([x["pred"] for x in v])), settled=int(sum(x["settled"] for x in v)), runs=len(v))
    return out


if __name__ == "__main__":
    s = summary()
    print(json.dumps(s["fit"], indent=1), "corr", round(s["corr_pred_obs"], 3))
    for k, v in sorted(s["cells"].items()):
        print(k, {a: round(b, 2) if isinstance(b, float) else b for a, b in v.items()})
    json.dump(s, open(os.path.join(RES, "llm_summary.json"), "w"), indent=1, default=float)


def held_out(R):
    """Fit on all graphs but one and predict the runs on the held-out graph."""
    graphs = sorted({(r["kind"], r["seed"]) for r in R})
    obs, pred = [], []
    for gk in graphs:
        train = [r for r in R if (r["kind"], r["seed"]) != gk]
        f = fit_update(train)
        for r in R:
            if (r["kind"], r["seed"]) == gk and r["rule"] != "none":
                t = np.array(r["traj"])
                n0 = np.linalg.norm(t[0] - r["ref"])
                obs.append(np.linalg.norm(t[-1] - r["ref"]) / n0)
                pred.append(np.linalg.norm(predict(r, f["c"], f["g"], f["b"]) - r["ref"]) / n0)
    obs, pred = np.array(obs), np.array(pred)
    return dict(corr=float(np.corrcoef(obs, pred)[0, 1]), mae=float(np.mean(np.abs(obs - pred))), n=len(obs))


def paired(R, a, b, k, n_boot=10000, seed=0):
    """Paired differences in final mean |e| between rankings a and b on the same graph and initial state."""
    idx = {(r["kind"], r["seed"], r["rep"], r["rule"], r["k"]): np.abs(np.array(r["traj"][-1]) - r["ref"]).mean() for r in R}
    diffs = np.array([idx[(kd, s, rp, a, k)] - idx[(kd, s, rp, b, k)]
                      for (kd, s, rp, ru, kk) in idx if ru == a and kk == k])
    rng = np.random.default_rng(seed)
    boot = [rng.choice(diffs, len(diffs)).mean() for _ in range(n_boot)]
    from scipy.stats import wilcoxon
    return dict(mean=float(diffs.mean()), ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                a_better=int((diffs < 0).sum()), n=len(diffs), p=float(wilcoxon(diffs).pvalue))
