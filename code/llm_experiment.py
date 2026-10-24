"""Pinning experiment with real LLM agents.

Each agent holds a number in [0, 100]. Every round it sees its own value and its
neighbours' values and proposes a new one. Driver agents also receive a verifier
message with the verified value R. We measure the error e_i = x_i - R per round and
the rounds until ||e|| <= eps ||e(0)||, for driver agents chosen by betweenness,
eigenvector centrality or at random.

Usage:
    python llm_experiment.py --mock            # offline test, no API calls
    python llm_experiment.py --budget 20       # real run, stops at $20
"""
import argparse
import asyncio
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from agentnet import make_graph_density, centrality_order  # noqa: E402

MODEL = "claude-haiku-5-5"
PRICE_IN, PRICE_OUT = 0.10 / 1e6, 0.50 / 1e6   # USD per token, prompts under 100K tokens
RES = os.path.join(os.path.dirname(__file__), "..", "results")

SYSTEM = (
    "You are one member of a team of agents that must agree on a single number between 0 and 100: "
    "the setting for a shared design parameter. Each round you see your own current value and the "
    "values currently proposed by the teammates you can hear. Decide your value for the next round. "
    "Reply only with the JSON object."
)
SCHEMA = {"type": "object", "properties": {"value": {"type": "number"}},
          "required": ["value"], "additionalProperties": False}


def prompt(i, x, nbrs, driver, ref, rnd):
    lines = [f"Round {rnd}. Your current value: {x[i]:.1f}."]
    lines.append("Teammates' values: " + ", ".join(f"{x[j]:.1f}" for j in nbrs) + ".")
    if driver:
        lines.append(f"A verifier who has checked the parameter tells you: the verified value is {ref:.0f}.")
    lines.append("What is your value for the next round?")
    return "\n".join(lines)


class Meter:
    def __init__(self, budget):
        self.budget, self.tin, self.tout, self.calls = budget, 0, 0, 0

    @property
    def cost(self):
        return self.tin * PRICE_IN + self.tout * PRICE_OUT


async def ask(client, sem, meter, text):
    async with sem:
        r = await client.messages.create(
            model=MODEL, max_tokens=2000, system=SYSTEM,
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": text}])
    meter.tin += r.usage.input_tokens
    meter.tout += r.usage.output_tokens
    meter.calls += 1
    if r.stop_reason != "end_turn":
        return None
    txt = next((b.text for b in r.content if b.type == "text"), None)
    try:
        return float(np.clip(json.loads(txt)["value"], 0, 100))
    except (TypeError, ValueError, KeyError):
        return None


def mock_update(i, x, nbrs, driver, ref, rng):
    """Stand-in agent for offline tests: bounded averaging plus pinning."""
    v = x[i] + 0.3 * np.mean(np.tanh((x[nbrs] - x[i]) / 20) * 20)
    if driver:
        v += 0.5 * (ref - x[i])
    return float(np.clip(v + rng.normal(0, 1), 0, 100))


async def run_one(g, drivers, x0, ref, T, client, sem, meter, rng):
    n = g.number_of_nodes()
    nb = [list(g.neighbors(i)) for i in range(n)]
    D = set(drivers)
    x = np.array(x0, float)
    traj, fails = [x.tolist()], 0
    for t in range(1, T + 1):
        if client is None:
            new = [mock_update(i, x, nb[i], i in D, ref, rng) for i in range(n)]
        else:
            if meter.cost >= meter.budget:
                raise RuntimeError("budget")
            new = await asyncio.gather(*[ask(client, sem, meter, prompt(i, x, nb[i], i in D, ref, t))
                                         for i in range(n)])
        fails += sum(v is None for v in new)
        x = np.array([x[i] if v is None else v for i, v in enumerate(new)])
        traj.append(x.tolist())
    return traj, fails


def settle(traj, ref, eps):
    e = np.array(traj) - ref
    n0 = np.linalg.norm(e[0])
    hit = np.where(np.linalg.norm(e, axis=1) <= eps * n0)[0]
    return int(hit[0]) if len(hit) else None


async def main(a):
    client = None
    if not a.mock:
        import anthropic
        # The key comes from a network secret that the environment's proxy adds to requests to
        # api.anthropic.com; the SDK only needs a placeholder value to build the request.
        key = os.environ.get("EXP_ANTHROPIC_KEY") or os.environ.get("ANTHROPIC_API_KEY") or "set-by-proxy"
        client = anthropic.AsyncAnthropic(api_key=key)
    sem, meter = asyncio.Semaphore(a.concurrency), Meter(a.budget)
    out_path = os.path.join(RES, "llm_mock.jsonl" if a.mock else "llm_exp.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["key"] for l in open(out_path)}
    for kind in a.kinds:
        for seed in range(a.graphs):
            g = make_graph_density(kind, a.n, a.d, seed)
            rng = np.random.default_rng(100 + seed)
            for rep in range(a.reps):
                x0 = rng.uniform(10, 50, a.n).round(0)   # every agent starts below the verified value
                for rule in a.rules:
                    for k in a.ks:
                        drivers = [] if rule == "none" else list(centrality_order(g, rule, seed + rep)[:k])
                        key = f"{kind}|{seed}|{rep}|{rule}|{k}"
                        if key in done or (rule == "none" and k != a.ks[0]):
                            continue
                        t0 = time.time()
                        try:
                            traj, fails = await run_one(g, drivers, x0, a.ref, a.T, client, sem, meter, rng)
                        except RuntimeError:
                            print(f"stopped at budget ${meter.cost:.2f}", flush=True)
                            return
                        rec = dict(key=key, kind=kind, seed=seed, rep=rep, rule=rule, k=k if rule != "none" else 0,
                                   n=a.n, d=a.d, ref=a.ref, drivers=[int(i) for i in drivers],
                                   edges=[list(map(int, e)) for e in g.edges()], traj=traj, fails=fails,
                                   settle=settle(traj, a.ref, a.eps), model=None if a.mock else MODEL)
                        with open(out_path, "a") as f:
                            f.write(json.dumps(rec) + "\n")
                        print(f"{key}: settle={rec['settle']} final|e|={np.abs(np.array(traj[-1]) - a.ref).mean():.1f} "
                              f"fails={fails} cost=${meter.cost:.2f} calls={meter.calls} ({time.time() - t0:.0f}s)",
                              flush=True)
    print(f"done, total cost ${meter.cost:.2f}, {meter.calls} calls")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mock", action="store_true")
    p.add_argument("--budget", type=float, default=20.0)
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--d", type=int, default=4)
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--ref", type=float, default=80.0)
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--graphs", type=int, default=2)
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--ks", type=int, nargs="+", default=[3, 6])
    p.add_argument("--kinds", nargs="+", default=["random", "small-world", "scale-free"])
    p.add_argument("--rules", nargs="+", default=["betweenness", "eigenvector", "random", "none"])
    p.add_argument("--concurrency", type=int, default=16)
    asyncio.run(main(p.parse_args()))
