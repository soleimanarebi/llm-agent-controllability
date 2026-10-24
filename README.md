# Controllability of LLM agent networks

Code and data for the manuscript *Controllability of LLM Agent Networks: Which Agents to Verify, and How Many* (S. Arebi, 2026, in preparation).

The paper treats a group of language-model agents as a networked dynamical system and asks which agents should receive a verifier's correction, and how few are enough for the whole group to reach the verified answer within a given number of rounds. It extends pinning control of complex dynamical networks (Ghaffari & Arebi, *Nonlinear Dynamics* 83:1003–1013, 2016) to agent collectives.

## Contents

| Path | What it does |
|---|---|
| `code/agentnet.py` | Graph generators, Laplacian and influence matrices, centrality rankings, settling rounds, minimum driver agents, SDRE oversight, harmonic capture |
| `code/experiments.py` | Simulation experiments (driver counts, control energy, region of attraction, compromised verifiers, minimum driver agents, SDRE) |
| `code/llm_experiment.py` | Pilot with 30 Claude agents converging to a verified value, with verifiers placed by betweenness, eigenvector centrality or at random |
| `code/llm_analysis.py` | Fits the linear update to the pilot data, leave-one-graph-out prediction, paired bootstrap and Wilcoxon tests |
| `code/figures.py` | Figures and LaTeX tables from the results |
| `results/` | Outputs of every experiment, including the 84 pilot runs (`llm_exp.jsonl`, 50,400 model calls) |

## Reproducing

```bash
pip install -r requirements.txt
cd code
python experiments.py          # all simulations -> ../results/*.json
python llm_analysis.py         # pilot statistics -> ../results/llm_summary.json
python figures.py              # figures -> ../figures, tables -> ../sections/generated
```

The pilot data is included, so the analysis and figures run without an API key. To rerun the pilot itself:

```bash
python llm_experiment.py --mock            # offline check, no API calls
export ANTHROPIC_API_KEY=...               # your own key
python llm_experiment.py --budget 5        # live run; the full pilot cost about $2
```

The live run uses `claude-haiku-5-5` with low effort and structured JSON output. It appends to `results/llm_exp.jsonl` and skips runs already recorded, so it can be resumed.

## Citation

S. Arebi. Controllability of LLM agent networks: which agents to verify, and how many. Manuscript in preparation, 2026.

## Licence

MIT (see `LICENSE`).
