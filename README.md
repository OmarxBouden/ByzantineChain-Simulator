# ByzantineChain

As part of my research project, I'm building a simulation framework to study how Byzantine agents exploit decentralized AI task markets — and how to catch them.

The idea is simple: in any open marketplace where agents bid on compute tasks and get paid for results, some will cheat. They'll underbid everyone, win the task, then submit garbage. This project simulates that dynamic and tests whether we can design mechanisms that reliably detect and penalize dishonest behavior.

## How it works

Each round runs a straightforward pipeline:

```
Task → Bids → Assignment → Execution → Verification → Settlement
```

- A task is generated with some difficulty and reward
- All agents submit a bid; lowest bid wins (naive baseline)
- The winner executes the task — honestly or not
- A panel of 3 verifiers votes accept/reject based on output quality
- Accepted work gets the reward, rejected work gets slashed 50%

Everything gets logged to an append-only JSONL ledger, mimicking an on-chain log. Ground-truth labels (who's honest, who isn't) are tracked internally but never exposed to the verification layer.

## Agent types

| Type | Bidding | Execution |
|---|---|---|
| **Honest** | True cost | Full effort |
| **Free-rider** | Underbids 50–70% | Minimal effort, low quality |
| **Strategic** | Moderate underbid | Shirks when expected gain > expected penalty |
| **Selective** | Honest on cheap tasks, aggressive on expensive ones | Shirks only on high-value tasks |

Default population: 14 honest, 3 free-riders, 2 strategic, 1 selective.

## Running it

```bash
python -m engine.simulation
```

Outputs `output/events.jsonl` and `output/ground_truth.json`. Tweak population and round count in `engine/simulation.py`.

## Dashboard

```bash
pip install streamlit plotly pandas
streamlit run dashboard/app.py
```

Four tabs:
- **Timeline** — balance curves, quality over rounds, task distributions
- **Agent Profiles** — stats table + per-agent drill-down
- **Detection Metrics** — bid ratio, effort ratio, quality distributions, ROC/PR curves
- **Assessor Inspector** — Phase 2 placeholder; shows the assessor-visible log with ground truth stripped

## Structure

```
engine/
  agents.py          # Agent types and behavioral logic
  tasks.py           # Task generation
  mechanisms.py      # Assignment + verification
  ledger.py          # Append-only event log
  simulation.py      # Main loop
events/
  schema.py          # Event dataclasses
dashboard/
  app.py             # Streamlit dashboard
output/              # Generated at runtime, git-ignored
```

## What I want to test

Phase 1 already shows that naive lowest-bid assignment is completely broken — free-riders win every auction. The real questions:

- Can an LLM-based Assessor identify Byzantine agents from on-chain history alone, without ground truth?
- Does trust-weighted assignment reduce exploitation without hurting honest low bidders?
- How do strategic agents that mix honest and dishonest behavior compare to pure free-riders in terms of detection difficulty?
- At what adversarial population ratio does the verification mechanism collapse?
- Can agents learn to game the Assessor itself — build reputation cheaply, then exploit expensive tasks?

## Roadmap

- **Phase 2** — LLM Assessor: trust-scored assignment based on prompted reasoning over agent histories
- **Phase 3** — Adaptive verification: parameterized (α, β) thresholds tuned by Assessor confidence
- **Phase 4** — Adversarial robustness: agents that actively try to fool the Assessor
