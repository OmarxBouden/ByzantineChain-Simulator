"""Main simulation loop.

Runs T rounds of: task → bids → assign → execute → verify → settle.
All events committed to ledger.
"""

import json
import time
from pathlib import Path
from typing import List, Dict

from configs.scenario import ScenarioConfig, PRESETS
from engine.tasks import TaskGenerator, Task
from engine.agents import Agent, HonestAgent, FreeRiderAgent, StrategicAgent, SelectiveAgent
from engine.ledger import Ledger
from engine.mechanisms import LowestBidAssignment, MajorityVoteVerification
from events.schema import (
    TaskCreated, BidSubmitted, AgentAssigned,
    ResultSubmitted, VerificationOutcome, Settlement,
)


class Simulation:
    def __init__(self, config: ScenarioConfig = None, output_root: str = "output"):
        self.config = config or PRESETS["baseline"]
        cfg = self.config

        self.output_dir = Path(output_root) / cfg.name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create agents
        self.agents: List[Agent] = []
        idx = 0
        for _ in range(cfg.n_honest):
            self.agents.append(HonestAgent(f"A{idx:03d}", seed=cfg.seed + idx))
            idx += 1
        for _ in range(cfg.n_freerider):
            self.agents.append(FreeRiderAgent(f"A{idx:03d}", seed=cfg.seed + idx))
            idx += 1
        for _ in range(cfg.n_strategic):
            self.agents.append(StrategicAgent(f"A{idx:03d}", seed=cfg.seed + idx))
            idx += 1
        for _ in range(cfg.n_selective):
            self.agents.append(SelectiveAgent(f"A{idx:03d}", seed=cfg.seed + idx))
            idx += 1

        # Components
        self.task_gen = TaskGenerator(seed=cfg.seed)
        self.ledger = Ledger(output_path=str(self.output_dir / "events.jsonl"))
        self.assigner = LowestBidAssignment()
        self.verifier = MajorityVoteVerification(
            quality_threshold=cfg.quality_threshold,
            num_verifiers=cfg.num_verifiers,
            seed=cfg.seed,
        )

        # Ground truth for evaluation
        self.ground_truth = {
            a.agent_id: a.agent_type for a in self.agents
        }

    def run(self):
        cfg = self.config
        print(f"[{cfg.name}] Starting: {cfg.total_agents} agents, {cfg.n_rounds} rounds")
        print(f"  Population: {self._population_summary()}")
        if cfg.description:
            print(f"  Description: {cfg.description}")

        for rnd in range(1, cfg.n_rounds + 1):
            self._run_round(rnd)

        # Save ground truth
        gt_path = self.output_dir / "ground_truth.json"
        with open(gt_path, "w") as f:
            json.dump(self.ground_truth, f, indent=2)

        # Save config
        self.config.save(self.output_dir / "config.json")

        self._print_summary()
        return self.output_dir

    def _run_round(self, rnd: int):
        ts = time.time()
        cfg = self.config

        # 1. Generate task
        task = self.task_gen.generate()
        self.ledger.append(TaskCreated(
            round=rnd, ts=ts, task_id=task.task_id,
            difficulty=task.difficulty, reward=task.reward,
            min_effort=task.min_effort,
        ))

        # 2. Collect bids
        bids = []
        for agent in self.agents:
            bid_amount, justification, true_cost = agent.bid(task)
            bids.append((agent, bid_amount, justification, true_cost))
            self.ledger.append(BidSubmitted(
                round=rnd, ts=ts, task_id=task.task_id,
                agent_id=agent.agent_id, bid=bid_amount,
                justification=justification,
                true_cost=true_cost, agent_true_type=agent.agent_type,
            ))

        # 3. Assignment
        winner, rationale = self.assigner.select(task, bids)
        verifier_ids = self.verifier.select_verifiers(self.agents, exclude=winner)
        self.ledger.append(AgentAssigned(
            round=rnd, ts=ts, task_id=task.task_id,
            agent_id=winner.agent_id, method="lowest_bid",
            verifier_ids=verifier_ids, rationale=rationale,
        ))

        # 4. Execution
        effort, quality = winner.execute(task)
        self.ledger.append(ResultSubmitted(
            round=rnd, ts=ts, task_id=task.task_id,
            agent_id=winner.agent_id, effort=effort,
            quality=quality, min_effort=task.min_effort,
            agent_true_type=winner.agent_type,
        ))

        # 5. Verification
        verdict, votes = self.verifier.verify(quality)
        self.ledger.append(VerificationOutcome(
            round=rnd, ts=ts, task_id=task.task_id,
            agent_id=winner.agent_id, verdict=verdict,
            verifier_votes=votes,
        ))

        # 6. Settlement
        if verdict == "accept":
            reward = task.reward
            penalty = 0.0
        else:
            reward = 0.0
            penalty = task.reward * cfg.penalty_ratio

        winner.balance += reward - penalty
        self.ledger.append(Settlement(
            round=rnd, ts=ts, task_id=task.task_id,
            agent_id=winner.agent_id,
            reward_received=reward, penalty_applied=penalty,
            cumulative_balance=round(winner.balance, 2),
        ))

    def _population_summary(self) -> str:
        counts: Dict[str, int] = {}
        for a in self.agents:
            counts[a.agent_type] = counts.get(a.agent_type, 0) + 1
        return ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))

    def _print_summary(self):
        events = self.ledger.get_all_events()
        assignments = [e for e in events if e["type"] == "agent_assigned"]
        settlements = [e for e in events if e["type"] == "settlement"]

        # Win rate by type
        wins_by_type: Dict[str, int] = {}
        for a in assignments:
            atype = self.ground_truth[a["agent_id"]]
            wins_by_type[atype] = wins_by_type.get(atype, 0) + 1

        # Accept rate by type
        verifications = [e for e in events if e["type"] == "verification_outcome"]
        accepts_by_type: Dict[str, List[bool]] = {}
        for v in verifications:
            atype = self.ground_truth[v["agent_id"]]
            if atype not in accepts_by_type:
                accepts_by_type[atype] = []
            accepts_by_type[atype].append(v["verdict"] == "accept")

        print(f"\n{'='*50}")
        print(f"SIMULATION COMPLETE — {self.config.name}")
        print(f"{'='*50}")
        print(f"\nAssignment wins by type:")
        for t, count in sorted(wins_by_type.items()):
            print(f"  {t:12s}: {count:4d} wins ({count/self.config.n_rounds*100:.1f}%)")

        print(f"\nAcceptance rate by type:")
        for t, outcomes in sorted(accepts_by_type.items()):
            rate = sum(outcomes) / len(outcomes) if outcomes else 0
            print(f"  {t:12s}: {rate:.1%} ({sum(outcomes)}/{len(outcomes)})")

        print(f"\nBalance by agent:")
        for a in sorted(self.agents, key=lambda x: x.balance, reverse=True)[:5]:
            print(f"  {a.agent_id} ({a.agent_type:12s}): {a.balance:+.2f}")
        print(f"  ...")
        for a in sorted(self.agents, key=lambda x: x.balance)[:3]:
            print(f"  {a.agent_id} ({a.agent_type:12s}): {a.balance:+.2f}")

        print(f"\nOutput: {self.output_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a simulation scenario")
    parser.add_argument("scenario", nargs="?", default="baseline",
                        help=f"Preset name or path to config.json. "
                             f"Presets: {', '.join(PRESETS.keys())}")
    parser.add_argument("--all", action="store_true",
                        help="Run all preset scenarios")
    parser.add_argument("--list", action="store_true",
                        help="List available presets")
    args = parser.parse_args()

    if args.list:
        for name, cfg in PRESETS.items():
            print(f"  {name:25s} {cfg.population_str:15s} {cfg.description}")
    elif args.all:
        for name, cfg in PRESETS.items():
            print(f"\n{'─'*60}")
            Simulation(config=cfg).run()
    else:
        if args.scenario.endswith(".json"):
            cfg = ScenarioConfig.load(Path(args.scenario))
        elif args.scenario in PRESETS:
            cfg = PRESETS[args.scenario]
        else:
            parser.error(f"Unknown preset '{args.scenario}'. "
                         f"Use --list to see available presets.")
        Simulation(config=cfg).run()
