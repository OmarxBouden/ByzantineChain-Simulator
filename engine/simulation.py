"""Main simulation loop.

Runs T rounds of: task → bids → assign → execute → verify → settle.
All events committed to ledger.
"""

import json
import time
from pathlib import Path
from typing import List, Dict

from engine.tasks import TaskGenerator, Task
from engine.agents import Agent, HonestAgent, FreeRiderAgent, StrategicAgent, SelectiveAgent
from engine.ledger import Ledger
from engine.mechanisms import LowestBidAssignment, MajorityVoteVerification
from events.schema import (
    TaskCreated, BidSubmitted, AgentAssigned,
    ResultSubmitted, VerificationOutcome, Settlement,
)


class Simulation:
    def __init__(
        self,
        n_honest: int = 14,
        n_freerider: int = 3,
        n_strategic: int = 2,
        n_selective: int = 1,
        n_rounds: int = 200,
        seed: int = 42,
        output_dir: str = "output",
    ):
        self.n_rounds = n_rounds
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create agents
        self.agents: List[Agent] = []
        idx = 0
        for i in range(n_honest):
            self.agents.append(HonestAgent(f"A{idx:03d}", seed=seed + idx))
            idx += 1
        for i in range(n_freerider):
            self.agents.append(FreeRiderAgent(f"A{idx:03d}", seed=seed + idx))
            idx += 1
        for i in range(n_strategic):
            self.agents.append(StrategicAgent(f"A{idx:03d}", seed=seed + idx))
            idx += 1
        for i in range(n_selective):
            self.agents.append(SelectiveAgent(f"A{idx:03d}", seed=seed + idx))
            idx += 1

        # Components
        self.task_gen = TaskGenerator(seed=seed)
        self.ledger = Ledger(output_path=str(self.output_dir / "events.jsonl"))
        self.assigner = LowestBidAssignment()
        self.verifier = MajorityVoteVerification(seed=seed)

        # Ground truth for evaluation
        self.ground_truth = {
            a.agent_id: a.agent_type for a in self.agents
        }

    def run(self):
        print(f"Starting simulation: {len(self.agents)} agents, {self.n_rounds} rounds")
        print(f"Population: {self._population_summary()}")

        for rnd in range(1, self.n_rounds + 1):
            self._run_round(rnd)

        # Save ground truth
        gt_path = self.output_dir / "ground_truth.json"
        with open(gt_path, "w") as f:
            json.dump(self.ground_truth, f, indent=2)

        # Print summary
        self._print_summary()

    def _run_round(self, rnd: int):
        ts = time.time()

        # 1. Generate task
        task = self.task_gen.generate()
        self.ledger.append(TaskCreated(
            round=rnd, ts=ts, task_id=task.task_id,
            difficulty=task.difficulty, reward=task.reward,
            min_effort=task.min_effort,
        ))

        # 2. Collect bids (all agents bid on every task for simplicity)
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
            penalty = task.reward * 0.5  # slash 50% of task reward

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
        print(f"SIMULATION COMPLETE — {self.n_rounds} rounds")
        print(f"{'='*50}")
        print(f"\nAssignment wins by type:")
        for t, count in sorted(wins_by_type.items()):
            print(f"  {t:12s}: {count:4d} wins ({count/self.n_rounds*100:.1f}%)")

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

        print(f"\nEvents written to: {self.ledger.output_path}")
        print(f"Ground truth: {self.output_dir / 'ground_truth.json'}")


if __name__ == "__main__":
    sim = Simulation(
        n_honest=14,
        n_freerider=3,
        n_strategic=2,
        n_selective=1,
        n_rounds=200,
        seed=42,
    )
    sim.run()
