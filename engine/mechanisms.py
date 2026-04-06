"""Assignment and verification mechanisms.

Phase 1: naive lowest-bidder + majority-vote verification.
Phase 2+: Assessor replaces assignment; (alpha,beta) replaces verification.
"""

import random
from typing import List, Tuple
from engine.agents import Agent
from engine.tasks import Task


class LowestBidAssignment:
    """Naive baseline: assign to lowest bidder."""

    def select(
        self,
        task: Task,
        bids: List[Tuple[Agent, float, str, float]],  # (agent, bid, justification, true_cost)
    ) -> Tuple[Agent, str]:
        """Returns (selected_agent, rationale)."""
        if not bids:
            raise ValueError("No bids received")
        # Sort by bid, pick lowest
        bids_sorted = sorted(bids, key=lambda x: x[1])
        winner = bids_sorted[0][0]
        rationale = f"Lowest bid: {bids_sorted[0][1]:.2f}"
        return winner, rationale


class MajorityVoteVerification:
    """V1 baseline: single-round majority vote.

    Quality threshold determines accept/reject per verifier.
    """

    def __init__(self, quality_threshold: float = 0.5, num_verifiers: int = 3, seed: int = 42):
        self.quality_threshold = quality_threshold
        self.num_verifiers = num_verifiers
        self.rng = random.Random(seed)

    def select_verifiers(self, agents: List[Agent], exclude: Agent) -> List[str]:
        """Pick random verifiers, excluding the executor."""
        candidates = [a for a in agents if a.agent_id != exclude.agent_id]
        k = min(self.num_verifiers, len(candidates))
        selected = self.rng.sample(candidates, k)
        return [a.agent_id for a in selected]

    def verify(self, quality: float) -> Tuple[str, List[int]]:
        """Deterministic verification: quality >= threshold → accept.

        Each verifier has slight noise to simulate stochasticity.
        """
        votes = []
        for _ in range(self.num_verifiers):
            noise = self.rng.uniform(-0.1, 0.1)
            vote = 1 if (quality + noise) >= self.quality_threshold else 0
            votes.append(vote)
        accept_count = sum(votes)
        verdict = "accept" if accept_count > self.num_verifiers / 2 else "reject"
        return verdict, votes
