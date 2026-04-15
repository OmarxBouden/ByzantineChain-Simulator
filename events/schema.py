"""Event schema for the simulation.

Every action emits one event. Events are append-only, serialised to JSONL.
Agent's history corresponds to H_i^t = {(b_i^j, e_i^j, s_i^j, v_i^j, tau_i^j)}.
Ground-truth fields (agent_true_type, true_cost) are NEVER shown to the Assessor.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List
import json
import time


@dataclass
class Event:
    type: str = ""
    round: int = 0
    ts: float = 0.0

    def __post_init__(self):
        if self.ts == 0.0:
            self.ts = time.time()

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


@dataclass
class TaskCreated(Event):
    type: str = "task_created"
    task_id: str = ""
    difficulty: float = 0.0
    reward: float = 0.0
    min_effort: float = 0.0


@dataclass
class BidSubmitted(Event):
    type: str = "bid_submitted"
    task_id: str = ""
    agent_id: str = ""
    bid: float = 0.0
    justification: str = ""
    # Ground truth — never visible to Assessor
    true_cost: float = 0.0
    agent_true_type: str = ""


@dataclass
class AgentAssigned(Event):
    type: str = "agent_assigned"
    task_id: str = ""
    agent_id: str = ""
    method: str = "lowest_bid"
    verifier_ids: List[str] = field(default_factory=list)
    rationale: str = ""


@dataclass
class ResultSubmitted(Event):
    type: str = "result_submitted"
    task_id: str = ""
    agent_id: str = ""
    effort: float = 0.0
    quality: float = 0.0
    # Ground truth
    min_effort: float = 0.0
    agent_true_type: str = ""


@dataclass
class VerificationOutcome(Event):
    type: str = "verification_outcome"
    task_id: str = ""
    agent_id: str = ""
    verdict: str = ""  # "accept" or "reject"
    verifier_votes: List[int] = field(default_factory=list)  # 1=accept, 0=reject


@dataclass
class Settlement(Event):
    type: str = "settlement"
    task_id: str = ""
    agent_id: str = ""
    reward_received: float = 0.0
    penalty_applied: float = 0.0
    cumulative_balance: float = 0.0
