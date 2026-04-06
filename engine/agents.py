"""Each agent implements: bid(task) -> (bid, justification, true_cost)
                       execute(task) -> (effort, quality)
"""

import random
from abc import ABC, abstractmethod
from engine.tasks import Task


class Agent(ABC):
    def __init__(self, agent_id: str, seed: int = 0):
        self.agent_id = agent_id
        self.rng = random.Random(seed + hash(agent_id))
        self.balance: float = 0.0

    @property
    @abstractmethod
    def agent_type(self) -> str:
        ...

    @abstractmethod
    def bid(self, task: Task) -> tuple[float, str, float]:
        """Returns (bid_amount, justification, true_cost)."""
        ...

    @abstractmethod
    def execute(self, task: Task) -> tuple[float, float]:
        """Returns (effort_expended, result_quality in [0,1])."""
        ...

    def __repr__(self):
        return f"{self.agent_type}({self.agent_id})"


class HonestAgent(Agent):
    @property
    def agent_type(self) -> str:
        return "honest"

    def bid(self, task: Task) -> tuple[float, str, float]:
        true_cost = task.min_effort * self.rng.uniform(0.9, 1.1)
        bid = true_cost  # honest: bid = true cost
        justification = "I can complete this task at standard cost."
        return round(bid, 2), justification, round(true_cost, 2)

    def execute(self, task: Task) -> tuple[float, float]:
        effort = task.min_effort * self.rng.uniform(1.0, 1.2)  # full effort
        quality = min(1.0, effort / task.min_effort) * self.rng.uniform(0.85, 1.0)
        return round(effort, 2), round(quality, 3)


class FreeRiderAgent(Agent):
    """Always underbids, always shirks."""

    @property
    def agent_type(self) -> str:
        return "free_rider"

    def bid(self, task: Task) -> tuple[float, str, float]:
        true_cost = task.min_effort * self.rng.uniform(0.9, 1.1)
        bid = true_cost * self.rng.uniform(0.3, 0.5)  # underbid 50-70%
        justification = "Efficient pipeline allows me to do this cheaply."
        return round(bid, 2), justification, round(true_cost, 2)

    def execute(self, task: Task) -> tuple[float, float]:
        effort = task.min_effort * self.rng.uniform(0.1, 0.3)  # minimal effort
        quality = (effort / task.min_effort) * self.rng.uniform(0.3, 0.6)
        return round(effort, 2), round(min(1.0, quality), 3)


class StrategicAgent(Agent):
    """Shirks when expected gain > expected penalty."""

    def __init__(self, agent_id: str, seed: int = 0, detection_estimate: float = 0.3):
        super().__init__(agent_id, seed)
        self.detection_estimate = detection_estimate

    @property
    def agent_type(self) -> str:
        return "strategic"

    def bid(self, task: Task) -> tuple[float, str, float]:
        true_cost = task.min_effort * self.rng.uniform(0.9, 1.1)
        # Moderate underbid — not as aggressive as free rider
        bid = true_cost * self.rng.uniform(0.6, 0.85)
        justification = "I have relevant experience that reduces my cost."
        return round(bid, 2), justification, round(true_cost, 2)

    def execute(self, task: Task) -> tuple[float, float]:
        # Shirking condition: E[gain] > Pr[detected] * penalty
        cost_saving = task.min_effort * 0.6
        expected_penalty = self.detection_estimate * task.reward * 1.5
        if cost_saving > expected_penalty:
            # Shirk
            effort = task.min_effort * self.rng.uniform(0.2, 0.5)
            quality = (effort / task.min_effort) * self.rng.uniform(0.4, 0.7)
        else:
            # Honest execution
            effort = task.min_effort * self.rng.uniform(0.95, 1.1)
            quality = min(1.0, effort / task.min_effort) * self.rng.uniform(0.85, 1.0)
        return round(effort, 2), round(min(1.0, quality), 3)


class SelectiveAgent(Agent):
    """Honest on low-value tasks, shirks on high-value ones."""

    @property
    def agent_type(self) -> str:
        return "selective"

    def bid(self, task: Task) -> tuple[float, str, float]:
        true_cost = task.min_effort * self.rng.uniform(0.9, 1.1)
        if task.is_high_value:
            bid = true_cost * self.rng.uniform(0.4, 0.6)
        else:
            bid = true_cost * self.rng.uniform(0.95, 1.05)
        justification = "Standard pricing based on task requirements."
        return round(bid, 2), justification, round(true_cost, 2)

    def execute(self, task: Task) -> tuple[float, float]:
        if task.is_high_value:
            effort = task.min_effort * self.rng.uniform(0.15, 0.35)
            quality = (effort / task.min_effort) * self.rng.uniform(0.3, 0.5)
        else:
            effort = task.min_effort * self.rng.uniform(1.0, 1.15)
            quality = min(1.0, effort / task.min_effort) * self.rng.uniform(0.85, 1.0)
        return round(effort, 2), round(min(1.0, quality), 3)
