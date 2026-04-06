"""Task generator — configurable difficulty and reward distributions."""

import random
from dataclasses import dataclass


@dataclass
class Task:
    task_id: str
    difficulty: float   # determines min_effort
    reward: float
    min_effort: float   # e*_j — minimum effort for correct result

    @property
    def is_high_value(self) -> bool:
        return self.reward > 15.0


class TaskGenerator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.counter = 0

    def generate(self) -> Task:
        self.counter += 1
        difficulty = self.rng.uniform(1.0, 10.0)
        reward = difficulty * self.rng.uniform(1.5, 3.0)
        min_effort = difficulty * self.rng.uniform(8.0, 12.0)  # tokens of compute
        return Task(
            task_id=f"T{self.counter:04d}",
            difficulty=round(difficulty, 2),
            reward=round(reward, 2),
            min_effort=round(min_effort, 2),
        )
