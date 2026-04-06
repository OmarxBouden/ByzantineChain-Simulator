"""Scenario configuration — defines all tunable simulation parameters."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ScenarioConfig:
    name: str = "default"
    description: str = ""

    # Population
    n_honest: int = 14
    n_freerider: int = 3
    n_strategic: int = 2
    n_selective: int = 1

    # Simulation
    n_rounds: int = 200
    seed: int = 42

    # Verification
    quality_threshold: float = 0.5
    num_verifiers: int = 3

    # Settlement
    penalty_ratio: float = 0.5  # fraction of reward slashed on reject

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ScenarioConfig":
        with open(path) as f:
            return cls(**json.load(f))

    @property
    def population_str(self) -> str:
        parts = []
        if self.n_honest:
            parts.append(f"{self.n_honest}h")
        if self.n_freerider:
            parts.append(f"{self.n_freerider}fr")
        if self.n_strategic:
            parts.append(f"{self.n_strategic}st")
        if self.n_selective:
            parts.append(f"{self.n_selective}sel")
        return "-".join(parts)

    @property
    def total_agents(self) -> int:
        return self.n_honest + self.n_freerider + self.n_strategic + self.n_selective


# ── Preset scenarios ─────────────────────────────────────────────────────────

PRESETS = {
    "baseline": ScenarioConfig(
        name="baseline",
        description="Default population, naive mechanisms",
    ),
    "no-freeriders": ScenarioConfig(
        name="no-freeriders",
        description="All honest agents — control group",
        n_honest=20, n_freerider=0, n_strategic=0, n_selective=0,
    ),
    "heavy-adversarial": ScenarioConfig(
        name="heavy-adversarial",
        description="50% adversarial population",
        n_honest=10, n_freerider=5, n_strategic=3, n_selective=2,
    ),
    "strategic-only": ScenarioConfig(
        name="strategic-only",
        description="Only strategic adversaries — harder to detect",
        n_honest=14, n_freerider=0, n_strategic=6, n_selective=0,
    ),
    "selective-only": ScenarioConfig(
        name="selective-only",
        description="Only selective adversaries — honest on cheap tasks",
        n_honest=14, n_freerider=0, n_strategic=0, n_selective=6,
    ),
    "stress-test": ScenarioConfig(
        name="stress-test",
        description="Overwhelming adversarial majority",
        n_honest=4, n_freerider=6, n_strategic=6, n_selective=4,
        n_rounds=500,
    ),
    "weak-verification": ScenarioConfig(
        name="weak-verification",
        description="Lower quality threshold makes verification easier to pass",
        quality_threshold=0.3, num_verifiers=1,
    ),
    "strict-verification": ScenarioConfig(
        name="strict-verification",
        description="High threshold and more verifiers",
        quality_threshold=0.7, num_verifiers=5,
    ),
}
