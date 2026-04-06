"""On-chain ledger L — append-only ordered log of events."""

import json
from pathlib import Path
from typing import List, Optional
from events.schema import Event


class Ledger:
    def __init__(self, output_path: str = "output/events.jsonl"):
        self.events: List[Event] = []
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Clear previous run
        self.output_path.write_text("")

    def append(self, event: Event):
        self.events.append(event)
        with open(self.output_path, "a") as f:
            f.write(event.to_json() + "\n")

    def get_agent_history(self, agent_id: str) -> List[dict]:
        """Return on-chain history for agent (no ground-truth fields)."""
        history = []
        for e in self.events:
            d = json.loads(e.to_json())
            if d.get("agent_id") == agent_id:
                # Strip ground-truth fields
                d.pop("true_cost", None)
                d.pop("agent_true_type", None)
                d.pop("min_effort", None)
                history.append(d)
        return history

    def get_all_events(self, event_type: Optional[str] = None) -> List[dict]:
        result = []
        for e in self.events:
            d = json.loads(e.to_json())
            if event_type is None or d["type"] == event_type:
                result.append(d)
        return result
