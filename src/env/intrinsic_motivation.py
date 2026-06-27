"""Intrinsic motivation / curiosity rewards for Harvy.

These rewards encourage exploration and discovery even when no human is
telling Harvy what to do. It gets a small bonus for:
  - Visiting a new 4x4 world chunk for the first time
  - Picking up a new item type for the first time
"""

import numpy as np
from typing import Dict, Any, Set, Tuple


class IntrinsicMotivation:
    """Adds curiosity-based rewards to encourage autonomous exploration."""

    def __init__(
        self,
        chunk_size: int = 4,
        new_chunk_bonus: float = 1.0,
        new_item_bonus: float = 2.0,
    ):
        self.chunk_size = chunk_size
        self.new_chunk_bonus = new_chunk_bonus
        self.new_item_bonus = new_item_bonus
        self.visited_chunks: Set[Tuple[int, int]] = set()
        self.discovered_items: Set[int] = set()

    def reset(self) -> None:
        """Clear episodic curiosity state.

        We keep global chunk and item discovery across episodes so Harvy is
        rewarded for exploring the world, not just running in circles within
        one episode.
        """
        # Currently no-op; global discovery persists.
        pass

    def compute(self, obs: Dict[str, Any]) -> float:
        """Return intrinsic reward for the current observation."""
        reward = 0.0

        # Position-based exploration bonus
        position = obs.get("self_position")
        if position is not None and len(position) >= 3:
            cx = int(np.floor(position[0] / self.chunk_size))
            cz = int(np.floor(position[2] / self.chunk_size))
            chunk = (cx, cz)
            if chunk not in self.visited_chunks:
                reward += self.new_chunk_bonus
                self.visited_chunks.add(chunk)

        # Item-collection novelty bonus
        inventory = obs.get("inventory")
        if inventory is not None:
            inventory = np.asarray(inventory)
            nonzero_ids = np.nonzero(inventory > 0)[0]
            for item_id in nonzero_ids:
                if int(item_id) not in self.discovered_items:
                    reward += self.new_item_bonus
                    self.discovered_items.add(int(item_id))

        return reward
