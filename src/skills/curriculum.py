"""Curriculum learning manager for progressive skill training."""

import yaml
import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass


@dataclass
class CurriculumStage:
    """Defines one stage of the curriculum."""

    name: str
    goal: str
    max_episode_steps: int
    success_threshold: float
    reward_shaping: Dict[str, float]
    description: str


class CurriculumManager:
    """Manages curriculum learning from simple to complex tasks."""

    DEFAULT_CURRICULUM = [
        CurriculumStage(
            name="basic_movement",
            goal="explore",
            max_episode_steps=200,
            success_threshold=0.8,
            reward_shaping={"distance_reward": 1.0, "fall_penalty": -5.0},
            description="Learn to walk, turn, jump, and look around",
        ),
        CurriculumStage(
            name="punch_wood",
            goal="gather_logs",
            max_episode_steps=500,
            success_threshold=0.7,
            reward_shaping={"log_reward": 2.0, "pickup_reward": 1.0},
            description="Find and punch trees to collect logs",
        ),
        CurriculumStage(
            name="craft_pickaxe",
            goal="craft_pickaxe",
            max_episode_steps=500,
            success_threshold=0.7,
            reward_shaping={
                "craft_reward": 5.0,
                "plank_reward": 1.0,
                "stick_reward": 1.0,
            },
            description="Craft a crafting table and wooden pickaxe",
        ),
        CurriculumStage(
            name="mine_stone",
            goal="gather_stone",
            max_episode_steps=1000,
            success_threshold=0.6,
            reward_shaping={"stone_reward": 1.0, "pickaxe_bonus": 3.0},
            description="Mine cobblestone with a pickaxe",
        ),
        CurriculumStage(
            name="fight_passive",
            goal="eat_food",
            max_episode_steps=500,
            success_threshold=0.6,
            reward_shaping={"kill_reward": 10.0, "hit_reward": 2.0},
            description="Fight and kill passive mobs (cows, pigs)",
        ),
        CurriculumStage(
            name="fight_hostile",
            goal="fight_hostile",
            max_episode_steps=1000,
            success_threshold=0.5,
            reward_shaping={
                "kill_reward": 20.0,
                "hit_reward": 5.0,
                "health_penalty": -2.0,
            },
            description="Fight and kill hostile mobs (zombies, skeletons)",
        ),
        CurriculumStage(
            name="build_shelter",
            goal="build_shelter",
            max_episode_steps=1000,
            success_threshold=0.5,
            reward_shaping={
                "block_placed": 5.0,
                "shelter_complete": 50.0,
                "wasted_block": -1.0,
            },
            description="Build a 5x5 enclosed shelter",
        ),
        CurriculumStage(
            name="survive_first_night",
            goal="survive_first_night",
            max_episode_steps=2000,
            success_threshold=0.6,
            reward_shaping={
                "survive_reward": 1.0,
                "mob_avoidance": 2.0,
                "shelter_bonus": 10.0,
            },
            description="Survive from spawn through the first night",
        ),
        CurriculumStage(
            name="mine_iron",
            goal="gather_iron",
            max_episode_steps=3000,
            success_threshold=0.4,
            reward_shaping={"iron_reward": 5.0, "coal_reward": 2.0, "depth_bonus": 1.0},
            description="Find and mine iron ore deep underground",
        ),
        CurriculumStage(
            name="full_survival",
            goal="survive",
            max_episode_steps=6000,
            success_threshold=0.5,
            reward_shaping={"composite_score": 1.0},
            description="Full open-world survival episode",
        ),
        CurriculumStage(
            name="pvp_combat",
            goal="fight_hostile",
            max_episode_steps=2000,
            success_threshold=0.4,
            reward_shaping={
                "player_kill": 50.0,
                "player_hit": 10.0,
                "survival_bonus": 5.0,
            },
            description="Fight and defeat other players/bots in PvP",
        ),
    ]

    def __init__(
        self,
        curriculum: Optional[List[CurriculumStage]] = None,
        auto_advance: bool = True,
        advance_after_n_successes: int = 5,
    ):
        self.stages = curriculum or self.DEFAULT_CURRICULUM
        self.current_stage_idx = 0
        self.auto_advance = auto_advance
        self.advance_after_n_successes = advance_after_n_successes
        self.success_history = []
        self.stage_episode_counts = [0] * len(self.stages)
        self.stage_success_counts = [0] * len(self.stages)

    @property
    def current_stage(self) -> CurriculumStage:
        """Return the current curriculum stage."""
        return self.stages[self.current_stage_idx]

    @property
    def is_complete(self) -> bool:
        """Return True if all curriculum stages are complete."""
        return self.current_stage_idx >= len(self.stages) - 1

    def get_current_config(self) -> Dict:
        """Get configuration for the current curriculum stage."""
        stage = self.current_stage
        return {
            "goal": stage.goal,
            "max_episode_steps": stage.max_episode_steps,
            "reward_shaping": stage.reward_shaping,
            "name": stage.name,
            "description": stage.description,
        }

    def record_episode(self, success: bool, score: float) -> None:
        """Record episode outcome and potentially advance curriculum."""
        self.stage_episode_counts[self.current_stage_idx] += 1
        if success:
            self.stage_success_counts[self.current_stage_idx] += 1

        self.success_history.append(
            {
                "stage": self.current_stage.name,
                "success": success,
                "score": score,
            }
        )

        # Check if we should advance
        if self.auto_advance and not self.is_complete:
            successes = self.stage_success_counts[self.current_stage_idx]
            episodes = self.stage_episode_counts[self.current_stage_idx]
            success_rate = successes / max(episodes, 1)

            if episodes >= 10 and success_rate >= self.current_stage.success_threshold:
                if successes >= self.advance_after_n_successes:
                    self.advance()

    def advance(self) -> CurriculumStage:
        """Manually advance to the next curriculum stage."""
        if self.current_stage_idx < len(self.stages) - 1:
            self.current_stage_idx += 1
            print(
                f"Curriculum advanced to stage {self.current_stage_idx}: "
                f"{self.current_stage.name} -- {self.current_stage.description}"
            )
        return self.current_stage

    def regress(self) -> CurriculumStage:
        """Regress to previous stage if current is too hard."""
        if self.current_stage_idx > 0:
            self.current_stage_idx -= 1
            print(
                f"Curriculum regressed to stage {self.current_stage_idx}: "
                f"{self.current_stage.name}"
            )
        return self.current_stage

    def get_progress(self) -> Dict:
        """Get curriculum progress summary."""
        return {
            "current_stage": self.current_stage_idx,
            "total_stages": len(self.stages),
            "stage_name": self.current_stage.name,
            "stage_episodes": self.stage_episode_counts[self.current_stage_idx],
            "stage_successes": self.stage_success_counts[self.current_stage_idx],
            "success_rate": (
                self.stage_success_counts[self.current_stage_idx]
                / max(self.stage_episode_counts[self.current_stage_idx], 1)
            ),
            "is_complete": self.is_complete,
        }

    @classmethod
    def from_yaml(cls, path: str) -> "CurriculumManager":
        """Load curriculum from YAML configuration file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        stages = [CurriculumStage(**stage) for stage in data["stages"]]
        return cls(stages)

    def to_yaml(self, path: str) -> None:
        """Save curriculum configuration to YAML."""
        data = {
            "stages": [
                {
                    "name": s.name,
                    "goal": s.goal,
                    "max_episode_steps": s.max_episode_steps,
                    "success_threshold": s.success_threshold,
                    "reward_shaping": s.reward_shaping,
                    "description": s.description,
                }
                for s in self.stages
            ]
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
