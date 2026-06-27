# Harvy RL Bot Upgrade Specification v2.0

## Overview
Transform Harvy from a low-level RL bot into an autonomous Minecraft survival agent.

## Priority Order
1. Survival Priors + High-Level Skills (highest impact)
2. Training Stability (essential for learning)
3. Better Observations + Memory + Knowledge
4. Intrinsic Curiosity (RND)
5. Self-Generated Goals
6. Evaluation Benchmarks

---

## MODULE 1: Survival Priority Layer (JavaScript)
**File**: `src/bot/mineflayer_bot.js`
**Branch**: `feat/survival-priors`

### Survival Priorities (hard-coded safety layer)
These override RL actions when critical conditions are met.

| Priority | Condition | Action | Override RL? |
|----------|-----------|--------|-------------|
| CRITICAL | Health < 4 | Flee using pathfinder, stop sprinting | Yes |
| HIGH | Health < 8 AND hostile nearby | Flee using pathfinder | Yes |
| HIGH | Food < 14 AND food in inventory | Equip and eat food | Yes |
| MEDIUM | Time > 12000 AND no shelter | Build basic shelter or dig down | Yes |
| MEDIUM | Danger level > 0.7 | Stop moving, assess, flee if needed | Yes |
| LOW | On fire | Jump into water or stop-drop-roll | Yes |
| LOW | In lava | Jump, place water bucket if held | Yes |
| LOW | Suffocating | Break block above head | Yes |

### Implementation
Add a `checkSurvivalPriorities()` function called BEFORE executing any RL action.
It returns either:
- `{ override: false }` — let RL action execute normally
- `{ override: true, action: "action_name", reason: "..." }` — execute survival action instead

The function checks conditions in priority order and returns the highest-priority match.

### New Dependencies
- `mineflayer-pathfinder` (already loaded)

---

## MODULE 2: High-Level Skill Actions (JavaScript + Python)
**JS File**: `src/bot/mineflayer_bot.js`
**Python File**: `src/env/actions.py`
**Branch**: `feat/high-level-skills`

### New High-Level Actions

The bot exposes macro actions that use Mineflayer plugins. Each skill action:
1. Sets a goal/state machine on the JS side
2. Returns immediately (the skill runs across multiple ticks)
3. Can be interrupted by survival priorities or a new action

| Action | Description | Mineflayer API Used | Duration |
|--------|-------------|---------------------|----------|
| `skill_gather_log` | Find and collect oak logs | `collectBlock.collect()` | ~5-15s |
| `skill_gather_stone` | Find and mine stone | `collectBlock.collect()` | ~5-15s |
| `skill_gather_coal` | Find and mine coal ore | `collectBlock.collect()` | ~10-20s |
| `skill_craft_planks` | Convert logs to planks | `bot.craft()` | ~1s |
| `skill_craft_sticks` | Craft sticks | `bot.craft()` | ~1s |
| `skill_craft_pickaxe` | Craft wooden/stone/iron pickaxe | `bot.craft()` | ~2s |
| `skill_craft_sword` | Craft sword | `bot.craft()` | ~2s |
| `skill_craft_crafting_table` | Craft crafting table | `bot.craft()` | ~1s |
| `skill_eat_food` | Find and eat food from inventory | `bot.consume()` | ~2s |
| `skill_equip_best_sword` | Equip best sword from inventory | `bot.equip()` | ~1s |
| `skill_equip_best_pickaxe` | Equip best pickaxe | `bot.equip()` | ~1s |
| `skill_build_shelter` | Build 3x3 shelter | `bot.placeBlock()` | ~10s |
| `skill_flee` | Run away from nearest hostile | `pathfinder.goto()` | ~5s |
| `skill_explore` | Walk to unexplored chunk | `pathfinder.goto()` | ~10s |
| `skill_attack_hostile` | Fight nearest hostile with strafing | `bot.attack()` + movement | ~5s |

### Action Space Update
`src/env/actions.py` gets new action names appended. ACTION_COUNT increases from 25 to 40.

### Skill State Machine
Add a `SkillExecutor` class on the JS side:
- `activeSkill`: currently running skill name or null
- `skillStartTick`: when the skill started
- `skillState`: internal state for the skill (e.g., "moving_to_tree", "breaking")
- `cancelSkill()`: interrupt current skill
- `updateActiveSkill()`: called each physics tick to advance skill state machine

Skills are non-blocking: they set control states each tick and let the normal physics loop continue.

---

## MODULE 3: Training Stability (Python)
**Files**: `src/env/minecraft_env.py`, `src/env/rewards.py`, `src/agent/callbacks.py`
**Branch**: `feat/training-stability`

### Reward Clipping
In `RewardCalculator.compute()`, clip total reward to [-10, 10] range before returning.

```python
def compute(self, obs: dict, reward_signal: dict) -> float:
    reward = 0.0
    # ... existing reward calculations ...
    return np.clip(reward, -10.0, 10.0)
```

### Observation Normalization
Add `ObservationNormalizer` class:
- Running mean and std for each observation key
- Updates online during training
- Normalizes observations before returning from `step()` and `reset()`
- Configurable via `normalize_observations: true` in config

### Frame Stacking
Add `FrameStack` wrapper:
- Stack last 4 observations
- Voxel grids: stack as channels (4, 11, 11, 7)
- Scalar values: concatenate as array of 4
- Used when `frame_stack: 4` in config

### Retry-on-Disconnect
In `MinecraftEnv._connect()`:
- Try to connect up to 3 times with exponential backoff (1s, 2s, 4s)
- Only end episode after all retries fail

### Episode Termination Logging
Add `termination_reason` to info dict:
- `"death"` — bot died
- `"timeout"` — max steps reached
- `"connection_lost"` — TCP connection failed
- `"goal_achieved"` — goal completed

### No-op Bias
Add small negative reward (-0.01) for every non-noop action when danger_level < 0.1.
This encourages the policy to do nothing when safe.

---

## MODULE 4: Better Observations + Memory + Knowledge (Both sides)
**JS File**: `src/bot/mineflayer_bot.js`
**Python Files**: `src/env/observation.py`, `src/env/minecraft_env.py`, new `src/env/knowledge_base.py`
**Branch**: `feat/better-observations`

### New Observations (JS side)

1. **Memory map**: Sparse grid of visited chunks
   - Key: `"visited_chunks"`: list of `{cx, cz, timestamp}` for last 64 visited chunks

2. **Nearby POI list**: Points of interest within 32 blocks
   - Key: `"nearby_pois"`: list of `{type, distance, position}`
   - Types: `tree`, `ore_coal`, `ore_iron`, `ore_diamond`, `chest`, `crafting_table`, `furnace`, `mob_hostile`, `mob_passive`, `water_source`

3. **Block metadata in voxel grid**:
   - Add `"voxel_hardness"`: base64-encoded float array matching voxel_grid shape, with block hardness values
   - Add `"voxel_tool_required"`: base64-encoded uint8 array with required tool type ID

4. **Event stream** (last 10 events):
   - Key: `"recent_events"`: list of `{type, timestamp, data}`
   - Types: `damage_taken`, `mob_attacked`, `block_broken`, `item_collected`, `food_eaten`

### Knowledge Base Module (Python side)
New file: `src/env/knowledge_base.py`

Uses `minecraft-data` Python package to provide:
- `get_recipe(item_name)` -> ingredients dict
- `get_block_drops(block_name, tool_name)` -> drop table
- `get_tool_tier(tool_name)` -> integer tier (wood=1, stone=2, iron=3, diamond=4)
- `get_mob_info(mob_name)` -> {hostile, ranged, health, damage, sunlight_sensitive}
- `get_biome_info(biome_name)` -> {temperature, rainfall, mob_spawns}
- `get_optimal_y(resource)` -> best Y level to find resource

Integrated into reward shaping: bonus for efficient tool use, penalty for breaking blocks with wrong tool.

### Feature Extractor Update
Update `MinecraftFeatureExtractor` to handle new observation modalities:
- POI encoder (similar to entity encoder)
- Voxel hardness channel (concatenate with voxel grid or separate conv)
- Event stream encoder (small RNN or MLP over event history)

---

## MODULE 5: Intrinsic Curiosity via RND (Python)
**New File**: `src/env/rnd_curiosity.py`
**Modified**: `src/env/minecraft_env.py`, `src/train.py`
**Branch**: `feat/rnd-curiosity`

### Random Network Distillation

Two networks:
- **Target network**: Fixed random CNN, processes voxel_grid + self_state
- **Predictor network**: Trained CNN, tries to match target network output

Intrinsic reward = ||target_output - predictor_output||^2

The predictor is trained via gradient descent each step to minimize this error.
Reward is normalized with a running mean/std.

### Integration
- Add `RNDCuriosity` class alongside `IntrinsicMotivation`
- Configurable via `rnd_curiosity: { enabled: true, reward_scale: 1.0, learning_rate: 1e-4 }`
- Updated in `step()` alongside existing intrinsic motivation

---

## MODULE 6: Self-Generated Goals (Python)
**Modified Files**: `src/skills/autonomous_curriculum.py`, `src/env/rewards.py`
**Branch**: `feat/goal-generator`

### Expanded AutonomousCurriculum

Replace simple success-rate sampling with a proper goal generator:

1. **Goal proposal**: Based on current state, propose 3-5 candidate goals
   - Look at inventory: if no logs -> propose `gather_logs`
   - Look at tools: if no pickaxe AND have planks+sticks -> propose `craft_pickaxe`
   - Look at time: if night approaching -> propose `build_shelter`
   - Look at health: if wounded -> propose `eat_food` or `flee`

2. **Subgoal chaining**: When a goal is selected, generate prerequisite chain
   - e.g., `mine_iron` -> needs `craft_stone_pickaxe` -> needs `gather_stone` -> needs `craft_wooden_pickaxe` -> needs `gather_logs`
   - Uses knowledge base for recipe-aware planning

3. **Success estimation**: Estimate success probability for each goal given current state
   - Use historical success rates + current inventory/tools as features
   - Prefer goals with 30-70% estimated success (zone of proximal development)

### Reward Shaping per Goal
Each goal gets specific reward bonuses:
- `gather_logs`: +1 per log collected, +5 for first log
- `craft_pickaxe`: +10 for crafting table, +10 for pickaxe
- `build_shelter`: +5 per block placed, +50 for enclosed shelter
- `survive_night`: +1 per tick alive during night, +20 for surviving full night

---

## MODULE 7: Evaluation Benchmarks (Python)
**New File**: `src/eval_benchmarks.py`
**Modified**: `src/agent/callbacks.py`, `src/dashboard.py`
**Branch**: `feat/eval-benchmarks`

### Benchmark Suite

| Benchmark | Success Criteria | Max Steps |
|-----------|-----------------|-----------|
| `survive_10min` | Health > 0 after 12000 ticks (10 min) | 12000 |
| `gather_16_logs` | Collect >= 16 oak logs | 3000 |
| `craft_pickaxe` | Craft any pickaxe | 2000 |
| `mine_16_stone` | Collect >= 16 cobblestone | 4000 |
| `kill_hostile` | Kill any hostile mob | 3000 |
| `build_shelter` | Place >= 8 blocks forming enclosed space | 2000 |
| `survive_first_night` | Health > 0 after first night | 12000 |

### Implementation
- `BenchmarkRunner` class that runs deterministic evaluation episodes
- Saves best checkpoint per benchmark
- Logs results to JSON
- Dashboard tab showing benchmark scores over time

---

## Interface Contracts

### TCP Protocol (JS <-> Python)

**Observation JSON** (bot -> Python):
```
{
  "self": { health, food, armor, position, yaw, pitch, held_item, inventory_counts },
  "nearby_entities": [{ type, distance, health, hostile }],
  "nearby_blocks": [{ type, position }],
  "nearby_pois": [{ type, distance, position }],
  "voxel_grid": "base64(11x11x7 uint8)",
  "voxel_hardness": "base64(11x11x7 float32)",
  "voxel_tool_required": "base64(11x11x7 uint8)",
  "visited_chunks": [{ cx, cz, timestamp }],
  "recent_events": [{ type, timestamp, data }],
  "environment": { time_of_day, can_see_sky, in_water, on_ground, danger_level },
  "goal": "current_goal_string",
  "reward_signal": { alive_tick, damage_taken, mob_killed, item_mined, item_lost, food_eaten, death },
  "skill_status": { active_skill, skill_progress }
}
```

**Action JSON** (Python -> bot):
```
{ "action": "move_forward", "value": 1.0 }
{ "action": "skill_gather_log", "value": 1.0 }
{ "action": "skill_eat_food", "value": 1.0 }
{ "action": "skill_flee", "value": 1.0 }
{ "action": "reset", "goal": "new_goal" }
```

### Python API Contracts

- `MinecraftEnv.step(action: int) -> (obs, reward, terminated, truncated, info)`
- `RewardCalculator.compute(obs, reward_signal) -> float` (clipped to [-10, 10])
- `IntrinsicMotivation.compute(obs) -> float`
- `RNDCuriosity.compute(obs) -> float`
- `KnowledgeBase.get_recipe(item) -> dict`
- `AutonomousCurriculum.select_goal(state) -> str`
- `ObservationNormalizer.normalize(obs) -> obs`
- `FrameStack.step(obs) -> stacked_obs`

---

## Configuration Updates

Add to `configs/default.yaml`:
```yaml
survival:
  enabled: true
  flee_health_threshold: 8.0
  eat_food_threshold: 14.0
  shelter_time_threshold: 12000
  danger_flee_threshold: 0.7

training:
  reward_clip: 10.0
  normalize_observations: true
  frame_stack: 4
  noop_bias: 0.01

rnd_curiosity:
  enabled: true
  reward_scale: 1.0
  learning_rate: 1.0e-4
  update_interval: 1  # update predictor every N steps

reconnect:
  max_retries: 3
  backoff_base: 1.0  # seconds

benchmarks:
  enabled: true
  eval_freq: 50000
  save_best: true
```

---

## Git Branches

| Branch | Module | Est. Lines Changed |
|--------|--------|-------------------|
| `feat/survival-priors` | JS survival layer | ~200 JS |
| `feat/high-level-skills` | JS skills + Python actions | ~400 JS, ~100 Py |
| `feat/training-stability` | Python env + rewards | ~300 Py |
| `feat/better-observations` | JS obs + Python obs + knowledge | ~400 JS, ~500 Py |
| `feat/rnd-curiosity` | Python RND module | ~200 Py |
| `feat/goal-generator` | Python goal generation | ~300 Py |
| `feat/eval-benchmarks` | Python benchmarks + dashboard | ~400 Py |
