# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed
- **Episode-boundary socket race**: `MinecraftEnv.reset()` no longer closes and
  reopens the TCP connection every episode. The bot accepts exactly one Python
  client and destroys extras, so a per-episode reconnect could race the server's
  disconnect handling and kill the fresh connection, hanging training. Sockets
  are now reused across episodes; failed sends/receives mark the socket dead and
  the next `reset()` reconnects transparently.
- **Goal/reward vocabulary mismatch**: `RewardCalculator.set_goal()` now
  canonicalizes legacy goal names (`punch_wood`, `mine_stone`, ...) through
  `LEGACY_GOAL_ALIASES`. Previously a legacy name showed the canonical goal in
  the observation but silently disabled goal reward shaping.
- **Mission Control reset swallowing terminal transitions**: a pending manual
  reset no longer discards a step's real reward/done flags when the episode
  ended on that step; the pending goal is applied on the next reset instead.
- **Dummy-path reset** now primes reward baselines like the live path.

### Added
- `reward_clip` parameter on `RewardCalculator` (default 10.0) -- the final
  reward clamp is now an explicit, configurable choice (death penalty scale).
- Curriculum `reward_shaping` is now wired end-to-end: `CurriculumCallback`
  pushes each stage's shaping into the env, and `RewardCalculator` applies
  entries whose keys match real bot reward signals (unknown keys are logged
  and ignored instead of silently doing nothing). Stage shaping keys in
  `DEFAULT_CURRICULUM` and `configs/curriculum.yaml` updated to signal names.
- Regression tests: socket reuse across episodes, transparent reconnect after
  connection loss, legacy-goal canonicalization, reward clip configurability,
  additive reward shaping.
- `pyproject.toml` (ruff/pytest config), pre-commit hooks,
  `docker-compose.yml` for a one-command reproducible training server.

## [2.0.0] - 2026-09-18

Full protocol and reward-system rewrite.

### Fixed
- Wire protocol is now strict request/response (one observation per
  action/reset); reward events accumulate on the bot and flush on send.
- Observation normalizer no longer z-scores categorical/embedding inputs
  (`CATEGORICAL_KEYS`); embedding indices are clamped defensively in the
  network.
- Single-client guard on the bot's TCP server; eval/benchmark configs disabled
  by default (second env requires a second bot instance).
- Canonical goal vocabulary (`GOAL_TO_ID` + legacy aliases) shared across
  observation, rewards, and curriculum.
- Death/damage penalties consolidated (no double-counting); noop bias now
  penalizes idling instead of rewarding statues.
- Voxel block IDs use a compact mapping (no more uint8 wraparound).
- Benchmark suite: inventory reads, `is_best_checkpoint`, per-benchmark step
  limits.
- Full checkpoint/resume: PPO weights + normalizer + RND + goal history.
- Mission Control server commands actually forwarded to the bot.

### Added
- Test suite: protocol integration tests with a fake bot server, reward
  composition tests, goal-vocabulary consistency tests, normalizer regression
  tests.
