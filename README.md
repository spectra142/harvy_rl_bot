# Harvy RL Bot

A Minecraft reinforcement-learning agent built with Python (Stable-Baselines3 PPO)
and a Mineflayer bot. Harvy learns to survive, gather, craft, build, and fight in
Minecraft through a custom Gymnasium environment that talks to the bot over a
strict request/response TCP protocol. A React **Mission Control** dashboard lets
you start/pause/stop training, send manual actions, tune hyperparameters live,
and watch bot telemetry.

## Features

- 40-action discrete space (25 low-level movement/look/slot actions + 15 high-level skills)
- Autonomous goal selection with zone-of-proximal-development curriculum
- Guided curriculum mode (human-defined stages) as an alternative
- Intrinsic motivation (novel chunk/item discovery) + optional RND curiosity
- Frame-stacked 3D voxel CNN observations (11×11×7)
- Knowledge-base-driven reward shaping (correct tool, mining efficiency)
- Skill FSMs on the bot: gather, craft, eat, equip, build shelter, flee, explore, attack
- Survival priority layer with **honest reporting** (`executed_action` / `action_overridden`)
- Mission Control dashboard (FastAPI + WebSocket + React)
- Benchmark suite for skill evaluation
- Full checkpointing: PPO weights + observation normalizer + RND + goal history

## How it works

```
┌─────────────┐   JSON/TCP :9876    ┌────────────────┐   Minecraft protocol   ┌────────────┐
│ Python PPO  │ ◄────────────────── │ Mineflayer bot │ ◄───────────────────── │ MC server  │
│ (training)  │  1 action → 1 obs   │ (Node.js)      │                        │ (Java 26.x)│
└─────────────┘                     └────────────────┘                        └────────────┘
```

The protocol is **strict request/response**: Python sends exactly one action (or
`reset`/`set_goal`), the bot executes it and replies with exactly one observation
on the next physics tick. Reward events (damage, kills, crafts, placements,
distance moved) accumulate between requests and are flushed on send — no signal
is ever lost, regardless of how slow the Python side steps.

## Requirements

| Component | Version |
|---|---|
| Python | 3.10+ (tested on 3.14) |
| Node.js | 18+ (tested on 26) |
| Java | 21+ (only if you host the server; 26.x servers want Java 25) |
| Minecraft server | Java Edition, any version supported by minecraft-data (up to **26.1**) |
| OS | Linux, macOS, or Windows |

## 1. Minecraft server setup

The bot joins as an **offline-mode** player, so the server must allow it:

```properties
# server.properties
online-mode=false
```

> [!WARNING]
> `online-mode=false` means no account verification. Only do this on a LAN or
> whitelisted server.

For full episode resets (the bot clears its inventory and respawns at spawn via
`/clear` + `/kill`), the bot account must be **opped**:

```
/op harvy        # from the server console (default bot name is "harvy")
```

If the bot is not opped, everything still works — episodes just continue from
wherever the bot is (reward baselines are primed so this doesn't corrupt rewards).
Set `MC_RESET_COMMANDS=0` to disable the reset commands entirely.

### Easiest option: vanilla server

1. Download `server.jar` for a supported version (e.g. 26.1) from
   [minecraft.net](https://www.minecraft.net/en-us/download/server).
2. Run `java -Xmx4G -jar server.jar nogui` once, accept the EULA in `eula.txt`.
3. Set `online-mode=false` in `server.properties`, restart, op the bot.

## 2. Install Harvy

### Linux / macOS

```bash
git clone https://github.com/spectra142/harvy_rl_bot.git
cd harvy_rl_bot

# Python side
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Mineflayer bot
cd src/bot && npm install && cd ../..

# Mission Control dashboard (optional, needed for the web UI)
cd mission_control/dashboard/app && npm install && npm run build && cd ../../..
```

### Windows (PowerShell)

```powershell
git clone https://github.com/spectra142/harvy_rl_bot.git
cd harvy_rl_bot

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

cd src\bot; npm install; cd ..\..

cd mission_control\dashboard\app; npm install; npm run build; cd ..\..\..
```

## 3. Run the bot

Point the bot at your server (any machine that can reach it):

**Linux / macOS:**

```bash
cd src/bot
MC_HOST=192.168.1.10 MC_PORT=25565 node mineflayer_bot.js
```

**Windows (PowerShell):**

```powershell
cd src\bot
$env:MC_HOST="192.168.1.10"; $env:MC_PORT="25565"; node mineflayer_bot.js
```

You should see `Bot logged in. Minecraft version: ...`, `Bot spawned in world`,
and `TCP server listening on port 9876`.

### Bot environment variables

| Variable | Default | Meaning |
|---|---|---|
| `MC_HOST` | `localhost` | Minecraft server address |
| `MC_PORT` | `25565` | Minecraft server port |
| `MC_USERNAME` | `harvy` | Bot's in-game name |
| `MC_VERSION` | auto-detect | Pin a Minecraft version if auto-detect fails |
| `TCP_PORT` | `9876` | Port for the Python ↔ bot protocol |
| `MC_RESET_COMMANDS` | `1` | Set `0` to disable `/clear` + `/kill` episode resets |
| `MC_CONFIG_WORKAROUND` | off | Set `1` only for old servers with a broken config phase |
| `ENABLE_VIEWER` | *(off)* | Set `1` to enable the prismarine-viewer POV (needs the native `canvas` module) |
| `VIEWER_PORT` | `3007` | POV viewer port |
| `DEBUG` | off | Set `1` for verbose logging |

## 4. Train

### Option A: command line

```bash
source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
python src/train.py --config configs/default.yaml --output_dir ./checkpoints
```

Useful flags:

```bash
python src/train.py --guided                 # fixed human curriculum instead of autonomous goals
python src/train.py --no-autonomous          # same as --guided
python src/train.py --total_timesteps 500000 # quick run
python src/train.py --device cuda            # GPU training
python src/train.py --resume checkpoints/final_model.zip  # resume (restores normalizer/RND/goal state too)
```

Checkpoints land in `./checkpoints/` (`harvy_*_steps.zip` plus
`*.normalizer.pkl` / `*.rnd.pt` sidecars — keep them together when moving models).

### Option B: Mission Control dashboard

```bash
./scripts/start_mission_control.sh    # Windows: scripts\start_mission_control.bat
# then open http://localhost:9877
```

Start/pause/stop training, send manual actions, change hyperparameters live,
and run server commands (they are actually forwarded to the bot).

## 5. Inference (watch a trained bot play)

```bash
python src/infer.py --model checkpoints/final_model.zip --episodes 5 --render
```

## 6. Benchmarks

The benchmark suite runs a second environment, so it needs a **second bot
instance** on a different TCP port (the bot accepts exactly one Python client):

```bash
# terminal 2: second bot (different username + TCP port)
cd src/bot
MC_USERNAME=harvy_eval TCP_PORT=9877 node mineflayer_bot.js
```

Then enable it in `configs/default.yaml` (`callbacks.eval_freq` and/or
`benchmarks.enabled: true`) with the environment `port: 9877`, or run
`src/eval_benchmarks.py` directly against a trained model.

## 7. Tests

```bash
source .venv/bin/activate
python -m pytest tests/ -q
```

Covers the observation space, frame stacking, normalizer (including the
categorical-key embedding crash regression), reward shaping, goal vocabulary
consistency across all three goal systems, Mission Control, and a fake-bot
integration test of the request/response wire protocol.

Live smoke tests (need a running server + bot):

```bash
python scripts/live_smoke_test.py 40     # random actions + PPO forward pass
python scripts/skill_live_test.py        # sustains skill_gather_log, checks rewards
python scripts/train_smoke_test.py       # tiny end-to-end PPO training run
```

## Project layout

```
.
├── configs/                  # Training configuration (default.yaml, curriculum.yaml)
├── mission_control/          # React dashboard
├── scripts/                  # Launchers + live smoke tests + mc_ssh.py helper
├── src/
│   ├── agent/                # PPO agent, networks, callbacks
│   ├── bot/                  # Mineflayer bot (mineflayer_bot.js)
│   ├── env/                  # Gymnasium env, observations, rewards, normalizer, RND
│   ├── mission_control/      # FastAPI/WebSocket bridge
│   ├── skills/               # Autonomous + guided curricula, skill policies
│   ├── eval_benchmarks.py    # Benchmark suite
│   ├── infer.py              # Run a trained model
│   └── train.py              # CLI training entry point
└── tests/
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| Bot kicked with "Failed to verify username" | Set `online-mode=false` on the server |
| Registry/dimension codec crash on login | Server version newer than minecraft-data supports (max 26.1); don't set `MC_CONFIG_WORKAROUND=1` |
| `Connection refused` on port 9876 | Start the bot first; check `TCP_PORT` matches |
| Training env freezes when an eval starts | You connected a second env to one bot — run a second bot on another `TCP_PORT` |
| Episode resets don't clear inventory | Op the bot (`/op harvy`) or accept carry-over (baselines handle it) |
| `index out of range` in embedding | You're on old code — pull; categorical keys are no longer normalized |

## License

MIT — see [LICENSE](./LICENSE).
