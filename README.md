# Harvy RL Bot

A work-in-progress Minecraft reinforcement-learning agent built with Python, Stable-Baselines3, and Mineflayer.

Harvy learns to survive in Minecraft through a custom Gymnasium environment that talks to a Mineflayer bot over a TCP socket. A React Mission Control dashboard lets you start/pause/stop training, send manual actions, tune hyperparameters, and watch live bot telemetry.

> **Status:** Work in progress. Contributions, issues, and pull requests are welcome.

## Features

- 40-action discrete space (low-level movement + high-level skills)
- Autonomous goal selection and curriculum support
- Intrinsic motivation + optional RND curiosity
- Real-time Mission Control dashboard
- prismarine-viewer POV feed
- Frame-stacked voxel observations
- Knowledge-base-driven reward shaping

## Quick start

### 1. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install the Mineflayer bot

```bash
cd src/bot
npm install
```

### 3. Build the dashboard

```bash
cd mission_control/dashboard/app
npm install
npm run build
```

### 4. Start everything

1. Start a Minecraft Java server (version ~1.21.1) or use a local/offline server.
2. Point the bot at it:
   ```bash
   cd src/bot
   export MC_HOST=localhost  # change to your server IP
   export MC_PORT=25565
   node mineflayer_bot.js
   ```
3. Start the Mission Control bridge:
   ```bash
   cd /path/to/harvy-rl-bot
   ./scripts/start_mission_control.sh
   ```
4. Open the dashboard:
   ```
   http://localhost:9877
   ```
   If the bridge runs on a different machine, replace `localhost` with that machine's IP.

### 5. Train

From the dashboard, connect to `ws://localhost:9877/ws` and click **Start Training**.

Or run training from the command line:

```bash
source .venv/bin/activate
python src/train.py --config configs/default.yaml --output_dir ./checkpoints --device cpu
```

## Project layout

```
.
├── configs/              # Training configuration
├── mission_control/      # React dashboard + FastAPI bridge
│   ├── dashboard/app/
│   └── README.md
├── scripts/              # Convenience launchers
├── src/
│   ├── agent/            # PPO agent and networks
│   ├── bot/              # Mineflayer bot
│   ├── env/              # Gymnasium environment
│   ├── mission_control/  # FastAPI/WebSocket bridge
│   ├── skills/           # Curriculum and goal generators
│   └── train.py          # CLI training script
└── tests/                # Unit tests
```

## Configuration

Edit `configs/default.yaml` to change training hyperparameters, observation normalization, frame stacking, and RND curiosity settings.

## Environment variables

- `MC_HOST` / `MC_PORT` — Minecraft server address for the bot
- `TCP_PORT` — Python ↔ Mineflayer TCP port (default `9876`)
- `VIEWER_PORT` — prismarine-viewer port (default `3007`)
- `ENABLE_VIEWER=0` — disable the POV viewer
- `VITE_POV_URL` — dashboard POV iframe URL (default `http://localhost:3007`)

## Running tests

```bash
source .venv/bin/activate
python -m pytest tests/ -q
```

## License

MIT — see [LICENSE](./LICENSE).
