# Mission Control Dashboard

The **Mission Control Dashboard** is a real-time web interface for the Harvy Minecraft RL agent. It is served by a FastAPI bridge (`src/mission_control/server.py`) and lets you start/pause/stop training, send manual actions, adjust hyperparameters, reset the environment, and view live bot state, metrics, and terminal output.

## Architecture

- `src/mission_control/server.py` — FastAPI app that serves the built React dashboard and exposes a `/ws` WebSocket endpoint.
- `src/mission_control/training_runner.py` — `TrainingManager` that runs training in a background thread and forwards messages to connected dashboards.
- `src/mission_control/env_wrapper.py` — Gymnasium wrapper that injects manual actions and environment resets from the dashboard.
- `src/mission_control/dashboard_callback.py` — Stable-Baselines3 callback that streams metrics to the manager.
- `mission_control/dashboard/app/` — React + Vite + Tailwind CSS frontend.

## Starting Mission Control

1. **Start the Minecraft bot** (Mineflayer bridge):
   ```bash
   cd src/bot
   node mineflayer_bot.js
   ```

2. **Start the Mission Control bridge** from the project root:
   ```bash
   ./scripts/start_mission_control.sh
   ```
   On Windows use:
   ```cmd
   scripts\start_mission_control.bat
   ```

3. **Open the dashboard** in a browser:
   ```
   http://localhost:9877
   (change `localhost` to your server IP if running remotely)
   ```
   The server binds to `0.0.0.0:9877` by default, so any address on the host works.

## Building the Dashboard

The server serves the static files from `mission_control/dashboard/app/dist/`. To rebuild after frontend changes:

```bash
cd mission_control/dashboard/app
npm install
npm run build
```

`npm run build` runs TypeScript compilation (`tsc -b`) and then Vite production build.

## WebSocket Message Types

The dashboard and server communicate over `/ws` using JSON messages.

### Client → Server

| Type | Payload | Effect |
|------|---------|--------|
| `start_training` | `{ "config"?, "device"?, "total_timesteps"? }` | Begins RL training in a background thread. |
| `pause_training` | — | Pauses the running training loop. |
| `resume_training` | — | Resumes paused training. |
| `stop_training` | — | Stops training and cleans up the thread. |
| `reset_environment` | `{ "goal"? }` | Requests an environment reset with an optional goal. |
| `manual_action` | `{ "action": "jump" }` | Overrides the next agent step with the named action. |
| `set_hyperparam` | `{ "key": "learningRate", "value": 0.001 }` | Updates a live hyperparameter value. |
| `execute_command` | `{ "command": "..." }` | Sends a raw command to the bot (echo/log for now). |
| `inventory_action` | `{ "action": "...", ... }` | Forwards an inventory-related action. |

### Server → Client

| Type | Description |
|------|-------------|
| `status` | Current training state: `isTraining`, `isPaused`, `episodeCount`, `stepCount`. Sent on connect and on every status change. |
| `metrics` | RL training metrics: timesteps, episode count, mean reward, mean episode length, value loss. |
| `bot_state` | Live bot telemetry: health, hunger, position, held item, inventory, goal. |
| `terminal` | Info/success/command log lines. |
| `error` | Error or unknown-message notifications. |
| `reset` | Confirmation that an environment reset was requested. |
