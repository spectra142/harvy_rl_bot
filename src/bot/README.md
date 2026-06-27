# Harvy RL Bot

A Mineflayer-based Minecraft Java Edition bot designed for reinforcement learning. The bot connects to a Minecraft server and communicates with a Python RL agent via a JSON-over-TCP socket protocol.

## Overview

- **Observation Frequency**: 20Hz (every physics tick)
- **Communication**: TCP socket on port `9876`
- **Protocol**: Newline-delimited JSON (NDJSON)
- **Voxel Grid**: 11x11x7 blocks centered on the bot, base64-encoded

## Architecture

```
+----------------+      TCP (port 9876)      +----------------+
|   Python RL    |  <--------------------->  |  Mineflayer    |
|    Agent       |   JSON observations +     |    Bot         |
|                |   JSON action commands    |                |
+----------------+                           +----------------+
                                                     |
                                                     | Bot protocol
                                                     v
                                              +----------------+
                                              |  Minecraft   |
                                              | Java Server  |
                                              +----------------+
```

## Installation

### Prerequisites

- Node.js >= 18.0.0
- A Minecraft Java Edition server (local or remote)

### Setup

```bash
cd src/bot
npm install
```

This installs the following dependencies:

| Package | Version | Purpose |
|---------|---------|---------|
| `mineflayer` | ^4.20.0 | Core Minecraft bot framework |
| `mineflayer-pathfinder` | ^2.4.5 | Pathfinding and navigation |
| `mineflayer-collectblock` | ^1.4.1 | Block collection automation |
| `mineflayer-pvp` | ^1.3.2 | Combat and PvP functionality |
| `minecraft-data` | ^3.65.0 | Minecraft block/item/entity data |
| `vec3` | ^0.1.10 | 3D vector math utilities |

## Configuration

All configuration is done via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MC_HOST` | `localhost` | Minecraft server hostname or IP |
| `MC_PORT` | `25565` | Minecraft server port |
| `MC_USERNAME` | `RLBOT` | Bot's in-game username |
| `TCP_PORT` | `9876` | TCP server port for Python agent |
| `DEBUG` | unset | Set to `1` for verbose debug logging |

## Running the Bot

### Basic Usage

```bash
cd src/bot
npm start
```

### With Custom Configuration

```bash
MC_HOST=mc.example.com MC_PORT=25565 MC_USERNAME=MyBot node mineflayer_bot.js
```

### With Debug Logging

```bash
DEBUG=1 npm start
```

## Communication Protocol

### Observation JSON (Bot -> Python)

Sent every tick (20Hz) as a single-line JSON object followed by a newline.

```json
{
  "self": {
    "health": 20.0,
    "food": 20.0,
    "armor": 0,
    "position": [100.5, 64.0, -50.2],
    "yaw": 90.0,
    "pitch": 0.0,
    "held_item": "wooden_pickaxe",
    "inventory_counts": {"oak_log": 12, "cobblestone": 32}
  },
  "nearby_entities": [
    {"type": "zombie", "distance": 5.2, "health": 20, "hostile": true}
  ],
  "nearby_blocks": [
    {"type": "lava", "position": [1, -2, 4]}
  ],
  "voxel_grid": "base64_encoded_11x11x7_grid",
  "environment": {
    "time_of_day": 6000,
    "can_see_sky": true,
    "in_water": false,
    "on_ground": true,
    "danger_level": 0.3
  },
  "goal": "survive_first_night",
  "reward_signal": {
    "alive_tick": 1,
    "damage_taken": 0,
    "mob_killed": 0,
    "item_mined": 0,
    "item_lost": 0,
    "food_eaten": 0,
    "death": 0
  }
}
```

### Action JSON (Python -> Bot)

Send a single-line JSON object followed by a newline:

```json
{"action": "move_forward", "value": 1.0}
```

## License

MIT
