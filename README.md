# Harvy RL Bot v2

> **Work in progress — still in development.**
> This project is functional, but it is not "finished" software. Expect rough edges, missing polish, and features that will change. Read this file carefully before running anything.

Harvy is a self-learning Minecraft bot. It joins a Minecraft Java server as a regular player, explores the world, dies, remembers what happened, and slowly gets better over time. It uses **Python** for the learning brain and **Node.js / Mineflayer** for the part that actually talks to Minecraft.

Think of it like raising a baby in Minecraft: at first it does random stuff, but the more it plays, the more it learns.

---

## Table of contents

1. [What you need before you start](#what-you-need-before-you-start)
2. [Download and install](#download-and-install)
3. [Configure the bot](#configure-the-bot)
4. [Start a Minecraft server](#start-a-minecraft-server)
5. [Run the bot](#run-the-bot)
6. [Train the bot](#train-the-bot)
7. [Use the web control panel](#use-the-web-control-panel)
8. [In-game commands](#in-game-commands)
9. [What the files do](#what-the-files-do)
10. [Troubleshooting](#troubleshooting)
11. [Current limitations](#current-limitations)

---

## What you need before you start

- A computer with Windows, macOS, or Linux.
- A Minecraft Java server you can connect to. The bot cannot join Minecraft Realms or Bedrock/console editions.
- Basic comfort with editing text files and running commands in a terminal / command prompt.

### Required software

| Program | Minimum version | Why you need it |
|---------|-----------------|-----------------|
| Python | 3.10 or newer | The learning brain runs in Python. |
| Node.js | 18 or newer | The Minecraft client part runs in Node.js. |
| Minecraft Java server | 1.20.1 to 1.21.1 recommended | The bot needs a server to join. |
| Git | Any recent version | For downloading this project. |

### How to check if Python and Node are installed

Open your terminal / command prompt and type:

```bash
python --version
node --version
npm --version
```

If any of those say "command not found" or "not recognized", you need to install that program first.

- **Python:** https://www.python.org/downloads/ (check "Add Python to PATH" during install on Windows)
- **Node.js:** https://nodejs.org/ (download the LTS version)

---

## Download and install

### 1. Download the project

```bash
git clone https://github.com/spectra142/harvy_rl_bot.git
cd harvy_rl_bot
```

If you do not want to use Git, click the green **Code** button on GitHub and choose **Download ZIP**, then extract it.

### 2. Install Python dependencies

This creates a virtual environment (a private Python folder) so the project does not mess with your system Python.

**Linux / macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (Command Prompt):**

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

You will know it worked if your terminal prompt now starts with `(.venv)`.

### 3. Install Node.js dependencies

```bash
cd bot
npm install
cd ..
```

---

## Configure the bot

All the user-specific settings live in one file: `configs/default.yaml`.

Open it in any text editor (Notepad, VS Code, Nano, etc.). You must replace every placeholder that looks like `YOUR_SOMETHING_HERE`.

### Settings to change

```yaml
environment:
  # The Minecraft server the bot will join.
  # Replace with the actual IP address or hostname of your server.
  # Examples:
  #   mc_host: "localhost"              # if the server is on the same computer
  #   mc_host: "192.168.1.50"           # a local home server
  #   mc_host: "myserver.example.com"   # an online server
  mc_host: "YOUR_SERVER_IP_OR_HOSTNAME_HERE"

  # The port your Minecraft server is running on.
  # The default Minecraft Java port is 25565. Only change this if you changed it on the server.
  mc_port: 25565

  # The username the bot will use when joining the server.
  # This must be a valid Minecraft username. It cannot be the same as your own username.
  mc_username: "YOUR_BOT_USERNAME_HERE"

  # The Minecraft version your server is running.
  # Ask the server owner, or look at the server console / server list.
  # Examples: "1.21.1", "1.20.6", "1.20.1"
  mc_version: "YOUR_SERVER_MINECRAFT_VERSION_HERE"

owner:
  # Set this to YOUR Minecraft username.
  # The bot will only obey chat commands from you.
  name: "YOUR_MINECRAFT_USERNAME_HERE"
```

### Example of a finished config

```yaml
environment:
  host: "localhost"
  port: 9876
  mc_host: "localhost"
  mc_port: 25565
  mc_username: "HarvyBot"
  mc_version: "1.21.1"

owner:
  name: "Steve"
  uuid: ""
```

Save the file after editing.

---

## Start a Minecraft server

If you already have a Minecraft Java server, skip this section and make sure it is online and in offline mode (or that you own a legitimate Minecraft account for the bot).

### Quick local server for testing

1. Download the official Minecraft server jar from https://www.minecraft.net/en-us/download/server
2. Put it in a new empty folder.
3. Create a file named `eula.txt` in that folder with this content:

   ```text
   eula=true
   ```

4. Run the server:

   **Linux / macOS:**

   ```bash
   java -jar server.jar nogui
   ```

   **Windows:**

   ```cmd
   java -jar server.jar nogui
   ```

5. The first time it runs it will create a `server.properties` file. Open that file and change:

   ```properties
   online-mode=false
   ```

   This lets the bot join without a paid Minecraft account. Save the file and restart the server.

6. Your server is now running on `localhost:25565`.

---

## Run the bot

The bot has two halves that must run at the same time:

1. The **Mineflayer client** (Node.js) — the actual Minecraft player.
2. The **Python trainer** — the learning brain that tells the bot what to do.

### Terminal 1 — start the Minecraft client

Make sure you are in the project root folder, then run:

```bash
cd bot
node index.js
```

You should see messages like "Bot spawned" and "tcp server listening on port 9876".

### Terminal 2 — start training

Activate the Python environment again, then run the trainer:

**Linux / macOS:**

```bash
source .venv/bin/activate
python src/train.py --config configs/default.yaml --output_dir ./checkpoints
```

**Windows (Command Prompt):**

```cmd
.venv\Scripts\activate.bat
python src\train.py --config configs\default.yaml --output_dir .\checkpoints
```

The trainer will:

- Connect to the bot over TCP port 9876.
- Load the latest checkpoint automatically if one exists.
- Start learning and saving new checkpoints.

To stop, press `Ctrl + C` in either terminal. Training progress is saved in the `checkpoints/` folder, so the bot will resume where it left off next time.

---

## Train the bot

Training means the bot tries random actions, gets rewards or penalties, and slowly learns which actions are good.

### Important training settings in `configs/default.yaml`

```yaml
training:
  # How many total steps the bot will practice before stopping.
  # One step is roughly 1/20 of a second. 5_000_000 steps is a long training run.
  total_timesteps: 5_000_000

  # Device the neural network runs on.
  # "auto" will pick a GPU if you have one, otherwise the CPU.
  device: "auto"

  # How often a checkpoint is saved, in training steps.
  checkpoint_freq: 100_000
```

You can lower `total_timesteps` to `100_000` for a quick test run.

### Checkpoints

Checkpoints are saved under:

```text
checkpoints/
  latest_model.zip          # always the most recent model
checkpoints/checkpoints/
  ppo_100000_steps.zip      # older checkpoints every checkpoint_freq steps
```

The trainer automatically loads `checkpoints/latest_model.zip` on startup if it exists.

---

## Use the web control panel

The web panel lets you watch the bot and give it commands from a browser.

### Start the panel

In a new terminal:

**Linux / macOS:**

```bash
source .venv/bin/activate
uvicorn src.web.main:app --host 0.0.0.0 --port 8080
```

**Windows (Command Prompt):**

```cmd
.venv\Scripts\activate.bat
uvicorn src.web.main:app --host 0.0.0.0 --port 8080
```

### Open the panel

Open your browser and go to:

```text
http://localhost:8080
```

The default login token is `changeme`. You can change it by setting an environment variable before starting the panel:

**Linux / macOS:**

```bash
export HARVY_WEB_TOKEN="your_secret_token_here"
uvicorn src.web.main:app --host 0.0.0.0 --port 8080
```

**Windows (PowerShell):**

```powershell
$env:HARVY_WEB_TOKEN="your_secret_token_here"
uvicorn src.web.main:app --host 0.0.0.0 --port 8080
```

### What you can do in the panel

- See the bot's current health, position, inventory, and nearby threats.
- Queue commands like `attack_nearest`, `build_wall`, `gather`, or `flee`.
- Mark players as trusted allies or hostiles.

The panel commands only work while the Python trainer is actively running.

---

## In-game commands

While you are playing Minecraft on the same server, you can type commands in chat to control the bot. **The bot only listens to the owner username you set in `configs/default.yaml`.**

| Chat command | What it does |
|--------------|--------------|
| `harvy trust <player>` | Mark a player as a friend. The bot will not attack them. |
| `harvy hostile <player>` | Mark a player as an enemy. The bot will attack on sight. |
| `harvy neutral <player>` | Reset a player to neutral. |

Examples:

```text
harvy trust Alex
harvy hostile Notch
harvy neutral Steve
```

---

## What the files do

```text
.
├── bot/                     # Minecraft client (Node.js + Mineflayer)
│   ├── index.js             # Entry point: starts the bot and TCP server
│   ├── src/
│   │   ├── bot.js           # Creates the Mineflayer bot instance
│   │   ├── actions.js       # Turns RL actions into Minecraft inputs
│   │   ├── observations.js  # Builds what the AI "sees"
│   │   ├── survival.js      # Emergency overrides (lava, suffocation, void)
│   │   └── connection.js    # TCP talk to the Python trainer
│   └── package.json         # Node.js dependencies
├── configs/
│   └── default.yaml         # YOUR settings file — edit this
├── schematics/              # Build-skill blueprints (JSON)
├── src/                     # Python RL code
│   ├── train.py             # Main training script
│   ├── eval.py              # Benchmark / evaluation script
│   ├── common/
│   │   └── config.py        # Loads configs/default.yaml
│   ├── env/
│   │   ├── minecraft_env.py # Gymnasium environment
│   │   ├── observation.py   # Decodes observations from the bot
│   │   ├── action_space.py  # Defines possible actions
│   │   └── rewards.py       # Calculates rewards
│   ├── agent/
│   │   ├── ppo_agent.py     # PPO training setup
│   │   ├── networks.py      # Neural networks
│   │   └── callbacks.py     # Checkpoint saving
│   ├── skills/
│   │   ├── combat_skill.py  # Fight enemies
│   │   ├── build_skill.py   # Place blocks from schematics
│   │   ├── gather_skill.py  # Mine logs / ore
│   │   ├── flee_skill.py    # Run away from danger
│   │   └── skill_executor.py# Runs the active skill
│   ├── memory/
│   │   ├── knowledge_base.py# Persistent SQLite memory
│   │   ├── trust_manager.py # Owner / threat system
│   │   └── experience_replay.py
│   └── web/
│       ├── main.py          # FastAPI web server
│       └── command_queue.py # Queue for panel commands
├── tests/                   # Unit tests
├── test.sh                  # Run this to check everything is okay
├── requirements.txt         # Python packages to install
└── README.md                # This file
```

---

## Troubleshooting

### "Connection refused" or "python connected" never shows

- Make sure the bot (`node index.js`) is running before the trainer.
- Make sure `configs/default.yaml` has the right `mc_host` and `mc_port`.
- Make sure nothing else is using port 9876.

### The bot does not obey my chat commands

- Check that `owner.name` in `configs/default.yaml` exactly matches your Minecraft username (case-sensitive).
- The bot ignores everyone by default, including you, if the owner name is wrong or empty.

### The bot joins but immediately gets kicked

- Check that `mc_version` matches your server's Minecraft version.
- If the server is in online mode, the bot needs a real paid Minecraft account. For testing, set `online-mode=false` in `server.properties`.

### Training is very slow

- Training runs in real-time Minecraft, so it is inherently slow.
- A GPU helps the neural network but not the Minecraft simulation itself.
- Lower `n_steps` or `total_timesteps` for faster test runs.

### `python` or `pip` not found

- Make sure Python was installed and "Add Python to PATH" was checked on Windows.
- On some systems the command is `python3` instead of `python`.

### Tests fail

Run the test script from the project root:

**Linux / macOS:**

```bash
source .venv/bin/activate
bash test.sh
```

**Windows (Git Bash / WSL):**

```bash
source .venv/bin/activate
bash test.sh
```

**Windows (Command Prompt / PowerShell):**

The `test.sh` script is written for Bash. On plain Windows, run the commands inside it manually:

```cmd
.venv\Scripts\activate.bat
python -m pytest tests\ -q
```

---

## Current limitations

- The bot is designed for a single Minecraft bot at a time. Running multiple bots on the same trainer is not supported yet.
- The web panel only works while the Python trainer is actively stepping.
- Block IDs are passed directly to the CNN as numbers. A block-embedding lookup would likely improve learning and is planned for the future.
- Complex architecture building is supported through schematics, but the bot must gather materials first.

---

## License

MIT

---

## Disclaimer

This is experimental research software. Use it on servers you own or have permission to use. Do not use it to grief, cheat, or harass other players.
