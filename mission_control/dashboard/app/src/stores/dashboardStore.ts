import { create } from 'zustand';

// ─── Types ───────────────────────────────────────────────────────────────────

export interface TerminalEntry {
  id: string;
  text: string;
  type: 'command' | 'bot_response' | 'rl_metric' | 'error' | 'success' | 'server_event' | 'hyperparam';
  timestamp: number;
}

export interface InventorySlot {
  index: number;
  item: string | null;
  count: number;
  durability?: number;
  maxDurability?: number;
  enchantments?: string[];
}

export interface ArmorSlots {
  helmet: InventorySlot | null;
  chestplate: InventorySlot | null;
  leggings: InventorySlot | null;
  boots: InventorySlot | null;
}

export interface GameEvent {
  id: string;
  type: 'death' | 'discovery' | 'milestone';
  title: string;
  description: string;
  position?: { x: number; y: number; z: number };
  timestamp: number;
}

export interface DataPoint {
  x: number;
  y: number;
}

export interface ActionProb {
  action: string;
  prob: number;
}

export interface Entity {
  id: string;
  type: 'hostile' | 'passive' | 'player' | 'item' | 'poi';
  x: number;
  z: number;
  label?: string;
}

interface DashboardState {
  // ── Connection ──
  serverName: string;
  ipAddress: string;
  port: string;
  mcVersion: string;
  isConnected: boolean;
  isHandshaking: boolean;
  ws: WebSocket | null;

  // ── Server status ──
  playerCount: number;
  maxPlayers: number;
  tps: number;
  worldSeed: string;
  difficulty: string;

  // ── Bot status (top bar + POV overlay) ──
  botHealth: number;
  botHunger: number;
  botXP: number;
  botEquipped: string;
  botPosition: { x: number; y: number; z: number };
  dimension: string;
  inGameTime: string;
  inGameDay: number;
  ping: number;
  goal: string;

  // ── Terminal ──
  terminalHistory: TerminalEntry[];
  terminalInput: string;
  addTerminalEntry: (entry: Omit<TerminalEntry, 'id' | 'timestamp'>) => void;
  setTerminalInput: (input: string) => void;
  clearTerminal: () => void;

  // ── Training ──
  isTraining: boolean;
  isPaused: boolean;
  episodeCount: number;
  stepCount: number;

  // ── CRT mode ──
  crtMode: boolean;
  toggleCrtMode: () => void;

  // ── Emergency stop ──
  emergencyStopArmed: boolean;
  armEmergencyStop: () => void;
  disarmEmergencyStop: () => void;

  // ── Hyperparameters ──
  learningRate: number;
  gamma: number;
  batchSize: number;
  replayBufferSize: number;
  targetNetworkUpdateFreq: number;
  epsilonDecay: number;
  setHyperparam: (key: string, value: number) => void;

  // ── Inventory (40 slots) ──
  inventory: InventorySlot[];
  armor: ArmorSlots;
  offhand: InventorySlot | null;
  selectedHotbar: number;

  // ── Events ──
  events: GameEvent[];
  addEvent: (event: Omit<GameEvent, 'id' | 'timestamp'>) => void;

  // ── RL Metrics (for charts) ──
  rewardHistory: DataPoint[];
  lossHistory: DataPoint[];
  epsilonHistory: DataPoint[];
  currentActionProbs: ActionProb[];

  // ── Mini-map ──
  botAngle: number;
  entities: Entity[];
  exploredChunks: Set<string>;

  // ── Actions ──
  connect: (wsUrl: string) => void;
  disconnect: () => void;
  startTraining: () => void;
  pauseTraining: () => void;
  resumeTraining: () => void;
  stopTraining: () => void;
  resetEnvironment: () => void;
  executeCommand: (cmd: string) => void;
  updateSlot: (index: number, slot: InventorySlot) => void;
  moveItem: (fromIndex: number, toIndex: number) => void;
  _send: (msg: object) => void;
}

let entryIdCounter = 0;
let eventIdCounter = 0;

function makeEntryId(): string {
  return `term-${++entryIdCounter}`;
}

function makeEventId(): string {
  return `evt-${++eventIdCounter}`;
}

function now(): number {
  return Date.now();
}

function formatTime(ts: number): string {
  const d = new Date(ts);
  const h = d.getHours().toString().padStart(2, '0');
  const m = d.getMinutes().toString().padStart(2, '0');
  const s = d.getSeconds().toString().padStart(2, '0');
  return `[${h}:${m}:${s}]`;
}

function createEmptyInventory(): InventorySlot[] {
  const slots: InventorySlot[] = [];
  for (let i = 0; i < 40; i++) {
    slots.push({ index: i, item: null, count: 0 });
  }
  return slots;
}

function mapTerminalKind(kind?: string): TerminalEntry['type'] {
  switch (kind) {
    case 'command':
      return 'command';
    case 'success':
      return 'success';
    case 'error':
      return 'error';
    case 'info':
      return 'bot_response';
    default:
      return 'bot_response';
  }
}

const KNOWN_MANUAL_ACTIONS = new Set([
  'noop',
  'move_forward',
  'move_back',
  'strafe_left',
  'strafe_right',
  'sprint',
  'sneak',
  'jump',
  'turn_left_15',
  'turn_right_15',
  'look_up_15',
  'look_down_15',
  'attack',
  'use_item',
  'place_block',
  'select_slot_0',
  'select_slot_1',
  'select_slot_2',
  'select_slot_3',
  'select_slot_4',
  'select_slot_5',
  'select_slot_6',
  'select_slot_7',
  'select_slot_8',
  'jump_forward',
  'skill_gather_log',
  'skill_gather_stone',
  'skill_gather_coal',
  'skill_craft_planks',
  'skill_craft_sticks',
  'skill_craft_pickaxe',
  'skill_craft_sword',
  'skill_craft_crafting_table',
  'skill_eat_food',
  'skill_equip_best_sword',
  'skill_equip_best_pickaxe',
  'skill_build_shelter',
  'skill_flee',
  'skill_explore',
  'skill_attack_hostile',
]);

const MAX_METRIC_HISTORY = 100;

// ─── Store ───────────────────────────────────────────────────────────────────

export const useDashboardStore = create<DashboardState>((set, get) => ({
  // ── Connection ──
  serverName: 'RL-BOT Server',
  ipAddress: 'localhost',
  port: '9877',
  mcVersion: '1.20+',
  isConnected: false,
  isHandshaking: false,
  ws: null,

  // ── Server status ──
  playerCount: 0,
  maxPlayers: 0,
  tps: 20.0,
  worldSeed: '',
  difficulty: 'Normal',

  // ── Bot status ──
  botHealth: 20,
  botHunger: 20,
  botXP: 0,
  botEquipped: '',
  botPosition: { x: 0, y: 64, z: 0 },
  dimension: 'overworld',
  inGameTime: '06:00',
  inGameDay: 1,
  ping: 0,
  goal: '',

  // ── Terminal ──
  terminalHistory: [
    { id: makeEntryId(), text: 'Mission Control initialized. Awaiting bridge connection...', type: 'bot_response', timestamp: now() },
  ],
  terminalInput: '',
  addTerminalEntry: (entry) =>
    set((state) => ({
      terminalHistory: [
        ...state.terminalHistory,
        { ...entry, id: makeEntryId(), timestamp: now() },
      ],
    })),
  setTerminalInput: (input) => set({ terminalInput: input }),
  clearTerminal: () => set({ terminalHistory: [] }),

  // ── Training ──
  isTraining: false,
  isPaused: false,
  episodeCount: 0,
  stepCount: 0,

  // ── CRT mode ──
  crtMode: false,
  toggleCrtMode: () => set((state) => ({ crtMode: !state.crtMode })),

  // ── Emergency stop ──
  emergencyStopArmed: false,
  armEmergencyStop: () => {
    set({ emergencyStopArmed: true });
    setTimeout(() => {
      const current = get().emergencyStopArmed;
      if (current) {
        set({ emergencyStopArmed: false });
      }
    }, 3000);
  },
  disarmEmergencyStop: () => set({ emergencyStopArmed: false }),

  // ── Hyperparameters ──
  learningRate: 0.001,
  gamma: 0.99,
  batchSize: 64,
  replayBufferSize: 100000,
  targetNetworkUpdateFreq: 100,
  epsilonDecay: 0.995,
  setHyperparam: (key, value) => {
    set(() => {
      const updates: Partial<DashboardState> = {};
      if (key === 'learningRate') updates.learningRate = value;
      else if (key === 'gamma') updates.gamma = value;
      else if (key === 'batchSize') updates.batchSize = value;
      else if (key === 'replayBufferSize') updates.replayBufferSize = value;
      else if (key === 'targetNetworkUpdateFreq') updates.targetNetworkUpdateFreq = value;
      else if (key === 'epsilonDecay') updates.epsilonDecay = value;
      return updates;
    });
    const labelMap: Record<string, string> = {
      learningRate: 'Learning Rate',
      gamma: 'Discount Factor',
      batchSize: 'Batch Size',
      replayBufferSize: 'Replay Buffer Size',
      targetNetworkUpdateFreq: 'Target Network Update',
      epsilonDecay: 'Epsilon Decay',
    };
    const label = labelMap[key] || key;
    const formatted = value < 0.01 ? value.toExponential(3) : value.toLocaleString();
    set((state) => ({
      terminalHistory: [
        ...state.terminalHistory,
        {
          id: makeEntryId(),
          timestamp: now(),
          text: `${formatTime(now())} Hyperparameter updated: ${label} = ${formatted}`,
          type: 'hyperparam',
        },
      ],
    }));
    get()._send({ type: 'set_hyperparam', key, value });
  },

  // ── Inventory ──
  inventory: createEmptyInventory(),
  armor: {
    helmet: null,
    chestplate: null,
    leggings: null,
    boots: null,
  },
  offhand: null,
  selectedHotbar: 0,

  // ── Events ──
  events: [],
  addEvent: (event) =>
    set((state) => ({
      events: [
        { ...event, id: makeEventId(), timestamp: now() },
        ...state.events,
      ].slice(0, 20),
    })),

  // ── RL Metrics ──
  rewardHistory: [],
  lossHistory: [],
  epsilonHistory: [],
  currentActionProbs: [],

  // ── Mini-map ──
  botAngle: 0,
  entities: [],
  exploredChunks: new Set(),

  // ── Actions ──
  connect: (wsUrl: string) => {
    const state = get();
    if (state.ws) return;
    if (state.isConnected || state.isHandshaking) return;

    set({ isHandshaking: true });
    set((s) => ({
      terminalHistory: [
        ...s.terminalHistory,
        {
          id: makeEntryId(),
          timestamp: now(),
          text: `${formatTime(now())} Connecting to bridge at ${wsUrl}...`,
          type: 'server_event',
        },
      ],
    }));

    let ws: WebSocket;
    try {
      ws = new WebSocket(wsUrl);
    } catch (err) {
      set({ isHandshaking: false, ws: null });
      set((s) => ({
        terminalHistory: [
          ...s.terminalHistory,
          {
            id: makeEntryId(),
            timestamp: now(),
            text: `${formatTime(now())} Failed to create WebSocket: ${err instanceof Error ? err.message : String(err)}`,
            type: 'error',
          },
        ],
      }));
      return;
    }

    set({ ws });

    ws.onopen = () => {
      set({ isConnected: true, isHandshaking: false });
      set((s) => ({
        terminalHistory: [
          ...s.terminalHistory,
          {
            id: makeEntryId(),
            timestamp: now(),
            text: `${formatTime(now())} Connected to mission control bridge`,
            type: 'success',
          },
        ],
      }));
    };

    ws.onmessage = (event) => {
      let payload: Record<string, unknown>;
      try {
        payload = JSON.parse(event.data);
      } catch (err) {
        set((s) => ({
          terminalHistory: [
            ...s.terminalHistory,
            {
              id: makeEntryId(),
              timestamp: now(),
              text: `${formatTime(now())} Invalid message from bridge: ${err instanceof Error ? err.message : String(err)}`,
              type: 'error',
            },
          ],
        }));
        return;
      }

      const msgType = typeof payload.type === 'string' ? payload.type : '';

      switch (msgType) {
        case 'status': {
          const updates: Partial<DashboardState> = {};
          if (typeof payload.isTraining === 'boolean') updates.isTraining = payload.isTraining;
          if (typeof payload.isPaused === 'boolean') updates.isPaused = payload.isPaused;
          if (typeof payload.episodeCount === 'number') updates.episodeCount = payload.episodeCount;
          if (typeof payload.stepCount === 'number') updates.stepCount = payload.stepCount;
          if (Object.keys(updates).length > 0) set(updates);
          break;
        }

        case 'metrics': {
          const x = typeof payload.timesteps === 'number' ? payload.timesteps : get().stepCount;
          set((state) => {
            const nextReward =
              typeof payload.mean_reward === 'number'
                ? [...state.rewardHistory, { x, y: payload.mean_reward }].slice(-MAX_METRIC_HISTORY)
                : state.rewardHistory;
            const nextLoss =
              typeof payload.loss === 'number'
                ? [...state.lossHistory, { x, y: payload.loss }].slice(-MAX_METRIC_HISTORY)
                : state.lossHistory;
            const nextEpsilon =
              typeof payload.epsilon === 'number'
                ? [...state.epsilonHistory, { x, y: payload.epsilon }].slice(-MAX_METRIC_HISTORY)
                : state.epsilonHistory;
            return {
              rewardHistory: nextReward,
              lossHistory: nextLoss,
              epsilonHistory: nextEpsilon,
            };
          });
          break;
        }

        case 'bot_state': {
          const updates: Partial<DashboardState> = {};
          if (typeof payload.health === 'number') updates.botHealth = payload.health;
          if (typeof payload.hunger === 'number') updates.botHunger = payload.hunger;
          if (typeof payload.held_item === 'string') updates.botEquipped = payload.held_item;
          if (typeof payload.goal === 'string') updates.goal = payload.goal;
          if (
            payload.position &&
            typeof payload.position === 'object' &&
            payload.position !== null
          ) {
            const pos = payload.position as Record<string, unknown>;
            const x = typeof pos.x === 'number' ? pos.x : 0;
            const y = typeof pos.y === 'number' ? pos.y : 64;
            const z = typeof pos.z === 'number' ? pos.z : 0;
            updates.botPosition = { x, y, z };
            const chunkX = Math.floor(x / 16);
            const chunkZ = Math.floor(z / 16);
            const chunkKey = `${chunkX},${chunkZ}`;
            set((state) => {
              if (state.exploredChunks.has(chunkKey)) return state;
              const nextChunks = new Set(state.exploredChunks);
              nextChunks.add(chunkKey);
              return { exploredChunks: nextChunks };
            });
          }

          if (Array.isArray(payload.inventory)) {
            const nextInventory = createEmptyInventory();
            for (const raw of payload.inventory) {
              if (!raw || typeof raw !== 'object') continue;
              const item = raw as Record<string, unknown>;
              const idx = typeof item.index === 'number' ? item.index : -1;
              const name = typeof item.item === 'string' ? item.item : null;
              const count = typeof item.count === 'number' ? item.count : 0;
              if (idx >= 0 && idx < 40) {
                nextInventory[idx] = { index: idx, item: name, count };
              } else if (name !== null) {
                const emptySlot = nextInventory.find((s) => s.item === null);
                if (emptySlot) {
                  emptySlot.item = name;
                  emptySlot.count = count;
                }
              }
            }
            updates.inventory = nextInventory;
          }

          if (Object.keys(updates).length > 0) set(updates);
          break;
        }

        case 'terminal': {
          const text = typeof payload.text === 'string' ? payload.text : String(payload.text ?? '');
          const kind = typeof payload.kind === 'string' ? payload.kind : undefined;
          set((s) => ({
            terminalHistory: [
              ...s.terminalHistory,
              {
                id: makeEntryId(),
                timestamp: now(),
                text,
                type: mapTerminalKind(kind),
              },
            ],
          }));
          break;
        }

        case 'error': {
          const text = typeof payload.message === 'string' ? payload.message : String(payload.message ?? payload.text ?? 'Unknown bridge error');
          set((s) => ({
            terminalHistory: [
              ...s.terminalHistory,
              {
                id: makeEntryId(),
                timestamp: now(),
                text: `${formatTime(now())} ${text}`,
                type: 'error',
              },
            ],
          }));
          break;
        }

        case 'reset': {
          set((s) => ({
            terminalHistory: [
              ...s.terminalHistory,
              {
                id: makeEntryId(),
                timestamp: now(),
                text: `${formatTime(now())} Environment reset by bridge`,
                type: 'server_event',
              },
            ],
          }));
          break;
        }

        default: {
          // Unrecognized type: ignore silently
          break;
        }
      }
    };

    ws.onclose = () => {
      const current = get();
      if (current.ws === ws) {
        set({ ws: null, isConnected: false, isHandshaking: false });
        set((s) => ({
          terminalHistory: [
            ...s.terminalHistory,
            {
              id: makeEntryId(),
              timestamp: now(),
              text: `${formatTime(now())} Connection to bridge lost. Reconnecting in 3s...`,
              type: 'error',
            },
          ],
        }));
        setTimeout(() => {
          get().connect(wsUrl);
        }, 3000);
      }
    };

    ws.onerror = () => {
      set((s) => ({
        terminalHistory: [
          ...s.terminalHistory,
          {
            id: makeEntryId(),
            timestamp: now(),
            text: `${formatTime(now())} WebSocket error`,
            type: 'error',
          },
        ],
      }));
    };
  },

  disconnect: () => {
    const { ws } = get();
    set({ ws: null, isConnected: false, isHandshaking: false, isTraining: false, isPaused: false });
    if (ws) {
      ws.close();
    }
    set((s) => ({
      terminalHistory: [
        ...s.terminalHistory,
        {
          id: makeEntryId(),
          timestamp: now(),
          text: `${formatTime(now())} Disconnected from bridge`,
          type: 'server_event',
        },
      ],
    }));
  },

  _send: (msg: object) => {
    const ws = get().ws;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(msg));
    }
  },

  startTraining: () => {
    set({ isTraining: true, isPaused: false });
    set((s) => ({
      terminalHistory: [
        ...s.terminalHistory,
        {
          id: makeEntryId(),
          timestamp: now(),
          text: `${formatTime(now())} Training started — Episode ${s.episodeCount + 1}`,
          type: 'rl_metric',
        },
      ],
    }));
    get()._send({ type: 'start_training' });
  },

  pauseTraining: () => {
    set({ isPaused: true });
    set((s) => ({
      terminalHistory: [
        ...s.terminalHistory,
        {
          id: makeEntryId(),
          timestamp: now(),
          text: `${formatTime(now())} Training paused at step ${s.stepCount.toLocaleString()}`,
          type: 'rl_metric',
        },
      ],
    }));
    get()._send({ type: 'pause_training' });
  },

  resumeTraining: () => {
    set({ isPaused: false, isTraining: true });
    set((s) => ({
      terminalHistory: [
        ...s.terminalHistory,
        {
          id: makeEntryId(),
          timestamp: now(),
          text: `${formatTime(now())} Training resumed`,
          type: 'rl_metric',
        },
      ],
    }));
    get()._send({ type: 'resume_training' });
  },

  stopTraining: () => {
    set({ isTraining: false, isPaused: false });
    set((s) => ({
      terminalHistory: [
        ...s.terminalHistory,
        {
          id: makeEntryId(),
          timestamp: now(),
          text: `${formatTime(now())} Training stopped`,
          type: 'rl_metric',
        },
      ],
    }));
    get()._send({ type: 'stop_training' });
  },

  resetEnvironment: () => {
    set({
      stepCount: 0,
      botHealth: 20,
      botHunger: 20,
      botPosition: { x: 0, y: 64, z: 0 },
    });
    set((s) => ({
      terminalHistory: [
        ...s.terminalHistory,
        {
          id: makeEntryId(),
          timestamp: now(),
          text: `${formatTime(now())} Environment reset — Bot returned to spawn`,
          type: 'success',
        },
      ],
    }));
    get()._send({ type: 'reset_environment' });
  },

  executeCommand: (cmd: string) => {
    const trimmed = cmd.trim();
    if (!trimmed) return;

    // Echo the command
    set((s) => ({
      terminalHistory: [
        ...s.terminalHistory,
        { id: makeEntryId(), timestamp: now(), text: `> ${trimmed}`, type: 'command' },
      ],
    }));

    const send = get()._send;

    // ── Bot commands ──
    if (trimmed.startsWith('!')) {
      const withoutBang = trimmed.slice(1);
      const actionName = withoutBang.split(' ')[0];

      // Local + remote control shortcuts
      if (actionName === 'train') {
        get().startTraining();
        return;
      }
      if (actionName === 'pause') {
        get().pauseTraining();
        return;
      }
      if (actionName === 'resume') {
        get().resumeTraining();
        return;
      }
      if (actionName === 'stop') {
        get().stopTraining();
        return;
      }
      if (actionName === 'reset') {
        get().resetEnvironment();
        return;
      }

      // Skill shortcuts
      if (actionName === 'explore') {
        send({ type: 'manual_action', action: 'skill_explore' });
        return;
      }
      if (actionName === 'mine') {
        send({ type: 'manual_action', action: 'skill_gather_log' });
        return;
      }
      if (actionName === 'goto') {
        set((s) => ({
          terminalHistory: [
            ...s.terminalHistory,
            { id: makeEntryId(), timestamp: now(), text: 'Goto not yet implemented — use !explore to wander', type: 'bot_response' },
          ],
        }));
        return;
      }

      // Local-only shortcuts
      if (actionName === 'status') {
        setTimeout(() => {
          const st = get();
          set((s) => ({
            terminalHistory: [
              ...s.terminalHistory,
              {
                id: makeEntryId(),
                timestamp: now(),
                text: `Health: ${st.botHealth}/20 | Hunger: ${st.botHunger}/20 | Pos: [${st.botPosition.x}, ${st.botPosition.y}, ${st.botPosition.z}] | Dim: ${st.dimension}`,
                type: 'bot_response',
              },
            ],
          }));
        }, 200);
        return;
      }
      if (actionName === 'save') {
        setTimeout(() => {
          set((s) => ({
            terminalHistory: [
              ...s.terminalHistory,
              { id: makeEntryId(), timestamp: now(), text: 'Checkpoint saved successfully', type: 'success' },
            ],
          }));
        }, 500);
        return;
      }
      if (actionName === 'load') {
        setTimeout(() => {
          set((s) => ({
            terminalHistory: [
              ...s.terminalHistory,
              { id: makeEntryId(), timestamp: now(), text: 'Checkpoint loaded', type: 'success' },
            ],
          }));
        }, 500);
        return;
      }
      if (actionName === 'spawn') {
        setTimeout(() => {
          set((s) => ({
            terminalHistory: [
              ...s.terminalHistory,
              { id: makeEntryId(), timestamp: now(), text: 'Bot returned to spawn', type: 'success' },
            ],
          }));
        }, 300);
        return;
      }

      // Known low-level/skill manual actions
      if (KNOWN_MANUAL_ACTIONS.has(actionName)) {
        send({ type: 'manual_action', action: actionName });
        return;
      }

      // Unknown !command: fall through to generic execute_command
      send({ type: 'execute_command', command: trimmed });
      return;
    }

    // ── Minecraft / generic commands ──
    if (trimmed.startsWith('/gamemode')) {
      const mode = trimmed.split(' ')[1] || 'survival';
      setTimeout(() => {
        set((s) => ({
          terminalHistory: [
            ...s.terminalHistory,
            { id: makeEntryId(), timestamp: now(), text: `Gamemode set to ${mode}`, type: 'success' },
          ],
        }));
      }, 200);
      send({ type: 'execute_command', command: trimmed });
      return;
    }
    if (trimmed.startsWith('/tp')) {
      setTimeout(() => {
        set((s) => ({
          terminalHistory: [
            ...s.terminalHistory,
            { id: makeEntryId(), timestamp: now(), text: 'Teleported to destination', type: 'success' },
          ],
        }));
      }, 200);
      send({ type: 'execute_command', command: trimmed });
      return;
    }
    if (trimmed.startsWith('/give')) {
      setTimeout(() => {
        set((s) => ({
          terminalHistory: [
            ...s.terminalHistory,
            { id: makeEntryId(), timestamp: now(), text: 'Items added to inventory', type: 'success' },
          ],
        }));
      }, 200);
      send({ type: 'execute_command', command: trimmed });
      return;
    }
    if (trimmed === '/clear') {
      setTimeout(() => {
        set({ terminalHistory: [] });
      }, 100);
      return;
    }
    if (trimmed === '/help') {
      setTimeout(() => {
        set((s) => ({
          terminalHistory: [
            ...s.terminalHistory,
            { id: makeEntryId(), timestamp: now(), text: 'Commands: !train !pause !resume !reset !explore !mine !goto !status !save !load !spawn /gamemode /tp /give /clear /help', type: 'bot_response' },
          ],
        }));
      }, 100);
      return;
    }

    // Unknown command: send to bridge anyway
    send({ type: 'execute_command', command: trimmed });
    setTimeout(() => {
      set((s) => ({
        terminalHistory: [
          ...s.terminalHistory,
          { id: makeEntryId(), timestamp: now(), text: `Sent to bridge: "${trimmed}"`, type: 'bot_response' },
        ],
      }));
    }, 200);
  },

  updateSlot: (index, slot) =>
    set((state) => {
      const newInventory = [...state.inventory];
      newInventory[index] = slot;
      return { inventory: newInventory };
    }),

  moveItem: (fromIndex, toIndex) => {
    set((state) => {
      const newInventory = [...state.inventory];
      const fromSlot = newInventory[fromIndex];
      const toSlot = newInventory[toIndex];
      newInventory[toIndex] = fromSlot;
      newInventory[fromIndex] = toSlot;
      return { inventory: newInventory };
    });
    get()._send({ type: 'inventory_action', action: 'move', fromIndex, toIndex });
  },
}));
