#!/usr/bin/env node
/**
 * mineflayer_bot.js
 *
 * Mineflayer-based Minecraft bot for reinforcement learning.
 * Communicates with a Python RL agent via JSON-over-TCP socket (port 9876).
 *
 * Architecture:
 *   - Connects to a Minecraft Java Edition server using Mineflayer
 *   - Runs a TCP server speaking a strict request/response protocol:
 *     Python sends one action (or reset), the bot executes it and replies
 *     with exactly one observation on the next physics tick. Reward event
 *     counters accumulate between requests and are flushed on send, so no
 *     reward signal is ever lost regardless of Python's step rate.
 *   - Tracks reward signals, danger levels, and voxel grids for the RL agent
 *
 * Environment variables:
 *   MC_HOST      - Minecraft server host (default: localhost)
 *   MC_PORT      - Minecraft server port (default: 25565)
 *   MC_USERNAME  - Bot username (default: RLBOT)
 *   MC_VERSION   - Pin a Minecraft version (default: auto-detect)
 *   MC_RESET_COMMANDS - Set to 0 to disable /clear+/kill episode resets
 *   TCP_PORT     - TCP server port for Python communication (default: 9876)
 *   DEBUG        - Set to 1 for verbose debug logging
 */

// =============================================================================
// IMPORTS
// =============================================================================

const mineflayer = require('mineflayer');
const pathfinder = require('mineflayer-pathfinder');
const collectBlock = require('mineflayer-collectblock');
const pvp = require('mineflayer-pvp').plugin;
const mcDataLoader = require('minecraft-data');
const vec3 = require('vec3');
const net = require('net');

// =============================================================================
// CONFIGURATION
// =============================================================================

const CONFIG = {
  host: process.env.MC_HOST || 'localhost',
  port: parseInt(process.env.MC_PORT, 10) || 25565,
  username: process.env.MC_USERNAME || 'harvy',
  tcpPort: parseInt(process.env.TCP_PORT, 10) || 9876,
  debug: process.env.DEBUG === '1',
  // Auto-detect the server version by default; set MC_VERSION to pin one.
  version: process.env.MC_VERSION || false,
};

// Voxel grid dimensions (must match SPEC)
const VOXEL_XZ = 11; // X and Z range: -5 to +5
const VOXEL_Y = 7;   // Y range: -3 to +3
const VOXEL_SIZE = VOXEL_XZ * VOXEL_XZ * VOXEL_Y; // 847 bytes

// Action definitions matching SPEC
const VALID_ACTIONS = new Set([
  'noop', 'move_forward', 'move_back', 'strafe_left', 'strafe_right',
  'sprint', 'sneak', 'jump', 'turn_left_15', 'turn_right_15',
  'look_up_15', 'look_down_15', 'attack', 'use_item', 'place_block',
  'select_slot_0', 'select_slot_1', 'select_slot_2', 'select_slot_3',
  'select_slot_4', 'select_slot_5', 'select_slot_6', 'select_slot_7',
  'select_slot_8', 'jump_forward',
  // High-level skill actions
  'skill_gather_log', 'skill_gather_stone', 'skill_gather_coal',
  'skill_craft_planks', 'skill_craft_sticks', 'skill_craft_pickaxe',
  'skill_craft_sword', 'skill_craft_crafting_table',
  'skill_eat_food', 'skill_equip_best_sword', 'skill_equip_best_pickaxe',
  'skill_build_shelter', 'skill_flee', 'skill_explore', 'skill_attack_hostile'
]);

// =============================================================================
// LOGGER UTILITY
// =============================================================================

const LOG_LEVELS = { DEBUG: 0, INFO: 1, WARN: 2, ERROR: 3 };
const CURRENT_LOG_LEVEL = CONFIG.debug ? LOG_LEVELS.DEBUG : LOG_LEVELS.INFO;

function log(level, ...args) {
  if (level >= CURRENT_LOG_LEVEL) {
    const label = Object.keys(LOG_LEVELS).find(k => LOG_LEVELS[k] === level) || 'INFO';
    const timestamp = new Date().toISOString();
    console.log(`[${timestamp}] [${label}]`, ...args);
  }
}

// =============================================================================
// GLOBAL STATE
// =============================================================================

let bot = null;
let mcData = null;
let tcpServer = null;
let pythonSocket = null;

// Reward tracking state. Counters ACCUMULATE between observation requests
// and are only flushed when an observation is actually sent to Python --
// never wiped per physics tick (that silently deleted reward signals).
const rewardState = {
  lastHealth: 20,
  lastFood: 20,
  mobKills: 0,          // mobs killed since last observation
  playerKills: 0,       // players killed since last observation
  cumulativeDamage: 0,
  lastInventory: new Map(),
  itemsMined: 0,
  itemsLost: 0,
  foodEaten: 0,
  died: false,
  killBuffer: [], // Buffer entity names killed since last observation
  blocksBrokenNames: [], // Names of blocks broken since last observation
  itemsCrafted: 0,
  blocksPlaced: 0,
  shelterComplete: false,
  distanceMoved: 0,
  lastPos: null,
};

// Episode-level cumulative stats (reset only on episode reset), reported in
// every observation so Python can populate info/benchmark metrics.
const episodeStats = {
  mobsKilled: 0,
  playersKilled: 0,
  blocksMined: 0,
  blocksPlaced: 0,
  itemsCrafted: 0,
  resourcesCollected: 0,
  pickaxeCrafted: false,
  deaths: 0,
};

// Request/response protocol state: the bot sends exactly one observation
// per action/reset command from Python -- never unsolicited at 20 Hz.
let pendingObservation = false;
let resetRespondAtTick = 0;      // Delay reset responses until the respawn settles
let lastExecutedAction = null;   // Action that actually ran (after survival override)
let lastActionOverride = false;  // True when the survival layer replaced the action
let suppressNextDeath = false;   // Set when resetEpisode kills the bot on purpose

// Observation buffer
let latestObservation = null;

// Reconnection state
let reconnectTimer = null;
let isShuttingDown = false;
let tickCount = 0;

// Current episode goal
let currentGoal = 'survive_first_night';

// Memory: last 64 visited chunks {cx, cz, timestamp}
const MAX_VISITED_CHUNKS = 64;
const visitedChunks = [];

// Recent events ring buffer {type, timestamp, data}
const MAX_RECENT_EVENTS = 10;
const recentEvents = [];

// Control states for timed actions (jump, jump_forward)
let timedControls = {
  jumpReleaseTick: -1,
  forwardReleaseTick: -1,
};

// =============================================================================
// SKILL EXECUTOR - State machine for high-level skill actions
// =============================================================================

const SkillExecutor = {
  activeSkill: null,      // Name of currently running skill
  skillState: 'idle',     // Internal FSM state
  skillTarget: null,      // Target entity/block
  skillStartTick: 0,      // Tick count when skill started
  skillData: {},          // Skill-specific temporary data
  lastSkillResult: null,  // Result of last completed skill

  /**
   * Cancel the currently running skill and reset all state.
   * Also clears pathfinder goals and movement controls used by skills.
   */
  cancelSkill() {
    if (this.activeSkill) {
      log(LOG_LEVELS.DEBUG, `Skill ${this.activeSkill} cancelled (was in state ${this.skillState})`);
    }
    if (bot) {
      try {
        bot.pathfinder.setGoal(null);
      } catch (err) {
        // pathfinder may not be loaded yet
      }
      releaseAllControls();
    }
    this.activeSkill = null;
    this.skillState = 'idle';
    this.skillTarget = null;
    this.skillStartTick = 0;
    this.skillData = {};
  },

  /**
   * Start a new skill, cancelling any active one.
   */
  startSkill(skillName) {
    this.cancelSkill();
    this.activeSkill = skillName;
    this.skillState = 'starting';
    this.skillStartTick = tickCount;
    this.skillData = {};
    log(LOG_LEVELS.DEBUG, `Skill ${skillName} started at tick ${tickCount}`);
  },

  /**
   * Check if a skill has timed out (default 300 ticks = ~15 seconds).
   */
  isTimedOut(timeoutTicks = 300) {
    return tickCount - this.skillStartTick > timeoutTicks;
  },

  /**
   * Mark the current skill as completed.
   */
  completeSkill(result = 'done') {
    const skill = this.activeSkill;
    this.lastSkillResult = { skill, result, tick: tickCount };
    log(LOG_LEVELS.DEBUG, `Skill ${skill} completed: ${result}`);

    // Track crafting / building outcomes for reward signals and episode stats.
    if (skill && result === 'done') {
      if (skill.startsWith('skill_craft_')) {
        rewardState.itemsCrafted += 1;
        episodeStats.itemsCrafted += 1;
        const crafted = skill.replace('skill_craft_', '');
        recordEvent('item_crafted', 1);
        log(LOG_LEVELS.DEBUG, `Crafted via skill: ${crafted}`);
        if (crafted.includes('pickaxe')) {
          episodeStats.pickaxeCrafted = true;
        }
      } else if (skill === 'skill_build_shelter') {
        rewardState.shelterComplete = true;
      }
    }

    this.cancelSkill();
  },

  /**
   * Progress the skill FSM. Called once per physicsTick when a skill is active.
   */
  updateActiveSkill() {
    if (!this.activeSkill) return;

    // Global timeout check
    if (this.isTimedOut()) {
      log(LOG_LEVELS.WARN, `Skill ${this.activeSkill} timed out after ${tickCount - this.skillStartTick} ticks`);
      this.completeSkill('timeout');
      return;
    }

    try {
      switch (this.activeSkill) {
        case 'skill_gather_log':
          this.updateGatherLog();
          break;
        case 'skill_gather_stone':
          this.updateGatherStone();
          break;
        case 'skill_gather_coal':
          this.updateGatherCoal();
          break;
        case 'skill_craft_planks':
          this.updateCraftPlanks();
          break;
        case 'skill_craft_sticks':
          this.updateCraftSticks();
          break;
        case 'skill_craft_pickaxe':
          this.updateCraftPickaxe();
          break;
        case 'skill_craft_sword':
          this.updateCraftSword();
          break;
        case 'skill_craft_crafting_table':
          this.updateCraftCraftingTable();
          break;
        case 'skill_eat_food':
          this.updateEatFood();
          break;
        case 'skill_equip_best_sword':
          this.updateEquipBestSword();
          break;
        case 'skill_equip_best_pickaxe':
          this.updateEquipBestPickaxe();
          break;
        case 'skill_build_shelter':
          this.updateBuildShelter();
          break;
        case 'skill_flee':
          this.updateFlee();
          break;
        case 'skill_explore':
          this.updateExplore();
          break;
        case 'skill_attack_hostile':
          this.updateAttackHostile();
          break;
        default:
          log(LOG_LEVELS.WARN, `Unknown skill: ${this.activeSkill}`);
          this.completeSkill('unknown');
      }
    } catch (err) {
      log(LOG_LEVELS.ERROR, `Skill ${this.activeSkill} error in state ${this.skillState}:`, err.message);
      this.completeSkill('error');
    }
  },

  // ===========================================================================
  // GATHER LOG
  // FSM: starting -> finding_tree -> moving_to_tree -> breaking -> collecting -> done
  // ===========================================================================
  updateGatherLog() {
    switch (this.skillState) {
      case 'starting':
        this.skillData.logType = 'oak_log';
        this.skillState = 'finding_tree';
        break;

      case 'finding_tree': {
        if (!mcData) return;
        const logId = mcData.blocksByName[this.skillData.logType]?.id;
        if (!logId) {
          this.completeSkill('no_log_id');
          return;
        }
        const block = bot.findBlock({ matching: logId, maxDistance: 32 });
        if (block) {
          this.skillTarget = block;
          this.skillState = 'moving_to_tree';
          log(LOG_LEVELS.DEBUG, `Found tree at ${block.position.toString()}`);
        } else {
          // No tree found - try exploring briefly
          this.skillData.searchAttempts = (this.skillData.searchAttempts || 0) + 1;
          if (this.skillData.searchAttempts > 5) {
            this.completeSkill('no_tree_found');
          } else {
            // Walk forward while searching
            bot.setControlState('forward', true);
          }
        }
        break;
      }

      case 'moving_to_tree': {
        const block = this.skillTarget;
        if (!block) { this.skillState = 'finding_tree'; return; }
        const dist = distance(bot.entity.position, block.position);
        if (dist <= 3.5) {
          bot.pathfinder.setGoal(null);
          bot.setControlState('forward', false);
          this.skillState = 'breaking';
        } else {
          const goal = new pathfinder.goals.GoalNear(block.position.x, block.position.y, block.position.z, 3);
          bot.pathfinder.setGoal(goal);
        }
        break;
      }

      case 'breaking': {
        const block = this.skillTarget;
        if (!block || !bot.blockAt(block.position)) {
          this.skillState = 'collecting';
          return;
        }
        const dist = distance(bot.entity.position, block.position);
        if (dist > 5) { this.skillState = 'finding_tree'; return; }

        bot.lookAt(block.position.offset(0.5, 0.5, 0.5), true);
        // Dig the block
        if (bot.canDigBlock(block)) {
          bot.dig(block, true)
            .then(() => { recordBrokenBlock(block); this.skillState = 'collecting'; })
            .catch(() => { this.skillState = 'finding_tree'; });
          this.skillState = 'digging_async';
        } else {
          // Try punching it anyway
          bot.attack(block);
          this.skillState = 'collecting';
        }
        break;
      }

      case 'digging_async':
        // Waiting for dig promise to resolve; state transitions handled in .then/.catch
        break;

      case 'collecting': {
        // Walk to the dropped item and give it time to be picked up
        this.skillData.collectWait = (this.skillData.collectWait || 0) + 1;
        moveToNearestDrop(8);
        if (this.skillData.collectWait > 20) {
          bot.pathfinder.setGoal(null);
          this.completeSkill('done');
        }
        break;
      }
    }
  },

  // ===========================================================================
  // GATHER STONE
  // FSM: starting -> finding_stone -> moving -> breaking -> collecting -> done
  // ===========================================================================
  updateGatherStone() {
    switch (this.skillState) {
      case 'starting':
        this.skillState = 'finding_stone';
        break;

      case 'finding_stone': {
        if (!mcData) return;
        const stoneId = mcData.blocksByName['stone']?.id;
        if (!stoneId) { this.completeSkill('no_stone_id'); return; }
        const block = bot.findBlock({ matching: stoneId, maxDistance: 24 });
        if (block) {
          this.skillTarget = block;
          this.skillState = 'moving_to_stone';
        } else {
          this.skillData.searchAttempts = (this.skillData.searchAttempts || 0) + 1;
          if (this.skillData.searchAttempts > 5) {
            this.completeSkill('no_stone_found');
          } else {
            bot.setControlState('forward', true);
          }
        }
        break;
      }

      case 'moving_to_stone': {
        const block = this.skillTarget;
        if (!block) { this.skillState = 'finding_stone'; return; }
        const dist = distance(bot.entity.position, block.position);
        if (dist <= 3.5) {
          bot.pathfinder.setGoal(null);
          bot.setControlState('forward', false);
          this.skillState = 'breaking';
        } else {
          const goal = new pathfinder.goals.GoalNear(block.position.x, block.position.y, block.position.z, 3);
          bot.pathfinder.setGoal(goal);
        }
        break;
      }

      case 'breaking': {
        const block = this.skillTarget;
        if (!block || !bot.blockAt(block.position)) {
          this.skillState = 'collecting';
          return;
        }
        const dist = distance(bot.entity.position, block.position);
        if (dist > 5) { this.skillState = 'finding_stone'; return; }

        bot.lookAt(block.position.offset(0.5, 0.5, 0.5), true);
        if (bot.canDigBlock(block)) {
          bot.dig(block, true)
            .then(() => { recordBrokenBlock(block); this.skillState = 'collecting'; })
            .catch(() => { this.skillState = 'finding_stone'; });
          this.skillState = 'digging_async';
        } else {
          this.skillState = 'collecting';
        }
        break;
      }

      case 'digging_async':
        break;

      case 'collecting': {
        this.skillData.collectWait = (this.skillData.collectWait || 0) + 1;
        moveToNearestDrop(8);
        if (this.skillData.collectWait > 20) {
          bot.pathfinder.setGoal(null);
          this.completeSkill('done');
        }
        break;
      }
    }
  },

  // ===========================================================================
  // GATHER COAL
  // FSM: starting -> finding_coal_ore -> moving -> breaking -> collecting -> done
  // ===========================================================================
  updateGatherCoal() {
    switch (this.skillState) {
      case 'starting':
        this.skillState = 'finding_coal_ore';
        break;

      case 'finding_coal_ore': {
        if (!mcData) return;
        const coalOreId = mcData.blocksByName['coal_ore']?.id;
        if (!coalOreId) { this.completeSkill('no_coal_ore_id'); return; }
        const block = bot.findBlock({ matching: coalOreId, maxDistance: 24 });
        if (block) {
          this.skillTarget = block;
          this.skillState = 'moving_to_coal';
        } else {
          this.skillData.searchAttempts = (this.skillData.searchAttempts || 0) + 1;
          if (this.skillData.searchAttempts > 5) {
            this.completeSkill('no_coal_found');
          } else {
            bot.setControlState('forward', true);
          }
        }
        break;
      }

      case 'moving_to_coal': {
        const block = this.skillTarget;
        if (!block) { this.skillState = 'finding_coal_ore'; return; }
        const dist = distance(bot.entity.position, block.position);
        if (dist <= 3.5) {
          bot.pathfinder.setGoal(null);
          bot.setControlState('forward', false);
          this.skillState = 'breaking';
        } else {
          const goal = new pathfinder.goals.GoalNear(block.position.x, block.position.y, block.position.z, 3);
          bot.pathfinder.setGoal(goal);
        }
        break;
      }

      case 'breaking': {
        const block = this.skillTarget;
        if (!block || !bot.blockAt(block.position)) {
          this.skillState = 'collecting';
          return;
        }
        const dist = distance(bot.entity.position, block.position);
        if (dist > 5) { this.skillState = 'finding_coal_ore'; return; }

        bot.lookAt(block.position.offset(0.5, 0.5, 0.5), true);
        if (bot.canDigBlock(block)) {
          bot.dig(block, true)
            .then(() => { recordBrokenBlock(block); this.skillState = 'collecting'; })
            .catch(() => { this.skillState = 'finding_coal_ore'; });
          this.skillState = 'digging_async';
        } else {
          this.skillState = 'collecting';
        }
        break;
      }

      case 'digging_async':
        break;

      case 'collecting': {
        this.skillData.collectWait = (this.skillData.collectWait || 0) + 1;
        moveToNearestDrop(8);
        if (this.skillData.collectWait > 20) {
          bot.pathfinder.setGoal(null);
          this.completeSkill('done');
        }
        break;
      }
    }
  },

  // ===========================================================================
  // CRAFT PLANKS
  // Craft logs into planks (1 log -> 4 planks)
  // ===========================================================================
  updateCraftPlanks() {
    switch (this.skillState) {
      case 'starting':
        this.skillState = 'checking_inventory';
        break;

      case 'checking_inventory': {
        const logTypes = ['oak_log', 'spruce_log', 'birch_log', 'jungle_log', 'acacia_log', 'dark_oak_log', 'mangrove_log', 'cherry_log'];
        let hasLogs = false;
        let logName = null;
        for (const lt of logTypes) {
          const count = bot.inventory.count(mcData.itemsByName[lt]?.id);
          if (count > 0) { hasLogs = true; logName = lt; break; }
        }
        if (!hasLogs) {
          this.completeSkill('no_logs');
          return;
        }
        this.skillData.logName = logName;
        this.skillState = 'crafting';
        break;
      }

      case 'crafting': {
        const logName = this.skillData.logName;
        const logItem = mcData.itemsByName[logName];
        const plankName = logName.replace('_log', '_planks');
        const plankItem = mcData.itemsByName[plankName];
        if (!logItem || !plankItem) {
          this.completeSkill('missing_items');
          return;
        }
        const recipe = bot.recipesFor(plankItem.id, null, 1, null)[0];
        if (recipe) {
          bot.craft(recipe, 1)
            .then(() => { this.completeSkill('done'); })
            .catch((err) => { log(LOG_LEVELS.DEBUG, 'Craft planks failed:', err.message); this.completeSkill('craft_failed'); });
          this.skillState = 'crafting_async';
        } else {
          // Manual conversion if recipe not available
          this.completeSkill('no_recipe');
        }
        break;
      }

      case 'crafting_async':
        // Waiting for craft to complete
        break;
    }
  },

  // ===========================================================================
  // CRAFT STICKS
  // Craft planks into sticks (2 planks -> 4 sticks)
  // ===========================================================================
  updateCraftSticks() {
    switch (this.skillState) {
      case 'starting':
        this.skillState = 'checking_inventory';
        break;

      case 'checking_inventory': {
        const plankTypes = ['oak_planks', 'spruce_planks', 'birch_planks', 'jungle_planks', 'acacia_planks', 'dark_oak_planks', 'mangrove_planks', 'cherry_planks'];
        let hasPlanks = false;
        for (const pt of plankTypes) {
          if (bot.inventory.count(mcData.itemsByName[pt]?.id) >= 2) { hasPlanks = true; break; }
        }
        if (!hasPlanks) {
          this.completeSkill('no_planks');
          return;
        }
        this.skillState = 'crafting';
        break;
      }

      case 'crafting': {
        const stickItem = mcData.itemsByName['stick'];
        if (!stickItem) { this.completeSkill('no_stick_item'); return; }
        const recipe = bot.recipesFor(stickItem.id, null, 1, null)[0];
        if (recipe) {
          bot.craft(recipe, 1)
            .then(() => { this.completeSkill('done'); })
            .catch((err) => { log(LOG_LEVELS.DEBUG, 'Craft sticks failed:', err.message); this.completeSkill('craft_failed'); });
          this.skillState = 'crafting_async';
        } else {
          this.completeSkill('no_recipe');
        }
        break;
      }

      case 'crafting_async':
        break;
    }
  },

  // ===========================================================================
  // CRAFT PICKAXE
  // Chooses the best available material tier: iron -> stone -> wood.
  // ===========================================================================
  updateCraftPickaxe() {
    switch (this.skillState) {
      case 'starting':
        this.skillState = 'checking_materials';
        break;

      case 'checking_materials': {
        const hasSticks = bot.inventory.count(mcData.itemsByName['stick']?.id) >= 2;
        if (!hasSticks) {
          this.completeSkill('insufficient_materials');
          return;
        }
        const plankTypes = ['oak_planks', 'spruce_planks', 'birch_planks', 'jungle_planks', 'acacia_planks', 'dark_oak_planks', 'mangrove_planks', 'cherry_planks'];
        let hasPlanks = false;
        for (const pt of plankTypes) {
          if (bot.inventory.count(mcData.itemsByName[pt]?.id) >= 3) { hasPlanks = true; break; }
        }
        const hasIron = bot.inventory.count(mcData.itemsByName['iron_ingot']?.id) >= 3;
        const hasStone = bot.inventory.count(mcData.itemsByName['cobblestone']?.id) >= 3;
        if (!hasIron && !hasStone && !hasPlanks) {
          this.completeSkill('insufficient_materials');
          return;
        }
        this.skillData.tier = hasIron ? 'iron' : (hasStone ? 'stone' : 'wood');
        this.skillState = 'crafting';
        break;
      }

      case 'crafting': {
        const tier = this.skillData.tier || 'wood';
        const pickaxeName = tier === 'iron' ? 'iron_pickaxe' : (tier === 'stone' ? 'stone_pickaxe' : 'wooden_pickaxe');
        const pickaxeItem = mcData.itemsByName[pickaxeName];
        if (!pickaxeItem) { this.completeSkill('no_pickaxe_item'); return; }
        this.skillState = 'crafting_async';
        (async () => {
          try {
            // Pickaxes need a 3x3 grid: find a nearby table or place one.
            const table = await ensureCraftingTable();
            const recipes = bot.recipesFor(pickaxeItem.id, null, 1, table || false);
            const recipesNoTable = bot.recipesFor(pickaxeItem.id, null, 1, false);
            log(LOG_LEVELS.DEBUG, `Pickaxe craft: table=${table ? table.position.toString() : 'null'} ` +
              `recipes=${recipes.length} recipesNoTable=${recipesNoTable.length} ` +
              `planks=${bot.inventory.count(mcData.itemsByName['oak_planks']?.id)} ` +
              `sticks=${bot.inventory.count(mcData.itemsByName['stick']?.id)} ` +
              `tier=${this.skillData.tier}`);
            const recipe = recipes[0];
            if (!recipe) { this.completeSkill('no_recipe'); return; }
            await bot.craft(recipe, 1, table || undefined);
            this.completeSkill('done');
          } catch (err) {
            log(LOG_LEVELS.DEBUG, 'Craft pickaxe failed:', err.message);
            this.completeSkill('craft_failed');
          }
        })();
        break;
      }

      case 'crafting_async':
        break;
    }
  },

  // ===========================================================================
  // CRAFT SWORD
  // Chooses the best available material tier: iron -> stone -> wood.
  // ===========================================================================
  updateCraftSword() {
    switch (this.skillState) {
      case 'starting':
        this.skillState = 'checking_materials';
        break;

      case 'checking_materials': {
        const hasSticks = bot.inventory.count(mcData.itemsByName['stick']?.id) >= 1;
        if (!hasSticks) {
          this.completeSkill('insufficient_materials');
          return;
        }
        const plankTypes = ['oak_planks', 'spruce_planks', 'birch_planks', 'jungle_planks', 'acacia_planks', 'dark_oak_planks', 'mangrove_planks', 'cherry_planks'];
        let hasPlanks = false;
        for (const pt of plankTypes) {
          if (bot.inventory.count(mcData.itemsByName[pt]?.id) >= 2) { hasPlanks = true; break; }
        }
        const hasIron = bot.inventory.count(mcData.itemsByName['iron_ingot']?.id) >= 2;
        const hasStone = bot.inventory.count(mcData.itemsByName['cobblestone']?.id) >= 2;
        if (!hasIron && !hasStone && !hasPlanks) {
          this.completeSkill('insufficient_materials');
          return;
        }
        this.skillData.tier = hasIron ? 'iron' : (hasStone ? 'stone' : 'wood');
        this.skillState = 'crafting';
        break;
      }

      case 'crafting': {
        const tier = this.skillData.tier || 'wood';
        const swordName = tier === 'iron' ? 'iron_sword' : (tier === 'stone' ? 'stone_sword' : 'wooden_sword');
        const swordItem = mcData.itemsByName[swordName];
        if (!swordItem) { this.completeSkill('no_sword_item'); return; }
        this.skillState = 'crafting_async';
        (async () => {
          try {
            // Swords need a 3x3 grid: find a nearby table or place one.
            const table = await ensureCraftingTable();
            const recipe = bot.recipesFor(swordItem.id, null, 1, table || false)[0];
            if (!recipe) { this.completeSkill('no_recipe'); return; }
            await bot.craft(recipe, 1, table || undefined);
            this.completeSkill('done');
          } catch (err) {
            log(LOG_LEVELS.DEBUG, 'Craft sword failed:', err.message);
            this.completeSkill('craft_failed');
          }
        })();
        break;
      }

      case 'crafting_async':
        break;
    }
  },

  // ===========================================================================
  // CRAFT CRAFTING TABLE
  // Requires: 4 planks
  // ===========================================================================
  updateCraftCraftingTable() {
    switch (this.skillState) {
      case 'starting':
        this.skillState = 'checking_materials';
        break;

      case 'checking_materials': {
        // Already holding a table? Don't burn 4 planks on another one.
        const existingTable = mcData.itemsByName['crafting_table'];
        if (existingTable && bot.inventory.count(existingTable.id) >= 1) {
          this.completeSkill('done');
          return;
        }
        const plankTypes = ['oak_planks', 'spruce_planks', 'birch_planks', 'jungle_planks', 'acacia_planks', 'dark_oak_planks', 'mangrove_planks', 'cherry_planks'];
        let hasPlanks = false;
        for (const pt of plankTypes) {
          if (bot.inventory.count(mcData.itemsByName[pt]?.id) >= 4) { hasPlanks = true; break; }
        }
        if (!hasPlanks) {
          this.completeSkill('insufficient_planks');
          return;
        }
        this.skillState = 'crafting';
        break;
      }

      case 'crafting': {
        const tableItem = mcData.itemsByName['crafting_table'];
        if (!tableItem) { this.completeSkill('no_table_item'); return; }
        const recipe = bot.recipesFor(tableItem.id, null, 1, false)[0];
        if (recipe) {
          bot.craft(recipe, 1)
            .then(() => { this.completeSkill('done'); })
            .catch((err) => { log(LOG_LEVELS.DEBUG, 'Craft table failed:', err.message); this.completeSkill('craft_failed'); });
          this.skillState = 'crafting_async';
        } else {
          this.completeSkill('no_recipe');
        }
        break;
      }

      case 'crafting_async':
        break;
    }
  },

  // ===========================================================================
  // EAT FOOD
  // Find food in inventory, equip it, and consume
  // ===========================================================================
  updateEatFood() {
    switch (this.skillState) {
      case 'starting':
        this.skillState = 'finding_food';
        break;

      case 'finding_food': {
        const foodItems = ['cooked_porkchop', 'cooked_beef', 'cooked_chicken', 'cooked_mutton', 'cooked_rabbit',
          'bread', 'apple', 'baked_potato', 'cooked_salmon', 'cooked_cod',
          'porkchop', 'beef', 'chicken', 'mutton', 'rabbit',
          'potato', 'carrot', 'melon_slice', 'cookie', 'pumpkin_pie',
          'beetroot', 'dried_kelp', 'sweet_berries', 'glow_berries'];
        let foodSlot = -1;
        let foodItem = null;
        for (let i = 9; i < bot.inventory.slots.length; i++) {
          const item = bot.inventory.slots[i];
          if (item && foodItems.some(f => item.name.includes(f))) {
            foodSlot = i;
            foodItem = item;
            break;
          }
        }
        if (foodSlot === -1) {
          this.completeSkill('no_food');
          return;
        }
        this.skillData.foodSlot = foodSlot;
        this.skillData.foodName = foodItem.name;
        this.skillState = 'equipping';
        break;
      }

      case 'equipping': {
        const item = bot.inventory.slots[this.skillData.foodSlot];
        if (!item) { this.completeSkill('food_gone'); return; }
        bot.equip(item, 'hand')
          .then(() => { this.skillState = 'eating'; })
          .catch(() => { this.completeSkill('equip_failed'); });
        this.skillState = 'equipping_async';
        break;
      }

      case 'equipping_async':
        break;

      case 'eating': {
        bot.consume()
          .then(() => { this.completeSkill('done'); })
          .catch((err) => { log(LOG_LEVELS.DEBUG, 'Eat failed:', err.message); this.completeSkill('eat_failed'); });
        this.skillState = 'eating_async';
        break;
      }

      case 'eating_async':
        break;
    }
  },

  // ===========================================================================
  // EQUIP BEST SWORD
  // Find and equip the best sword from inventory
  // ===========================================================================
  updateEquipBestSword() {
    switch (this.skillState) {
      case 'starting': {
        const swordTiers = ['netherite_sword', 'diamond_sword', 'iron_sword', 'stone_sword', 'wooden_sword', 'golden_sword'];
        let bestSlot = -1;
        let bestTier = -1;
        for (let i = 9; i < bot.inventory.slots.length; i++) {
          const item = bot.inventory.slots[i];
          if (!item) continue;
          const tierIdx = swordTiers.indexOf(item.name);
          if (tierIdx !== -1 && tierIdx > bestTier) {
            bestTier = tierIdx;
            bestSlot = i;
          }
        }
        if (bestSlot === -1) {
          this.completeSkill('no_sword');
          return;
        }
        this.skillData.bestSlot = bestSlot;
        this.skillState = 'equipping';
        break;
      }

      case 'equipping': {
        const item = bot.inventory.slots[this.skillData.bestSlot];
        if (!item) { this.completeSkill('item_gone'); return; }
        bot.equip(item, 'hand')
          .then(() => { this.completeSkill('done'); })
          .catch(() => { this.completeSkill('equip_failed'); });
        this.skillState = 'equipping_async';
        break;
      }

      case 'equipping_async':
        break;
    }
  },

  // ===========================================================================
  // EQUIP BEST PICKAXE
  // Find and equip the best pickaxe from inventory
  // ===========================================================================
  updateEquipBestPickaxe() {
    switch (this.skillState) {
      case 'starting': {
        const pickaxeTiers = ['netherite_pickaxe', 'diamond_pickaxe', 'iron_pickaxe', 'stone_pickaxe', 'wooden_pickaxe', 'golden_pickaxe'];
        let bestSlot = -1;
        let bestTier = -1;
        for (let i = 9; i < bot.inventory.slots.length; i++) {
          const item = bot.inventory.slots[i];
          if (!item) continue;
          const tierIdx = pickaxeTiers.indexOf(item.name);
          if (tierIdx !== -1 && tierIdx > bestTier) {
            bestTier = tierIdx;
            bestSlot = i;
          }
        }
        if (bestSlot === -1) {
          this.completeSkill('no_pickaxe');
          return;
        }
        this.skillData.bestSlot = bestSlot;
        this.skillState = 'equipping';
        break;
      }

      case 'equipping': {
        const item = bot.inventory.slots[this.skillData.bestSlot];
        if (!item) { this.completeSkill('item_gone'); return; }
        bot.equip(item, 'hand')
          .then(() => { this.completeSkill('done'); })
          .catch(() => { this.completeSkill('equip_failed'); });
        this.skillState = 'equipping_async';
        break;
      }

      case 'equipping_async':
        break;
    }
  },

  // ===========================================================================
  // BUILD SHELTER
  // Build a 3x3 wall/roof shelter around the bot on a safe floor.
  // Falls back to digging down if no building blocks are available.
  // ===========================================================================
  updateBuildShelter() {
    switch (this.skillState) {
      case 'starting':
        this.skillState = 'checking_materials';
        break;

      case 'checking_materials': {
        const buildBlock = findBestBuildBlock();
        if (!buildBlock) {
          this.skillData.shelterMode = 'dig_down';
          this.skillState = 'executing';
          return;
        }
        this.skillData.shelterMode = 'build';
        this.skillData.buildBlockName = buildBlock.name;
        this.skillData.buildBlockSlot = buildBlock.slot;
        this.skillState = 'finding_spot';
        break;
      }

      case 'finding_spot': {
        const pos = bot.entity.position;
        const floorBlock = bot.blockAt(vec3(Math.floor(pos.x), Math.floor(pos.y) - 1, Math.floor(pos.z)));
        if (isSafeFloorBlock(floorBlock)) {
          this.skillData.targets = generateShelterTargets(pos);
          this.skillData.targetIndex = 0;
          this.skillData.placing = false;
          this.skillState = 'building';
          return;
        }
        // Find a nearby safe spot and move onto it
        const safeSpot = findSafeShelterSpot(pos);
        if (safeSpot) {
          const goal = new pathfinder.goals.GoalNear(safeSpot.x, safeSpot.y + 1, safeSpot.z, 1);
          bot.pathfinder.setGoal(goal);
          bot.setControlState('forward', true);
        } else {
          // No safe spot found, fall back to digging down
          this.skillData.shelterMode = 'dig_down';
          this.skillState = 'executing';
        }
        break;
      }

      case 'building': {
        const targets = this.skillData.targets;
        const index = this.skillData.targetIndex || 0;
        if (index >= targets.length || this.skillData.placing) {
          if (index >= targets.length) {
            bot.pathfinder.setGoal(null);
            this.completeSkill('done');
          } else if (this.skillData.placing &&
                     tickCount - (this.skillData.placingSince || 0) > 100) {
            // Watchdog: a place promise that never settles must not stall
            // the FSM -- skip the target and move on.
            this.skillData.placing = false;
            this.skillData.targetIndex = (this.skillData.targetIndex || 0) + 1;
          }
          return;
        }
        const target = targets[index];
        const blockName = this.skillData.buildBlockName;
        if (!blockName) {
          this.completeSkill('no_block');
          return;
        }
        const item = bot.inventory.items().find(i => i.name === blockName);
        if (!item) {
          this.completeSkill('out_of_blocks');
          return;
        }
        bot.equip(item, 'hand')
          .then(() => {
            const ref = bot.blockAt(vec3(target.x, target.y - 1, target.z));
            const tgt = bot.blockAt(vec3(target.x, target.y, target.z));
            // Place when the target cell is empty/replaceable and the block
            // below it is solid -- even if that block is one we just placed
            // (upper wall courses sit on our own lower course).
            const targetEmpty = !tgt || tgt.boundingBox !== 'block';
            const refSolid = ref && ref.boundingBox === 'block';
            if (targetEmpty && refSolid) {
              return bot.placeBlock(ref, vec3(0, 1, 0));
            }
            return Promise.reject(new Error('no_reference'));
          })
          .then(() => {
            this.skillData.targetIndex = (this.skillData.targetIndex || 0) + 1;
            this.skillData.placing = false;
          })
          .catch((err) => {
            log(LOG_LEVELS.DEBUG, `Shelter place failed at target ${this.skillData.targetIndex}: ${err.message}`);
            this.skillData.targetIndex = (this.skillData.targetIndex || 0) + 1;
            this.skillData.placing = false;
          });
        this.skillData.placing = true;
        this.skillData.placingSince = tickCount;
        break;
      }

      case 'executing': {
        if (this.skillData.shelterMode === 'dig_down') {
          this.skillData.digStep = (this.skillData.digStep || 0) + 1;
          if (this.skillData.digStep <= 2) {
            const pos = bot.entity.position;
            const blockBelow = bot.blockAt(vec3(Math.floor(pos.x), Math.floor(pos.y) - 1, Math.floor(pos.z)));
            if (blockBelow && bot.canDigBlock(blockBelow)) {
              bot.dig(blockBelow)
                .then(() => { /* stay in executing for next step */ })
                .catch(() => { this.completeSkill('dig_failed'); });
            }
            bot.setControlState('sneak', true);
          } else {
            bot.setControlState('sneak', false);
            bot.setControlState('jump', false);
            this.completeSkill('done');
          }
        }
        break;
      }
    }
  },

  // ===========================================================================
  // FLEE
  // Run away from the nearest hostile entity using pathfinder
  // ===========================================================================
  updateFlee() {
    switch (this.skillState) {
      case 'starting':
        this.skillState = 'finding_threat';
        break;

      case 'finding_threat': {
        const nearestHostile = findNearestHostile(16);
        if (!nearestHostile) {
          this.completeSkill('no_threat');
          return;
        }
        this.skillTarget = nearestHostile;
        this.skillState = 'fleeing';
        break;
      }

      case 'fleeing': {
        const hostile = this.skillTarget;
        if (!hostile || !hostile.position) {
          bot.pathfinder.setGoal(null);
          bot.setControlState('sprint', false);
          this.completeSkill('escaped');
          return;
        }
        // Check if we're far enough
        const dist = distance(bot.entity.position, hostile.position);
        if (dist > 20) {
          bot.pathfinder.setGoal(null);
          bot.setControlState('sprint', false);
          this.completeSkill('escaped');
          return;
        }
        // Move away from hostile
        const botPos = bot.entity.position;
        const threatPos = hostile.position;
        const dx = botPos.x - threatPos.x;
        const dz = botPos.z - threatPos.z;
        const len = Math.sqrt(dx * dx + dz * dz) || 1;
        const fleeX = botPos.x + (dx / len) * 15;
        const fleeZ = botPos.z + (dz / len) * 15;
        const goal = new pathfinder.goals.GoalXZ(Math.floor(fleeX), Math.floor(fleeZ));
        bot.pathfinder.setGoal(goal);
        bot.setControlState('sprint', true);
        break;
      }
    }
  },

  // ===========================================================================
  // EXPLORE
  // Walk in a random direction to discover new areas
  // ===========================================================================
  updateExplore() {
    switch (this.skillState) {
      case 'starting': {
        // Pick a random direction
        const yaw = bot.entity.yaw + (Math.random() * Math.PI - Math.PI / 2);
        const dist = 20 + Math.random() * 20;
        const targetX = bot.entity.position.x + Math.sin(yaw) * dist;
        const targetZ = bot.entity.position.z + Math.cos(yaw) * dist;
        this.skillData.targetX = targetX;
        this.skillData.targetZ = targetZ;
        this.skillData.exploreTicks = 0;
        this.skillState = 'moving';
        break;
      }

      case 'moving': {
        this.skillData.exploreTicks++;
        if (this.skillData.exploreTicks > 200) {
          bot.pathfinder.setGoal(null);
          bot.setControlState('forward', false);
          bot.setControlState('sprint', false);
          this.completeSkill('done');
          return;
        }
        const goal = new pathfinder.goals.GoalXZ(
          Math.floor(this.skillData.targetX),
          Math.floor(this.skillData.targetZ)
        );
        bot.pathfinder.setGoal(goal);
        bot.setControlState('forward', true);
        bot.setControlState('sprint', true);
        break;
      }
    }
  },

  // ===========================================================================
  // ATTACK HOSTILE
  // Find and attack the nearest hostile mob
  // ===========================================================================
  updateAttackHostile() {
    switch (this.skillState) {
      case 'starting':
        this.skillState = 'finding_target';
        break;

      case 'finding_target': {
        const target = findNearestHostile(16);
        if (!target) {
          this.completeSkill('no_hostile');
          return;
        }
        this.skillTarget = target;
        this.skillState = 'approaching';
        break;
      }

      case 'approaching': {
        const target = this.skillTarget;
        if (!target || !target.position) {
          this.skillState = 'finding_target';
          return;
        }
        const dist = distance(bot.entity.position, target.position);
        if (dist <= 3.5) {
          bot.pathfinder.setGoal(null);
          bot.setControlState('forward', false);
          bot.setControlState('sprint', false);
          this.skillState = 'attacking';
        } else {
          const goal = new pathfinder.goals.GoalNear(target.position.x, target.position.y, target.position.z, 3);
          bot.pathfinder.setGoal(goal);
          bot.setControlState('sprint', dist > 6);
          bot.setControlState('forward', true);
        }
        break;
      }

      case 'attacking': {
        const target = this.skillTarget;
        if (!target || !target.position) {
          this.skillState = 'finding_target';
          return;
        }
        const dist = distance(bot.entity.position, target.position);
        if (dist > 5) {
          this.skillState = 'approaching';
          return;
        }
        bot.lookAt(target.position.offset(0, target.height * 0.8, 0), true);
        if (dist <= 3.5) {
          bot.attack(target);
          // Check if target is still alive after a few hits
          this.skillData.attackCount = (this.skillData.attackCount || 0) + 1;
          if (this.skillData.attackCount > 30) {
            this.completeSkill('timeout_attacking');
          }
        }
        break;
      }
    }
  },
};

/**
 * Helper: find the nearest hostile entity within a given radius.
 */
function findNearestHostile(radius) {
  if (!bot || !bot.entities) return null;
  let nearest = null;
  let nearestDist = Infinity;
  for (const entity of Object.values(bot.entities)) {
    if (entity === bot.entity) continue;
    if (!entity.position) continue;
    const dist = distance(bot.entity.position, entity.position);
    if (dist < nearestDist && dist <= radius && isHostileMob(entity)) {
      nearest = entity;
      nearestDist = dist;
    }
  }
  return nearest;
}

/**
 * Walk toward the nearest dropped item so dug blocks actually get picked up.
 */
function moveToNearestDrop(radius) {
  if (!bot || !bot.entities) return false;
  let nearest = null;
  let nearestDist = Infinity;
  for (const entity of Object.values(bot.entities)) {
    if (!entity.position) continue;
    const name = (entity.name || '').toLowerCase();
    if (name !== 'item' && entity.displayName !== 'Item') continue;
    const dist = distance(bot.entity.position, entity.position);
    if (dist < nearestDist && dist <= radius) {
      nearest = entity;
      nearestDist = dist;
    }
  }
  if (nearest) {
    const goal = new pathfinder.goals.GoalNear(
      nearest.position.x, nearest.position.y, nearest.position.z, 1
    );
    bot.pathfinder.setGoal(goal);
    return true;
  }
  return false;
}

/**
 * Record the name of a block the bot intentionally broke (the
 * diggingCompleted event only carries the replacement air block).
 */
function recordBrokenBlock(block) {
  if (block && block.name) {
    rewardState.blocksBrokenNames.push(block.name);
  }
}

/**
 * diggingCompleted handler. NOTE: mineflayer emits the NEW block (air), not
 * the broken one -- the broken block's name is recorded by the skill FSMs,
 * which know their dig target. Here we only count the dig.
 */
function onDiggingCompleted(_newBlock) {
  recordEvent('block_broken', 1);
  rewardState.itemsMined += 1;
  episodeStats.blocksMined += 1;
  log(LOG_LEVELS.DEBUG, 'Block dig completed');
}

/**
 * Helper: attempt to place a block at a specific position.
 */
async function placeBlockAt(botInstance, referenceBlock, faceVector) {
  if (!botInstance.heldItem) return false;
  try {
    await botInstance.placeBlock(referenceBlock, faceVector);
    return true;
  } catch (err) {
    return false;
  }
}

/**
 * Shelter helpers: choose a build block, validate floor safety, and generate target placements.
 */
function findBestBuildBlock() {
  const buildBlockNames = ['dirt', 'cobblestone', 'stone', 'oak_planks', 'spruce_planks', 'birch_planks',
    'granite', 'diorite', 'andesite', 'netherrack'];
  for (let i = 9; i < bot.inventory.slots.length; i++) {
    const item = bot.inventory.slots[i];
    if (item && buildBlockNames.includes(item.name)) {
      return { name: item.name, slot: i };
    }
  }
  return null;
}

function isSafeFloorBlock(block) {
  if (!block) return false;
  if (block.name === 'air' || block.name === 'cave_air') return false;
  if (block.name === 'lava' || block.name === 'flowing_lava') return false;
  if (block.name === 'fire' || block.name === 'soul_fire') return false;
  if (block.name === 'void_air') return false;
  return block.boundingBox === 'block';
}

function findSafeShelterSpot(pos) {
  const cx = Math.floor(pos.x);
  const cy = Math.floor(pos.y);
  const cz = Math.floor(pos.z);
  for (let r = 1; r <= 3; r++) {
    for (let dx = -r; dx <= r; dx++) {
      for (let dz = -r; dz <= r; dz++) {
        if (Math.abs(dx) !== r && Math.abs(dz) !== r) continue;
        const floor = bot.blockAt(vec3(cx + dx, cy - 1, cz + dz));
        if (isSafeFloorBlock(floor)) {
          return { x: cx + dx, y: cy - 1, z: cz + dz };
        }
      }
    }
  }
  return null;
}

/**
 * Find a crafting table within reach, or place one from inventory next to
 * the bot. Returns the table block, or null if neither was possible.
 */
async function ensureCraftingTable() {
  if (!mcData) return null;
  const tableId = mcData.blocksByName['crafting_table']?.id;
  if (!tableId) return null;
  const existing = bot.findBlock({ matching: tableId, maxDistance: 6 });
  if (existing) return existing;
  const item = bot.inventory.items().find(i => i.name === 'crafting_table');
  if (!item) return null;
  const pos = bot.entity.position.floored();
  const offsets = [[1, -1, 0], [-1, -1, 0], [0, -1, 1], [0, -1, -1]];
  for (const [dx, dy, dz] of offsets) {
    const ref = bot.blockAt(pos.offset(dx, dy, dz));
    const cell = bot.blockAt(pos.offset(dx, dy + 1, dz));
    const cellEmpty = !cell || (cell.boundingBox !== 'block' && cell.name !== 'crafting_table');
    if (ref && ref.boundingBox === 'block' && cellEmpty) {
      try {
        await bot.equip(item, 'hand');
        await bot.placeBlock(ref, vec3(0, 1, 0));
        const placed = bot.blockAt(pos.offset(dx, dy + 1, dz));
        if (placed && placed.name === 'crafting_table') return placed;
      } catch (err) {
        log(LOG_LEVELS.DEBUG, `Table placement failed at ${dx},${dz}: ${err.message}`);
      }
    }
  }
  return null;
}

function generateShelterTargets(pos) {
  const cx = Math.floor(pos.x);
  const cy = Math.floor(pos.y);
  const cz = Math.floor(pos.z);
  const targets = [];
  // Ground ring (leave center empty for the bot)
  for (let dx = -1; dx <= 1; dx++) {
    for (let dz = -1; dz <= 1; dz++) {
      if (dx === 0 && dz === 0) continue;
      targets.push({ x: cx + dx, y: cy, z: cz + dz });
    }
  }
  // Upper ring
  for (let dx = -1; dx <= 1; dx++) {
    for (let dz = -1; dz <= 1; dz++) {
      if (dx === 0 && dz === 0) continue;
      targets.push({ x: cx + dx, y: cy + 1, z: cz + dz });
    }
  }
  // Roof
  for (let dx = -1; dx <= 1; dx++) {
    for (let dz = -1; dz <= 1; dz++) {
      targets.push({ x: cx + dx, y: cy + 2, z: cz + dz });
    }
  }
  return targets;
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

function degToRad(deg) {
  return (deg * Math.PI) / 180;
}

function distance(pos1, pos2) {
  const dx = pos1.x - pos2.x;
  const dy = pos1.y - pos2.y;
  const dz = pos1.z - pos2.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function clamp(val, min, max) {
  return Math.min(max, Math.max(min, val));
}

function writeVarInt(value) {
  const buf = [];
  while (true) {
    if ((value & ~0x7f) === 0) {
      buf.push(value);
      return Buffer.from(buf);
    }
    buf.push((value & 0x7f) | 0x80);
    value >>>= 7;
  }
}

function writeString(str) {
  const strBuf = Buffer.from(str, 'utf-8');
  return Buffer.concat([writeVarInt(strBuf.length), strBuf]);
}

function writeUUID(uuid) {
  return Buffer.from(uuid.replace(/-/g, ''), 'hex');
}

function getBlockTypeId(block) {
  if (!block) return 0;
  // Compact, stable block IDs shared with Python's BLOCK_TYPE_TO_ID.
  // Raw minecraft-data IDs are 4-digit in modern versions and would wrap
  // around in the Uint8Array voxel grid.
  return BLOCK_NAME_TO_ID[block.name] ?? UNKNOWN_BLOCK_ID;
}

// Must match BLOCK_TYPE_TO_ID in src/env/observation.py exactly.
const BLOCK_NAME_TO_ID = {
  air: 0, stone: 1, granite: 2, diorite: 3, andesite: 4, grass_block: 5,
  dirt: 6, coarse_dirt: 7, podzol: 8, cobblestone: 9, oak_planks: 10,
  oak_log: 11, oak_leaves: 12, glass: 13, lapis_ore: 14, sandstone: 15,
  sand: 16, gravel: 17, coal_ore: 18, iron_ore: 19, gold_ore: 20,
  diamond_ore: 21, redstone_ore: 22, emerald_ore: 23, oak_sapling: 24,
  bedrock: 25, water: 26, lava: 27, flowing_water: 28, flowing_lava: 29,
  torch: 30, crafting_table: 31, furnace: 32, chest: 33, ladder: 34,
  oak_door: 35, wheat: 36, farmland: 37, furnace_lit: 38, oak_sign: 39,
  oak_stairs: 40, cobblestone_stairs: 41,
};
// Any block not in the table gets this shared "other" ID.
const UNKNOWN_BLOCK_ID = 255;

function round2(val) {
  return Math.round(val * 100) / 100;
}

function normalizeYaw(yaw) {
  while (yaw < 0) yaw += 360;
  while (yaw >= 360) yaw -= 360;
  return yaw;
}

// =============================================================================
// MINECRAFT BOT CREATION
// =============================================================================

function createBot() {
  log(LOG_LEVELS.INFO, `Creating bot '${CONFIG.username}' connecting to ${CONFIG.host}:${CONFIG.port}`);

  bot = mineflayer.createBot({
    host: CONFIG.host,
    port: CONFIG.port,
    username: CONFIG.username,
    version: CONFIG.version,
    auth: 'offline',
    checkTimeoutInterval: 30000,
  });

  bot.loadPlugin(pathfinder.pathfinder);
  bot.loadPlugin(collectBlock.plugin);
  bot.loadPlugin(pvp);

  applyConfigurationWorkaround(bot);

  bot.once('login', () => {
    mcData = mcDataLoader(bot.version);
    log(LOG_LEVELS.INFO, `Bot logged in. Minecraft version: ${bot.version}`);
  });

  bot.once('spawn', () => {
    log(LOG_LEVELS.INFO, 'Bot spawned in world');
    resetRewardState();
    startTcpServer();
    // The viewer is opt-in: it needs the native 'canvas' module, which is
    // painful to install on some platforms. ENABLE_VIEWER=1 to turn it on.
    if (process.env.ENABLE_VIEWER === '1') {
      try {
        const viewer = require('prismarine-viewer').bot;
        viewer(bot, { port: parseInt(process.env.VIEWER_PORT, 10) || 3007, firstPerson: true });
        log(LOG_LEVELS.INFO, 'Prismarine viewer started');
      } catch (err) {
        log(LOG_LEVELS.WARN, 'Failed to start viewer:', err.message);
      }
    }
  });

  bot.on('physicsTick', onPhysicsTick);

  bot.on('death', () => {
    log(LOG_LEVELS.WARN, 'Bot died');
    episodeStats.deaths += 1;
    if (suppressNextDeath) {
      // Death caused by the episode-reset /kill command -- not a real
      // gameplay death, so don't emit a death reward signal.
      suppressNextDeath = false;
    } else {
      rewardState.died = true;
      // Everything in the inventory is lost on death (no keepInventory).
      const carried = Object.values(getInventoryCounts()).reduce((a, b) => a + b, 0);
      rewardState.itemsLost += carried;
    }
    // Always auto-respawn so the bot is ready for the next episode/step.
    if (bot._client) {
      try {
        bot._client.write('client_command', { action: 0 });
      } catch (err) {
        log(LOG_LEVELS.DEBUG, 'Auto-respawn failed:', err.message);
      }
    }
  });

  bot.on('entityDead', (entity) => {
    if (!entity) return;
    const type = entity.type || 'unknown';
    const name = entity.name || entity.username || 'unknown';
    log(LOG_LEVELS.DEBUG, `Entity died: type=${type}, name=${name}`);
    if (type === 'player') {
      rewardState.killBuffer.push({ type: 'player', name });
    } else if (type === 'mob' || type === 'hostile' || type === 'animal' || isHostileMob(entity)) {
      // mineflayer reports living entities with fine-grained types
      // ('hostile', 'animal', ...) -- anything non-player that lived counts.
      rewardState.killBuffer.push({ type: 'mob', name });
      recordEvent('mob_attacked', 1);
    }
  });

  bot.on('health', () => {
    if (suppressNextDeath) {
      // During the episode-reset /kill window, don't record the scripted
      // death's damage as a gameplay penalty.
      rewardState.lastHealth = bot.health;
      return;
    }
    if (bot.health < rewardState.lastHealth) {
      const damage = rewardState.lastHealth - bot.health;
      rewardState.cumulativeDamage += damage;
      recordEvent('damage_taken', damage);
      log(LOG_LEVELS.DEBUG, `Bot took ${damage.toFixed(1)} damage. Health: ${rewardState.lastHealth} -> ${bot.health}`);
    }
    rewardState.lastHealth = bot.health;
  });

  bot.on('eat', () => {
    rewardState.foodEaten += 1;
    recordEvent('food_eaten', 1);
    log(LOG_LEVELS.DEBUG, 'Bot ate food');
  });

  bot.on('diggingCompleted', onDiggingCompleted);
  bot.on('spawn', () => {
    // mineflayer's digging plugin removes ALL diggingCompleted listeners on
    // death -- re-register ours after every (re)spawn.
    bot.removeListener('diggingCompleted', onDiggingCompleted);
    bot.on('diggingCompleted', onDiggingCompleted);
  });

  bot.on('blockPlaced', (_oldBlock, newBlock) => {
    rewardState.blocksPlaced += 1;
    episodeStats.blocksPlaced += 1;
    recordEvent('block_placed', 1);
    log(LOG_LEVELS.DEBUG, `Block placed: ${newBlock ? newBlock.name : 'unknown'}`);
  });

  bot.on('playerCollect', (collector, collected) => {
    if (collector === bot.entity && collected) {
      recordEvent('item_collected', 1);
      episodeStats.resourcesCollected += 1;
      log(LOG_LEVELS.DEBUG, `Item collected: ${collected.name || 'unknown'}`);
    }
  });

  bot.on('error', (err) => {
    log(LOG_LEVELS.ERROR, 'Bot error:', err.message);
  });

  bot.on('kicked', (reason) => {
    log(LOG_LEVELS.WARN, 'Bot kicked:', reason);
    attemptReconnect();
  });

  bot.on('end', () => {
    log(LOG_LEVELS.WARN, 'Bot disconnected from server');
    if (!isShuttingDown) attemptReconnect();
  });
}

function applyConfigurationWorkaround(bot) {
  // Legacy workaround for servers whose configuration phase mineflayer
  // didn't handle natively. Modern mineflayer handles select_known_packs
  // and resource packs itself -- answering twice desyncs the registry
  // codec. Only enable via MC_CONFIG_WORKAROUND=1 if a server needs it.
  if (process.env.MC_CONFIG_WORKAROUND !== '1') return;
  const client = bot._client;
  if (!client) return;

  client.on('select_known_packs', (data) => {
    const packs = data?.packs ?? [];
    const parts = [writeVarInt(0x07), writeVarInt(packs.length)];
    for (const pack of packs) {
      parts.push(writeString(pack.namespace));
      parts.push(writeString(pack.id));
      parts.push(writeString(pack.version));
    }
    client.writeRaw(Buffer.concat(parts));
    log(LOG_LEVELS.DEBUG, `Sent select_known_packs response with ${packs.length} pack(s)`);
  });

  client.on('add_resource_pack', (data) => {
    const uuid = data?.uuid;
    if (!uuid) return;
    const parts = [writeVarInt(0x06), writeUUID(uuid), writeVarInt(2)];
    client.writeRaw(Buffer.concat(parts));
    log(LOG_LEVELS.DEBUG, `Sent resource_pack_receive (failed) for ${uuid}`);
  });
}

function resetRewardState() {
  rewardState.lastHealth = bot.health || 20;
  rewardState.lastFood = bot.food || 20;
  rewardState.mobKills = 0;
  rewardState.playerKills = 0;
  rewardState.cumulativeDamage = 0;
  rewardState.lastInventory.clear();
  rewardState.itemsMined = 0;
  rewardState.itemsLost = 0;
  rewardState.foodEaten = 0;
  rewardState.died = false;
  rewardState.killBuffer = [];
  rewardState.blocksBrokenNames = [];
  rewardState.itemsCrafted = 0;
  rewardState.blocksPlaced = 0;
  rewardState.shelterComplete = false;
  rewardState.distanceMoved = 0;
  rewardState.lastPos = bot && bot.entity ? bot.entity.position.clone() : null;
  tickCount = 0;
  log(LOG_LEVELS.DEBUG, 'Reward state reset');
}

function resetEpisodeStats() {
  episodeStats.mobsKilled = 0;
  episodeStats.playersKilled = 0;
  episodeStats.blocksMined = 0;
  episodeStats.blocksPlaced = 0;
  episodeStats.itemsCrafted = 0;
  episodeStats.resourcesCollected = 0;
  episodeStats.pickaxeCrafted = false;
  episodeStats.deaths = 0;
}

function resetEpisode(goal) {
  if (!bot) return;
  currentGoal = goal || 'survive_first_night';
  SkillExecutor.cancelSkill();
  releaseAllControls();
  resetRewardState();
  resetEpisodeStats();
  recentEvents.length = 0;
  visitedChunks.length = 0;
  lastExecutedAction = null;
  lastActionOverride = false;

  // Best-effort world reset via server commands (requires the bot to be
  // opped on the server; failures are harmless and logged at debug level).
  if (process.env.MC_RESET_COMMANDS !== '0') {
    try {
      suppressNextDeath = true;
      // If the bot isn't opped the /kill never lands; expire the flag so a
      // later real death is not swallowed.
      setTimeout(() => { suppressNextDeath = false; }, 5000);
      // Hold the reset response until the kill/respawn has settled, so the
      // observation reflects the fresh spawn state, not the corpse.
      resetRespondAtTick = tickCount + 20;
      bot.chat(`/clear ${CONFIG.username}`);
      bot.chat(`/kill ${CONFIG.username}`);
      bot.chat('/time set day');
      log(LOG_LEVELS.DEBUG, 'Sent world-reset server commands (/clear, /kill, /time set day)');
    } catch (err) {
      log(LOG_LEVELS.DEBUG, 'World-reset commands failed (bot not opped?):', err.message);
    }
  }

  const isDead = bot.health === 0 || (bot.isAlive === false);
  if (isDead && bot._client) {
    try {
      bot._client.write('client_command', { action: 0 });
      log(LOG_LEVELS.INFO, 'Sent respawn command');
    } catch (err) {
      log(LOG_LEVELS.ERROR, 'Failed to send respawn command:', err.message);
    }
  }
}

// =============================================================================
// TCP SERVER
// =============================================================================

function startTcpServer() {
  if (tcpServer) return;

  tcpServer = net.createServer((socket) => {
    // Exactly ONE Python client may drive the bot. A second connection
    // (e.g. an eval env) would otherwise silently steal pythonSocket and
    // starve the training env of observations.
    if (pythonSocket && !pythonSocket.destroyed) {
      log(LOG_LEVELS.WARN,
        `Rejecting extra Python client ${socket.remoteAddress}:${socket.remotePort} -- one client per bot`);
      socket.destroy();
      return;
    }
    log(LOG_LEVELS.INFO, `Python agent connected from ${socket.remoteAddress}:${socket.remotePort}`);
    pythonSocket = socket;
    socket.setEncoding('utf8');
    let buffer = '';

    socket.on('data', (data) => {
      buffer += data;
      let lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed) handleActionMessage(trimmed);
      }
    });

    socket.on('end', () => {
      log(LOG_LEVELS.INFO, 'Python agent disconnected');
      if (pythonSocket === socket) pythonSocket = null;
    });

    socket.on('error', (err) => {
      log(LOG_LEVELS.ERROR, 'TCP socket error:', err.message);
      if (pythonSocket === socket) pythonSocket = null;
    });
  });

  tcpServer.listen(CONFIG.tcpPort, '0.0.0.0', () => {
    log(LOG_LEVELS.INFO, `TCP server listening on port ${CONFIG.tcpPort}`);
  });

  tcpServer.on('error', (err) => {
    log(LOG_LEVELS.ERROR, 'TCP server error:', err.message);
  });
}

function sendObservation(obs) {
  if (!pythonSocket || pythonSocket.destroyed) return;
  try {
    const json = JSON.stringify(obs);
    pythonSocket.write(json + '\n');
  } catch (err) {
    log(LOG_LEVELS.ERROR, 'Failed to send observation:', err.message);
  }
}

// =============================================================================
// SURVIVAL PRIORITY LAYER
// =============================================================================

/**
 * Check survival priorities BEFORE executing any RL action.
 * Returns { override: false } to let the RL action execute,
 * or { override: true, action: "...", reason: "..." } to execute a survival action instead.
 *
 * Priority order (highest first):
 *   CRITICAL: Health < 4 -> flee using pathfinder, stop sprinting
 *   HIGH:     Health < 8 AND hostile within 8 blocks -> flee using pathfinder
 *   HIGH:     Food < 14 AND food item in inventory -> equip and eat food
 *   MEDIUM:   Time > 12000 (night) AND not in shelter -> build basic shelter or dig down
 *   MEDIUM:   Danger level > 0.7 -> stop moving, assess
 *   LOW:      On fire -> jump into water or stop-drop-roll
 *   LOW:      In lava -> jump repeatedly
 *   LOW:      Suffocating (head in solid block) -> break block above head
 */
function checkSurvivalPriorities() {
  if (!bot || !bot.entity || !mcData) return { override: false };

  const health = bot.health || 20;
  const food = bot.food || 20;
  const pos = bot.entity.position;

  // --- CRITICAL: Health < 4 -> flee immediately ---
  if (health < 4) {
    const hostile = findNearestHostile(16);
    if (hostile && hostile.position) {
      // Flee in opposite direction from hostile
      const dx = pos.x - hostile.position.x;
      const dz = pos.z - hostile.position.z;
      const len = Math.sqrt(dx * dx + dz * dz) || 1;
      const fleeX = pos.x + (dx / len) * 20;
      const fleeZ = pos.z + (dz / len) * 20;
      const goal = new pathfinder.goals.GoalXZ(Math.floor(fleeX), Math.floor(fleeZ));
      bot.pathfinder.setGoal(goal);
      bot.setControlState('sprint', true);
      return { override: true, action: 'skill_flee', reason: `CRITICAL: health=${health.toFixed(1)}, fleeing from ${hostile.name}` };
    }
    // No hostile nearby but critical health - just stop
    releaseAllControls();
    return { override: true, action: 'noop', reason: `CRITICAL: health=${health.toFixed(1)}, no hostile nearby, halting` };
  }

  // --- HIGH: Health < 8 AND hostile within 8 blocks -> flee ---
  if (health < 8) {
    const hostile = findNearestHostile(8);
    if (hostile && hostile.position) {
      const dx = pos.x - hostile.position.x;
      const dz = pos.z - hostile.position.z;
      const len = Math.sqrt(dx * dx + dz * dz) || 1;
      const fleeX = pos.x + (dx / len) * 20;
      const fleeZ = pos.z + (dz / len) * 20;
      const goal = new pathfinder.goals.GoalXZ(Math.floor(fleeX), Math.floor(fleeZ));
      bot.pathfinder.setGoal(goal);
      bot.setControlState('sprint', true);
      return { override: true, action: 'skill_flee', reason: `HIGH: health=${health.toFixed(1)}, hostile ${hostile.name} within 8 blocks` };
    }
  }

  // --- HIGH: Food < 14 AND food item in inventory -> eat ---
  if (food < 14) {
    const foodItems = ['cooked_porkchop', 'cooked_beef', 'cooked_chicken', 'cooked_mutton', 'cooked_rabbit',
      'bread', 'apple', 'baked_potato', 'cooked_salmon', 'cooked_cod',
      'porkchop', 'beef', 'chicken', 'mutton', 'rabbit',
      'potato', 'carrot', 'melon_slice', 'cookie', 'pumpkin_pie',
      'beetroot', 'dried_kelp', 'sweet_berries', 'glow_berries'];
    let foundFood = null;
    for (let i = 9; i < bot.inventory.slots.length; i++) {
      const item = bot.inventory.slots[i];
      if (item && foodItems.some(f => item.name.includes(f))) {
        foundFood = item;
        break;
      }
    }
    if (foundFood) {
      // Don't interrupt if already eating
      if (SkillExecutor.activeSkill === 'skill_eat_food') return { override: false };
      return { override: true, action: 'skill_eat_food', reason: `HIGH: food=${food}, eating ${foundFood.name}` };
    }
  }

  // --- MEDIUM: Night time AND not in shelter -> build shelter ---
  const timeOfDay = getTimeOfDay();
  if (timeOfDay > 12000 && timeOfDay < 23000) {
    const inShelter = isInShelter(pos);
    if (!inShelter) {
      if (SkillExecutor.activeSkill === 'skill_build_shelter') return { override: false };
      return { override: true, action: 'skill_build_shelter', reason: `MEDIUM: night time (${timeOfDay}), not in shelter` };
    }
  }

  // --- MEDIUM: Danger level > 0.7 -> stop and assess ---
  const dangerLevel = computeDangerLevel(pos);
  if (dangerLevel > 0.7) {
    releaseAllControls();
    return { override: true, action: 'noop', reason: `MEDIUM: danger_level=${dangerLevel.toFixed(2)}, assessing` };
  }

  // --- LOW: On fire -> jump into water or stop-drop-roll ---
  if (bot.entity.onFire || (bot.entity.metadata && (bot.entity.metadata[0] & 0x01))) {
    // Look for water nearby
    const waterBlock = findNearbyBlock('water', 8);
    if (waterBlock && waterBlock.position) {
      const goal = new pathfinder.goals.GoalNear(waterBlock.position.x, waterBlock.position.y, waterBlock.position.z, 1);
      bot.pathfinder.setGoal(goal);
      bot.setControlState('sprint', true);
      return { override: true, action: 'skill_flee', reason: `LOW: on fire, seeking water at ${waterBlock.position.toString()}` };
    }
    // Stop-drop-roll: stop moving
    releaseAllControls();
    return { override: true, action: 'noop', reason: 'LOW: on fire, stop-drop-roll' };
  }

  // --- LOW: In lava -> jump repeatedly ---
  if (bot.entity.isInLava || isInLava(pos)) {
    bot.setControlState('jump', true);
    // Try to move to nearest edge
    const solidBlock = findNearestSolidBlock(pos, 3);
    if (solidBlock && solidBlock.position) {
      const goal = new pathfinder.goals.GoalNear(solidBlock.position.x, solidBlock.position.y + 1, solidBlock.position.z, 1);
      bot.pathfinder.setGoal(goal);
    }
    return { override: true, action: 'jump', reason: 'LOW: in lava, jumping to escape' };
  }

  // --- LOW: Suffocating (head in solid block) -> break block above head ---
  const headBlock = bot.blockAt(bot.entity.position.offset(0, bot.entity.height, 0));
  if (headBlock && headBlock.name !== 'air' && headBlock.name !== 'cave_air' && headBlock.name !== 'water') {
    // Try to break the block above head
    if (bot.canDigBlock(headBlock)) {
      bot.dig(headBlock).catch(() => {});
    }
    return { override: true, action: 'noop', reason: `LOW: suffocating in ${headBlock.name}, breaking block above head` };
  }

  // No survival priorities triggered - let RL action execute
  return { override: false };
}

/**
 * Helper: check if the bot is in a shelter (enclosed space with roof and walls).
 */
function isInShelter(pos) {
  // Check if there are blocks above and around the bot
  const headY = Math.floor(pos.y + bot.entity.height);
  const cx = Math.floor(pos.x);
  const cz = Math.floor(pos.z);
  // Check for roof
  const roof = bot.blockAt(vec3(cx, headY + 1, cz));
  if (!roof || roof.name === 'air' || roof.name === 'cave_air') return false;
  // Check for at least 2 walls nearby
  let wallCount = 0;
  const directions = [[1, 0], [-1, 0], [0, 1], [0, -1]];
  for (const [dx, dz] of directions) {
    const wall = bot.blockAt(vec3(cx + dx, headY, cz + dz));
    if (wall && wall.name !== 'air' && wall.name !== 'cave_air') wallCount++;
  }
  return wallCount >= 2;
}

/**
 * Helper: find a nearby block of a specific type.
 */
function findNearbyBlock(blockName, maxDistance) {
  if (!mcData) return null;
  const blockId = mcData.blocksByName[blockName]?.id;
  if (!blockId) return null;
  return bot.findBlock({ matching: blockId, maxDistance });
}

/**
 * Helper: check if bot position is in lava.
 */
function isInLava(pos) {
  const block = bot.blockAt(vec3(Math.floor(pos.x), Math.floor(pos.y), Math.floor(pos.z)));
  if (block && (block.name === 'lava' || block.name === 'flowing_lava')) return true;
  const footBlock = bot.blockAt(vec3(Math.floor(pos.x), Math.floor(pos.y + 0.5), Math.floor(pos.z)));
  if (footBlock && (footBlock.name === 'lava' || footBlock.name === 'flowing_lava')) return true;
  return false;
}

/**
 * Helper: find the nearest solid block (for escaping lava).
 */
function findNearestSolidBlock(pos, radius) {
  const cx = Math.floor(pos.x);
  const cy = Math.floor(pos.y);
  const cz = Math.floor(pos.z);
  for (let r = 1; r <= radius; r++) {
    for (let dx = -r; dx <= r; dx++) {
      for (let dz = -r; dz <= r; dz++) {
        for (let dy = -1; dy <= 2; dy++) {
          const block = bot.blockAt(vec3(cx + dx, cy + dy, cz + dz));
          if (block && block.name !== 'air' && block.name !== 'cave_air' && block.name !== 'lava' && block.name !== 'flowing_lava' && block.name !== 'water') {
            return block;
          }
        }
      }
    }
  }
  return null;
}

// =============================================================================
// ACTION HANDLING
// =============================================================================

function handleActionMessage(jsonStr) {
  let action;
  try {
    action = JSON.parse(jsonStr);
  } catch (err) {
    log(LOG_LEVELS.WARN, 'Invalid JSON from Python:', jsonStr);
    return;
  }

  if (!action || typeof action.action !== 'string') {
    log(LOG_LEVELS.WARN, 'Malformed action JSON (missing "action" field):', jsonStr);
    return;
  }

  const actionName = action.action;
  const value = action.value !== undefined ? action.value : 1.0;

  if (actionName === 'reset') {
    resetEpisode(action.goal);
    // Respond with a fresh observation on the next physics tick.
    pendingObservation = true;
    return;
  }

  // Goal change WITHOUT a world reset (used by autonomous goal re-selection
  // right after reset -- avoids killing the bot twice per episode).
  if (actionName === 'set_goal') {
    if (typeof action.goal === 'string' && action.goal) {
      currentGoal = action.goal;
      log(LOG_LEVELS.DEBUG, `Goal updated: ${currentGoal}`);
    }
    pendingObservation = true;
    return;
  }

  // Out-of-band server command (Mission Control "execute_command"): run a
  // slash command as chat. Does NOT produce an observation response.
  if (actionName === 'server_command') {
    const cmd = typeof action.command === 'string' ? action.command : '';
    if (cmd) {
      try {
        bot.chat(cmd.startsWith('/') ? cmd : `/${cmd}`);
        log(LOG_LEVELS.INFO, `Executed server command: /${cmd.replace(/^\//, '')}`);
      } catch (err) {
        log(LOG_LEVELS.WARN, `Server command failed: ${err.message}`);
      }
    }
    return;
  }

  if (!VALID_ACTIONS.has(actionName)) {
    log(LOG_LEVELS.WARN, `Unknown action received: ${actionName}`);
    return;
  }

  log(LOG_LEVELS.DEBUG, `Executing action: ${actionName} (value=${value})`);
  executeAction(actionName, value);
  // Strict request/response: exactly one observation per action, sent on
  // the next physics tick so the action has had time to take effect.
  pendingObservation = true;
}

function executeAction(action, value) {
  if (!bot || !bot.entity) return;

  // --- SURVIVAL PRIORITY LAYER ---
  // Check survival priorities BEFORE executing any RL action.
  // If a survival condition is critical, override the RL action. The
  // actually-executed action is reported back in the next observation so
  // Python can log the transition honestly.
  const survival = checkSurvivalPriorities();
  if (survival.override) {
    log(LOG_LEVELS.DEBUG, `Survival override: ${survival.reason}`);
    lastExecutedAction = survival.action;
    lastActionOverride = survival.action !== action;

    // Don't cancel/restart the very skill the override wants -- restarting
    // every step would pin the FSM in its first state forever.
    if (SkillExecutor.activeSkill && survival.action !== SkillExecutor.activeSkill) {
      SkillExecutor.cancelSkill();
    }
    const startOnce = (skill) => {
      if (SkillExecutor.activeSkill !== skill) SkillExecutor.startSkill(skill);
    };
    if (survival.action === 'noop') {
      releaseAllControls();
      return;
    } else if (survival.action === 'skill_flee') {
      startOnce('skill_flee');
      return;
    } else if (survival.action === 'skill_eat_food') {
      startOnce('skill_eat_food');
      return;
    } else if (survival.action === 'skill_build_shelter') {
      startOnce('skill_build_shelter');
      return;
    } else if (survival.action === 'jump') {
      bot.setControlState('jump', true);
      return;
    } else if (survival.action === 'attack') {
      const target = findNearestAttackableEntity();
      if (target) bot.attack(target);
      return;
    }
    return;
  }
  // --- END SURVIVAL PRIORITY LAYER ---

  lastExecutedAction = action;
  lastActionOverride = false;

  const boolValue = value > 0.5;

  // --- SKILL ACTIONS ---
  // If the action is a skill action, start the skill executor and return.
  // The skill FSM will run in updateActiveSkill() each physicsTick.
  if (action.startsWith('skill_')) {
    if (!VALID_ACTIONS.has(action)) {
      log(LOG_LEVELS.WARN, `Unknown skill action: ${action}`);
      return;
    }
    // Re-selecting the already-running skill must not restart its FSM.
    if (SkillExecutor.activeSkill !== action) {
      SkillExecutor.startSkill(action);
    }
    return;
  }

  // Cancel active skill if a non-skill action is received (except noop)
  if (action !== 'noop' && SkillExecutor.activeSkill) {
    SkillExecutor.cancelSkill();
  }

  switch (action) {
    case 'noop':
      releaseAllControls();
      break;
    case 'move_forward':
      bot.setControlState('forward', boolValue);
      break;
    case 'move_back':
      bot.setControlState('back', boolValue);
      break;
    case 'strafe_left':
      bot.setControlState('left', boolValue);
      break;
    case 'strafe_right':
      bot.setControlState('right', boolValue);
      break;
    case 'sprint':
      bot.setControlState('sprint', boolValue);
      break;
    case 'sneak':
      bot.setControlState('sneak', boolValue);
      break;
    case 'jump':
      bot.setControlState('jump', true);
      timedControls.jumpReleaseTick = tickCount + 1;
      break;
    case 'jump_forward':
      bot.setControlState('jump', true);
      bot.setControlState('forward', true);
      timedControls.jumpReleaseTick = tickCount + 1;
      timedControls.forwardReleaseTick = tickCount + 1;
      break;
    case 'turn_left_15':
      bot.entity.yaw -= degToRad(15);
      break;
    case 'turn_right_15':
      bot.entity.yaw += degToRad(15);
      break;
    case 'look_up_15':
      bot.entity.pitch += degToRad(15);
      bot.entity.pitch = clamp(bot.entity.pitch, -Math.PI / 2, Math.PI / 2);
      break;
    case 'look_down_15':
      bot.entity.pitch -= degToRad(15);
      bot.entity.pitch = clamp(bot.entity.pitch, -Math.PI / 2, Math.PI / 2);
      break;
    case 'attack': {
      const target = findNearestAttackableEntity();
      if (target) {
        bot.attack(target);
        log(LOG_LEVELS.DEBUG, `Attacked entity: ${target.name || target.type}`);
      }
      break;
    }
    case 'use_item':
      bot.activateItem();
      log(LOG_LEVELS.DEBUG, 'Used held item');
      break;
    case 'place_block':
      placeBlockAction();
      break;
    case 'select_slot_0':
    case 'select_slot_1':
    case 'select_slot_2':
    case 'select_slot_3':
    case 'select_slot_4':
    case 'select_slot_5':
    case 'select_slot_6':
    case 'select_slot_7':
    case 'select_slot_8': {
      const slotIndex = parseInt(action.split('_')[2], 10);
      selectHotbarSlot(slotIndex);
      break;
    }
    default:
      log(LOG_LEVELS.WARN, `Unhandled action: ${action}`);
  }
}

function releaseAllControls() {
  if (!bot) return;
  bot.setControlState('forward', false);
  bot.setControlState('back', false);
  bot.setControlState('left', false);
  bot.setControlState('right', false);
  bot.setControlState('sprint', false);
  bot.setControlState('sneak', false);
  bot.setControlState('jump', false);
}

function findNearestAttackableEntity() {
  if (!bot || !bot.entities) return null;
  const entities = Object.values(bot.entities);
  let nearest = null;
  let nearestDist = Infinity;
  for (const entity of entities) {
    if (entity === bot.entity) continue;
    if (!entity.position) continue;
    const dist = distance(bot.entity.position, entity.position);
    if (dist < nearestDist && dist <= 4.5) {
      nearest = entity;
      nearestDist = dist;
    }
  }
  return nearest;
}

function placeBlockAction() {
  if (!bot || !bot.heldItem) {
    log(LOG_LEVELS.DEBUG, 'Place block: no item held');
    return;
  }
  const mc = mcData || mcDataLoader(CONFIG.version);
  const heldItem = bot.heldItem;
  const blockType = mc.blocksByName[heldItem.name];
  if (!blockType) {
    log(LOG_LEVELS.DEBUG, `Place block: held item '${heldItem.name}' is not a placeable block`);
    return;
  }
  const blockAtCursor = bot.blockAtCursor(4);
  if (!blockAtCursor) {
    log(LOG_LEVELS.DEBUG, 'Place block: no block in view range');
    return;
  }
  const botPos = bot.entity.position;
  const refPos = blockAtCursor.position;
  const dx = botPos.x - (refPos.x + 0.5);
  const dy = botPos.y - (refPos.y + 0.5);
  const dz = botPos.z - (refPos.z + 0.5);
  let faceVector;
  if (Math.abs(dx) >= Math.abs(dy) && Math.abs(dx) >= Math.abs(dz)) {
    faceVector = vec3(dx > 0 ? 1 : -1, 0, 0);
  } else if (Math.abs(dy) >= Math.abs(dx) && Math.abs(dy) >= Math.abs(dz)) {
    faceVector = vec3(0, dy > 0 ? 1 : -1, 0);
  } else {
    faceVector = vec3(0, 0, dz > 0 ? 1 : -1);
  }
  try {
    bot.placeBlock(blockAtCursor, faceVector);
    log(LOG_LEVELS.DEBUG, `Placed block ${heldItem.name} at ${refPos.plus(faceVector).toString()}`);
  } catch (err) {
    log(LOG_LEVELS.DEBUG, `Place block failed: ${err.message}`);
  }
}

function selectHotbarSlot(slotIndex) {
  if (!bot || slotIndex < 0 || slotIndex > 8) return;
  try {
    bot.setQuickBarSlot(slotIndex);
    log(LOG_LEVELS.DEBUG, `Selected hotbar slot ${slotIndex}`);
  } catch (err) {
    log(LOG_LEVELS.DEBUG, `Failed to select slot ${slotIndex}: ${err.message}`);
  }
}

// =============================================================================
// PHYSICS TICK - OBSERVATION COLLECTION (20Hz)
// =============================================================================

function onPhysicsTick() {
  if (!bot || !bot.entity || !mcData) return;
  tickCount++;

  if (tickCount >= timedControls.jumpReleaseTick && timedControls.jumpReleaseTick > 0) {
    bot.setControlState('jump', false);
    timedControls.jumpReleaseTick = -1;
  }
  if (tickCount >= timedControls.forwardReleaseTick && timedControls.forwardReleaseTick > 0) {
    bot.setControlState('forward', false);
    timedControls.forwardReleaseTick = -1;
  }

  // Accumulate horizontal distance travelled (for exploration rewards).
  if (rewardState.lastPos) {
    const dx = bot.entity.position.x - rewardState.lastPos.x;
    const dz = bot.entity.position.z - rewardState.lastPos.z;
    const step = Math.sqrt(dx * dx + dz * dz);
    // Ignore teleports/spawn snaps (> 20 blocks in one tick).
    if (step <= 20) rewardState.distanceMoved += step;
  }
  rewardState.lastPos = bot.entity.position.clone();

  // Update active skill FSM if a skill is running
  if (SkillExecutor.activeSkill) {
    SkillExecutor.updateActiveSkill();
  }

  // Strict request/response protocol: only send an observation when Python
  // asked for one (via an action or reset). Reward counters accumulate
  // between requests and are flushed here -- nothing is wiped per tick.
  if (pendingObservation && tickCount >= resetRespondAtTick && pythonSocket && !pythonSocket.destroyed) {
    pendingObservation = false;
    const observation = buildObservation();
    latestObservation = observation;
    sendObservation(observation);
    flushRewardSignals();
  }
}

/**
 * Clear the per-step reward accumulators. Called ONLY right after an
 * observation has actually been sent to Python, so no signal is ever lost.
 */
function flushRewardSignals() {
  rewardState.cumulativeDamage = 0;
  rewardState.itemsMined = 0;
  rewardState.itemsLost = 0;
  rewardState.foodEaten = 0;
  rewardState.killBuffer = [];
  rewardState.blocksBrokenNames = [];
  rewardState.itemsCrafted = 0;
  rewardState.blocksPlaced = 0;
  rewardState.shelterComplete = false;
  rewardState.distanceMoved = 0;
  rewardState.mobKills = 0;
  rewardState.playerKills = 0;
  if (rewardState.died) rewardState.died = false;
}

// =============================================================================
// OBSERVATION BUILDING
// =============================================================================

function buildObservation() {
  const pos = bot.entity.position;
  const yawDeg = ((bot.entity.yaw * 180) / Math.PI) % 360;
  const pitchDeg = ((bot.entity.pitch * 180) / Math.PI);

  updateVisitedChunks(pos);

  return {
    self: buildSelfObservation(pos, yawDeg, pitchDeg),
    nearby_entities: buildNearbyEntities(),
    nearby_blocks: buildNearbyBlocks(pos),
    nearby_pois: buildNearbyPOIs(pos),
    voxel_grid: buildVoxelGrid(pos),
    voxel_hardness: buildVoxelHardness(pos),
    voxel_tool_required: buildVoxelToolRequired(pos),
    visited_chunks: visitedChunks.slice(),
    recent_events: recentEvents.slice(),
    environment: buildEnvironment(pos),
    goal: currentGoal,
    reward_signal: buildRewardSignal(),
    episode_stats: { ...episodeStats },
    executed_action: lastExecutedAction,
    action_overridden: lastActionOverride,
    skill_status: {
      active_skill: SkillExecutor.activeSkill,
      skill_state: SkillExecutor.skillState,
      skill_progress: SkillExecutor.activeSkill
        ? clamp((tickCount - SkillExecutor.skillStartTick) / 300, 0, 1)
        : 0,
    },
  };
}

function updateVisitedChunks(pos) {
  const cx = Math.floor(pos.x) >> 4;
  const cz = Math.floor(pos.z) >> 4;
  // Update existing entry or add new one
  const existing = visitedChunks.find(c => c.cx === cx && c.cz === cz);
  const now = Date.now();
  if (existing) {
    existing.timestamp = now;
  } else {
    visitedChunks.push({ cx, cz, timestamp: now });
    if (visitedChunks.length > MAX_VISITED_CHUNKS) {
      visitedChunks.shift();
    }
  }
}

function buildNearbyPOIs(botPos) {
  const pois = [];
  if (!bot || !bot.entities || !mcData) return pois;

  // Block-based POIs
  const blockPoiTypes = [
    { type: 'tree', names: ['oak_log', 'spruce_log', 'birch_log', 'jungle_log', 'acacia_log', 'dark_oak_log', 'mangrove_log', 'cherry_log'] },
    { type: 'ore_coal', names: ['coal_ore', 'deepslate_coal_ore'] },
    { type: 'ore_iron', names: ['iron_ore', 'deepslate_iron_ore'] },
    { type: 'ore_diamond', names: ['diamond_ore', 'deepslate_diamond_ore'] },
    { type: 'chest', names: ['chest', 'trapped_chest'] },
    { type: 'crafting_table', names: ['crafting_table'] },
    { type: 'furnace', names: ['furnace', 'blast_furnace', 'smoker'] },
    { type: 'water_source', names: ['water', 'water_source'] },
  ];

  for (const poiType of blockPoiTypes) {
    const ids = poiType.names
      .map(name => mcData.blocksByName[name]?.id)
      .filter(id => id !== undefined);
    if (ids.length === 0) continue;
    const block = bot.findBlock({ matching: ids, maxDistance: 32 });
    if (block && block.position) {
      const dx = block.position.x - botPos.x;
      const dy = block.position.y - botPos.y;
      const dz = block.position.z - botPos.z;
      pois.push({
        type: poiType.type,
        distance: round2(Math.sqrt(dx * dx + dy * dy + dz * dz)),
        position: [round2(dx), round2(dy), round2(dz)],
      });
    }
  }

  // Entity-based POIs
  for (const entity of Object.values(bot.entities)) {
    if (entity === bot.entity) continue;
    if (!entity.position) continue;
    const dist = distance(botPos, entity.position);
    if (dist > 32) continue;
    const name = (entity.name || entity.username || '').toLowerCase();
    if (isHostileMob(entity)) {
      pois.push({
        type: 'mob_hostile',
        distance: round2(dist),
        position: [round2(entity.position.x - botPos.x), round2(entity.position.y - botPos.y), round2(entity.position.z - botPos.z)],
      });
    } else if (['cow', 'pig', 'sheep', 'chicken', 'rabbit', 'horse', 'donkey', 'mule', 'wolf', 'cat', 'goat', 'axolotl'].some(p => name.includes(p))) {
      pois.push({
        type: 'mob_passive',
        distance: round2(dist),
        position: [round2(entity.position.x - botPos.x), round2(entity.position.y - botPos.y), round2(entity.position.z - botPos.z)],
      });
    }
  }

  pois.sort((a, b) => a.distance - b.distance);
  return pois.slice(0, 16);
}

function buildVoxelHardness(botPos) {
  const centerX = Math.floor(botPos.x);
  const centerY = Math.floor(botPos.y);
  const centerZ = Math.floor(botPos.z);
  const arr = new Float32Array(VOXEL_SIZE);
  let idx = 0;
  for (let dy = -3; dy <= 3; dy++) {
    for (let dz = -5; dz <= 5; dz++) {
      for (let dx = -5; dx <= 5; dx++) {
        const block = bot.blockAt(vec3(centerX + dx, centerY + dy, centerZ + dz));
        arr[idx] = (block && mcData.blocksByName[block.name]?.hardness) || 0;
        idx++;
      }
    }
  }
  return Buffer.from(arr.buffer).toString('base64');
}

function buildVoxelToolRequired(botPos) {
  const centerX = Math.floor(botPos.x);
  const centerY = Math.floor(botPos.y);
  const centerZ = Math.floor(botPos.z);
  const arr = new Uint8Array(VOXEL_SIZE);
  let idx = 0;
  for (let dy = -3; dy <= 3; dy++) {
    for (let dz = -5; dz <= 5; dz++) {
      for (let dx = -5; dx <= 5; dx++) {
        const block = bot.blockAt(vec3(centerX + dx, centerY + dy, centerZ + dz));
        arr[idx] = getToolRequiredId(block);
        idx++;
      }
    }
  }
  return Buffer.from(arr.buffer).toString('base64');
}

function getToolRequiredId(block) {
  if (!block || !mcData) return 0;
  const blockData = mcData.blocksByName[block.name];
  if (!blockData || !blockData.harvestTools) return 0;
  const tools = Object.keys(blockData.harvestTools).map(id => {
    const item = mcData.items[id];
    return item ? item.name : '';
  });
  const hasPickaxe = tools.some(n => n.includes('pickaxe'));
  const hasAxe = tools.some(n => n.includes('axe'));
  const hasShovel = tools.some(n => n.includes('shovel'));
  const hasSword = tools.some(n => n.includes('sword'));
  const hasHoe = tools.some(n => n.includes('hoe'));
  if (hasPickaxe && !hasAxe && !hasShovel && !hasSword && !hasHoe) return 1;
  if (hasAxe && !hasPickaxe && !hasShovel && !hasSword && !hasHoe) return 2;
  if (hasShovel && !hasPickaxe && !hasAxe && !hasSword && !hasHoe) return 3;
  if (hasSword && !hasPickaxe && !hasAxe && !hasShovel && !hasHoe) return 4;
  if (hasHoe && !hasPickaxe && !hasAxe && !hasShovel && !hasSword) return 5;
  return 0;
}

function recordEvent(type, data = 0) {
  recentEvents.push({ type, timestamp: Date.now(), data });
  if (recentEvents.length > MAX_RECENT_EVENTS) {
    recentEvents.shift();
  }
}

function buildSelfObservation(pos, yawDeg, pitchDeg) {
  const heldItem = bot.heldItem;
  const inventoryCounts = getInventoryCounts();
  return {
    health: bot.health || 0,
    food: bot.food || 0,
    armor: getArmorValue(),
    position: [round2(pos.x), round2(pos.y), round2(pos.z)],
    yaw: round2(normalizeYaw(yawDeg)),
    pitch: round2(clamp(pitchDeg, -90, 90)),
    held_item: heldItem ? heldItem.name : null,
    inventory_counts: inventoryCounts,
  };
}

// Armor points per piece (Java edition values).
const ARMOR_POINTS = {
  leather_helmet: 1, leather_chestplate: 3, leather_leggings: 2, leather_boots: 1,
  golden_helmet: 2, golden_chestplate: 5, golden_leggings: 3, golden_boots: 1,
  chainmail_helmet: 2, chainmail_chestplate: 5, chainmail_leggings: 4, chainmail_boots: 1,
  iron_helmet: 2, iron_chestplate: 6, iron_leggings: 5, iron_boots: 2,
  diamond_helmet: 3, diamond_chestplate: 8, diamond_leggings: 6, diamond_boots: 3,
  netherite_helmet: 3, netherite_chestplate: 8, netherite_leggings: 6, netherite_boots: 3,
  turtle_helmet: 2, elytra: 0,
};

function getArmorValue() {
  if (!bot.inventory) return 0;
  const armorSlots = [5, 6, 7, 8];
  let total = 0;
  for (const slotIdx of armorSlots) {
    const item = bot.inventory.slots[slotIdx];
    if (item) total += ARMOR_POINTS[item.name] || 0;
  }
  return total;
}

function getInventoryCounts() {
  const counts = {};
  if (!bot.inventory || !bot.inventory.slots) return counts;
  for (let i = 9; i < bot.inventory.slots.length; i++) {
    const item = bot.inventory.slots[i];
    if (item) {
      counts[item.name] = (counts[item.name] || 0) + item.count;
    }
  }
  return counts;
}

// Typical max health per mob, used only when the server hasn't sent an
// entity's actual health yet (entity.health is undefined until damaged).
const DEFAULT_ENTITY_HEALTH = {
  zombie: 20, skeleton: 20, creeper: 20, spider: 16, enderman: 40,
  witch: 26, drowned: 20, husk: 20, stray: 20, phantom: 20, slime: 16,
  cave_spider: 12, blaze: 20, cow: 10, pig: 10, sheep: 8, chicken: 4,
  rabbit: 3, horse: 30, villager: 20, wolf: 8, goat: 10, player: 20,
};

function buildNearbyEntities() {
  const entities = [];
  if (!bot || !bot.entities) return entities;
  for (const entity of Object.values(bot.entities)) {
    if (entity === bot.entity) continue;
    if (!entity.position) continue;
    const dist = distance(bot.entity.position, entity.position);
    if (dist > 32) continue;
    const isHostile = isHostileMob(entity);
    const name = (entity.name || entity.type || 'unknown').toLowerCase();
    let health = entity.health;
    if (health === undefined || health === null) {
      health = DEFAULT_ENTITY_HEALTH[name] ?? (isHostile ? 20 : 10);
    }
    entities.push({
      type: entity.name || entity.type || 'unknown',
      distance: round2(dist),
      health,
      hostile: isHostile,
    });
  }
  entities.sort((a, b) => a.distance - b.distance);
  return entities.slice(0, 16);
}

function isHostileMob(entity) {
  if (!entity || !mcData) return false;
  const hostileTypes = [
    'zombie', 'skeleton', 'creeper', 'spider', 'enderman',
    'witch', 'slime', 'cave_spider', 'silverfish', 'blaze',
    'ghast', 'magma_cube', 'husk', 'stray', 'drowned',
    'phantom', 'pillager', 'vindicator', 'evoker', 'ravager',
    'vex', 'guardian', 'elder_guardian', 'hoglin', 'piglin_brute',
    'zombified_piglin', 'piglin', 'wither_skeleton', 'wither',
    'ender_dragon', 'shulker', 'warden',
  ];
  const name = (entity.name || entity.username || '').toLowerCase();
  return hostileTypes.some(h => name.includes(h)) || entity.type === 'hostile';
}

function buildNearbyBlocks(botPos) {
  const blocks = [];
  const searchRadius = 6;
  const centerX = Math.floor(botPos.x);
  const centerY = Math.floor(botPos.y);
  const centerZ = Math.floor(botPos.z);
  const importantBlocks = [
    'lava', 'water', 'fire', 'soul_fire', 'cactus',
    'magma_block', 'sweet_berry_bush', 'campfire',
    'soul_campfire', 'wither_rose',
  ];

  for (let dx = -searchRadius; dx <= searchRadius; dx++) {
    for (let dy = -searchRadius; dy <= searchRadius; dy++) {
      for (let dz = -searchRadius; dz <= searchRadius; dz++) {
        const x = centerX + dx;
        const y = centerY + dy;
        const z = centerZ + dz;
        const block = bot.blockAt(vec3(x, y, z));
        if (!block || block.name === 'air') continue;
        if (importantBlocks.includes(block.name)) {
          blocks.push({ type: block.name, position: [dx, dy, dz] });
        }
      }
    }
  }

  blocks.sort((a, b) => {
    const distA = Math.abs(a.position[0]) + Math.abs(a.position[1]) + Math.abs(a.position[2]);
    const distB = Math.abs(b.position[0]) + Math.abs(b.position[1]) + Math.abs(b.position[2]);
    return distA - distB;
  });

  return blocks.slice(0, 16);
}

function buildVoxelGrid(botPos) {
  const centerX = Math.floor(botPos.x);
  const centerY = Math.floor(botPos.y);
  const centerZ = Math.floor(botPos.z);
  const grid = new Uint8Array(VOXEL_SIZE);
  let idx = 0;

  for (let dy = -3; dy <= 3; dy++) {
    for (let dz = -5; dz <= 5; dz++) {
      for (let dx = -5; dx <= 5; dx++) {
        const x = centerX + dx;
        const y = centerY + dy;
        const z = centerZ + dz;
        const block = bot.blockAt(vec3(x, y, z));
        grid[idx] = getBlockTypeId(block);
        idx++;
      }
    }
  }

  return Buffer.from(grid).toString('base64');
}

function buildEnvironment(pos) {
  const canSeeSky = canBotSeeSky(pos);
  const inWater = bot.entity.isInWater || false;
  const onGround = bot.entity.onGround || false;
  const dangerLevel = computeDangerLevel(pos);

  return {
    time_of_day: getTimeOfDay(),
    can_see_sky: canSeeSky,
    in_water: inWater,
    on_ground: onGround,
    danger_level: round2(dangerLevel),
  };
}

function getTimeOfDay() {
  if (!bot.time) return 6000;
  return bot.time.timeOfDay || 0;
}

function canBotSeeSky(pos) {
  const block = bot.blockAt(vec3(Math.floor(pos.x), Math.floor(pos.y) + 1, Math.floor(pos.z)));
  if (!block || block.name === 'air') {
    const block2 = bot.blockAt(vec3(Math.floor(pos.x), Math.floor(pos.y) + 5, Math.floor(pos.z)));
    return !block2 || block2.name === 'air';
  }
  return false;
}

function computeDangerLevel(pos) {
  let danger = 0;
  if (!bot || !bot.entities) return danger;

  let hostileCount = 0;
  for (const entity of Object.values(bot.entities)) {
    if (entity === bot.entity) continue;
    if (!entity.position) continue;
    const dist = distance(bot.entity.position, entity.position);
    if (dist <= 8 && isHostileMob(entity)) hostileCount++;
  }
  danger += Math.min(hostileCount, 3) * 0.3;

  const health = bot.health || 20;
  if (health < 10) danger += 0.3;
  if (health < 5) danger += 0.2;

  if (checkLavaNearby(pos, 3)) danger += 0.3;
  if (bot.entity.onFire || bot.entity.metadata?.[0] & 0x01) danger += 0.1;

  const headBlock = bot.blockAt(bot.entity.position.offset(0, bot.entity.height, 0));
  if (headBlock && headBlock.name !== 'air' && headBlock.name !== 'cave_air') danger += 0.1;

  return clamp(danger, 0, 1);
}

function checkLavaNearby(pos, radius) {
  const cx = Math.floor(pos.x);
  const cy = Math.floor(pos.y);
  const cz = Math.floor(pos.z);
  for (let dx = -radius; dx <= radius; dx++) {
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dz = -radius; dz <= radius; dz++) {
        const block = bot.blockAt(vec3(cx + dx, cy + dy, cz + dz));
        if (block && (block.name === 'lava' || block.name === 'flowing_lava')) return true;
      }
    }
  }
  return false;
}

/**
 * Build the raw reward signal: event COUNTS since the last observation was
 * sent. All scaling into reward magnitudes happens on the Python side.
 */
function buildRewardSignal() {
  for (const kill of rewardState.killBuffer) {
    if (kill.type === 'mob') {
      rewardState.mobKills++;
      episodeStats.mobsKilled++;
    } else if (kill.type === 'player') {
      rewardState.playerKills++;
      episodeStats.playersKilled++;
    }
  }

  return {
    alive_tick: 1,
    damage_taken: round2(rewardState.cumulativeDamage),
    mob_killed: rewardState.mobKills,
    player_killed: rewardState.playerKills,
    item_mined: rewardState.itemsMined,
    item_lost: rewardState.itemsLost,
    item_crafted: rewardState.itemsCrafted,
    block_placed: rewardState.blocksPlaced,
    shelter_complete: rewardState.shelterComplete ? 1 : 0,
    distance_moved: round2(rewardState.distanceMoved),
    block_broken_name: rewardState.blocksBrokenNames.length > 0
      ? rewardState.blocksBrokenNames[rewardState.blocksBrokenNames.length - 1]
      : null,
    food_eaten: rewardState.foodEaten,
    death: rewardState.died ? 1 : 0,
  };
}

// =============================================================================
// RECONNECTION & SHUTDOWN
// =============================================================================

function attemptReconnect() {
  if (isShuttingDown) return;
  if (reconnectTimer) clearTimeout(reconnectTimer);
  log(LOG_LEVELS.INFO, 'Attempting to reconnect in 5 seconds...');
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    if (!isShuttingDown) {
      log(LOG_LEVELS.INFO, 'Reconnecting...');
      createBot();
    }
  }, 5000);
}

function shutdown() {
  log(LOG_LEVELS.INFO, 'Shutting down bot...');
  isShuttingDown = true;
  if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
  if (pythonSocket) { pythonSocket.destroy(); pythonSocket = null; }
  if (tcpServer) { tcpServer.close(() => { log(LOG_LEVELS.INFO, 'TCP server closed'); }); tcpServer = null; }
  if (bot) { bot.quit(); bot = null; }
  setTimeout(() => { log(LOG_LEVELS.INFO, 'Shutdown complete'); process.exit(0); }, 1000);
}

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
process.on('uncaughtException', (err) => {
  log(LOG_LEVELS.ERROR, 'Uncaught exception:', err.stack || err.message);
  shutdown();
});
process.on('unhandledRejection', (reason, promise) => {
  log(LOG_LEVELS.ERROR, 'Unhandled rejection at:', promise, 'reason:', reason);
});

// =============================================================================
// MAIN ENTRY POINT
// =============================================================================

log(LOG_LEVELS.INFO, '============================================');
log(LOG_LEVELS.INFO, '  Harvy RL Bot Starting');
log(LOG_LEVELS.INFO, `  Server: ${CONFIG.host}:${CONFIG.port}`);
log(LOG_LEVELS.INFO, `  Username: ${CONFIG.username}`);
log(LOG_LEVELS.INFO, `  TCP Port: ${CONFIG.tcpPort}`);
log(LOG_LEVELS.INFO, '============================================');

createBot();
