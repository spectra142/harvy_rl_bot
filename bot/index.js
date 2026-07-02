"use strict";

/**
 * Entry point for the Harvy v2 Mineflayer bot.
 *
 * Configuration is loaded from configs/default.yaml by default.
 * You can override any value with environment variables:
 *   MC_HOST, MC_PORT, MC_USERNAME, MC_VERSION, TCP_PORT, HARVY_CONFIG
 */

const fs = require("fs");
const path = require("path");
const yaml = require("js-yaml");

const { createBot, getBot, shutdown } = require("./src/bot");
const { buildObservation } = require("./src/observations");
const { BotConnection } = require("./src/connection");
const { releaseAllControls } = require("./src/actions");
const { HOSTILE_MOBS } = require("./src/constants");

function loadConfig() {
  const configPath =
    process.env.HARVY_CONFIG ||
    path.join(__dirname, "..", "configs", "default.yaml");

  let fileConfig = {};
  if (fs.existsSync(configPath)) {
    try {
      fileConfig = yaml.load(fs.readFileSync(configPath, "utf8")) || {};
    } catch (err) {
      console.warn(`[config] failed to load ${configPath}: ${err.message}`);
    }
  } else {
    console.warn(`[config] file not found: ${configPath}`);
  }

  const env = fileConfig.environment || {};

  return {
    mc_host: process.env.MC_HOST || env.mc_host || "localhost",
    mc_port: parseInt(process.env.MC_PORT || env.mc_port || "25565", 10),
    mc_username: process.env.MC_USERNAME || env.mc_username || "harvy",
    mc_version: process.env.MC_VERSION || env.mc_version || "1.21.1",
    tcp_port: parseInt(process.env.TCP_PORT || env.port || "9876", 10),
  };
}

const CONFIG = loadConfig();

let connection = null;

// Reward signal state per tick.
const rewardState = {
  lastHealth: 20,
  lastFood: 20,
  cumulativeDamage: 0,
  itemsMined: 0,
  foodEaten: 0,
  hostilesKilled: 0,
  died: false,
};

// Async events that need to be reported with the next observation.
const pendingEvent = {
  deathContext: null,
  chatEvent: null,
};

function resetRewardState(bot) {
  rewardState.lastHealth = bot.health || 20;
  rewardState.lastFood = bot.food || 20;
  rewardState.cumulativeDamage = 0;
  rewardState.itemsMined = 0;
  rewardState.foodEaten = 0;
  rewardState.hostilesKilled = 0;
  rewardState.died = false;
  pendingEvent.deathContext = null;
  pendingEvent.chatEvent = null;
}

function buildRewardSignal(bot) {
  const death = rewardState.died ? 1 : 0;
  const signal = {
    alive_tick: 1,
    damage_taken: round2(rewardState.cumulativeDamage),
    item_mined: rewardState.itemsMined,
    food_eaten: rewardState.foodEaten,
    hostiles_killed: rewardState.hostilesKilled,
    death: death,
  };

  // Reset per-tick accumulators.
  rewardState.cumulativeDamage = 0;
  rewardState.itemsMined = 0;
  rewardState.foodEaten = 0;
  rewardState.hostilesKilled = 0;
  if (rewardState.died) rewardState.died = false;

  return signal;
}

function buildExtras() {
  const extras = {
    deathContext: pendingEvent.deathContext,
    chatEvent: pendingEvent.chatEvent,
  };
  // Chat events are one-shot; death context is kept until Python sees terminated.
  pendingEvent.chatEvent = null;
  return extras;
}

function onSpawn(bot) {
  resetRewardState(bot);
  releaseAllControls(bot);

  bot.on("health", () => {
    if (bot.health < rewardState.lastHealth) {
      rewardState.cumulativeDamage += rewardState.lastHealth - bot.health;
    }
    rewardState.lastHealth = bot.health;
  });

  bot.on("diggingCompleted", () => {
    rewardState.itemsMined += 1;
  });

  bot.on("eat", () => {
    rewardState.foodEaten += 1;
  });

  bot.on("death", () => {
    rewardState.died = true;
    pendingEvent.deathContext = {
      cause: "unknown",
      killer_uuid: null,
      killer_name: null,
      position: bot.entity ? [
        round2(bot.entity.position.x),
        round2(bot.entity.position.y),
        round2(bot.entity.position.z),
      ] : [0, 0, 0],
      health: 0,
      inventory: bot.inventory ? inventorySummary(bot) : {},
    };
  });

  bot.on("entityDead", (entity) => {
    if (!entity || !entity.position || !bot.entity) return;
    const dist = bot.entity.position.distanceTo(entity.position);
    if (dist > 8) return;
    const name = (entity.name || "").toLowerCase();
    if (HOSTILE_MOBS.has(name) || entity.type === "hostile") {
      rewardState.hostilesKilled += 1;
    }
  });

  bot.on("message", (jsonMsg) => {
    const text = jsonMsg.toString();
    // Best-effort parse for chat events. Mineflayer also has a dedicated
    // 'chat' event with (username, message); prefer that when available.
    pendingEvent.chatEvent = {
      username: null,
      message: text,
      position: bot.entity ? [
        round2(bot.entity.position.x),
        round2(bot.entity.position.y),
        round2(bot.entity.position.z),
      ] : null,
    };
  });

  bot.on("chat", (username, message) => {
    pendingEvent.chatEvent = {
      username: username,
      message: message,
      position: bot.entity ? [
        round2(bot.entity.position.x),
        round2(bot.entity.position.y),
        round2(bot.entity.position.z),
      ] : null,
    };
  });

  // Observations are sent only in response to Python actions.
  // Reward events are accumulated here and flushed when the connection
  // builds the next observation.
}

function inventorySummary(bot) {
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

function round2(v) {
  return Math.round(v * 100) / 100;
}

function main() {
  console.log("============================================");
  console.log("  Harvy RL Bot v2");
  console.log(`  MC: ${CONFIG.mc_host}:${CONFIG.mc_port}`);
  console.log(`  TCP: ${CONFIG.tcp_port}`);
  console.log("============================================");

  connection = new BotConnection(
    CONFIG.tcp_port,
    (overridden) => {
      const bot = getBot();
      if (!bot) return null;
      return buildObservation(
        bot,
        buildRewardSignal(bot),
        overridden,
        buildExtras()
      );
    },
    getBot
  );
  connection.start();

  createBot(CONFIG, onSpawn);

  process.on("SIGINT", () => {
    console.log("\n[main] shutting down");
    connection.close();
    shutdown();
    process.exit(0);
  });

  process.on("SIGTERM", () => {
    connection.close();
    shutdown();
    process.exit(0);
  });
}

main();
