"use strict";

/**
 * Bot lifecycle: create, spawn, death, reconnect, shutdown.
 */

const mineflayer = require("mineflayer");
const pathfinder = require("mineflayer-pathfinder");

let bot = null;
let reconnectTimer = null;
let isShuttingDown = false;

function createBot(config, onSpawn, onDeath) {
  if (bot) {
    try {
      bot.removeAllListeners();
      bot.quit();
    } catch (e) {
      // ignore
    }
    bot = null;
  }

  bot = mineflayer.createBot({
    host: config.mc_host,
    port: config.mc_port,
    username: config.mc_username,
    version: config.mc_version,
    auth: "offline",
    checkTimeoutInterval: 30000,
  });

  bot.loadPlugin(pathfinder.pathfinder);

  bot.once("login", () => {
    console.log(`[bot] logged in as ${bot.username} on ${bot.version}`);
  });

  bot.once("spawn", () => {
    console.log("[bot] spawned");
    if (onSpawn) onSpawn(bot);
  });

  bot.on("death", () => {
    console.log("[bot] died");
    if (onDeath) onDeath();
  });

  bot.on("kicked", (reason) => {
    console.log("[bot] kicked:", reason);
    scheduleReconnect(config, onSpawn, onDeath);
  });

  bot.on("end", () => {
    console.log("[bot] disconnected");
    if (!isShuttingDown) {
      scheduleReconnect(config, onSpawn, onDeath);
    }
  });

  bot.on("error", (err) => {
    console.error("[bot] error:", err.message);
  });

  return bot;
}

function getBot() {
  return bot;
}

function scheduleReconnect(config, onSpawn, onDeath) {
  if (isShuttingDown || reconnectTimer) return;
  console.log("[bot] reconnecting in 5s...");
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    if (!isShuttingDown) {
      createBot(config, onSpawn, onDeath);
    }
  }, 5000);
}

function shutdown() {
  isShuttingDown = true;
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
  if (bot) {
    try {
      bot.quit();
    } catch (e) {
      // ignore
    }
    bot = null;
  }
}

module.exports = { createBot, getBot, shutdown };
