"use strict";

/**
 * TCP server that talks to the Python RL agent.
 *
 * Protocol: line-delimited JSON.
 * Python -> JS: { seq: int, action: {...} }
 * JS -> Python: { seq: int, obs: {...}, reward_signal: {...}, terminated: bool, overridden: bool }
 */

const net = require("net");
const { applyAction, releaseAllControls } = require("./actions");
const { checkSurvival } = require("./survival");

class BotConnection {
  constructor(port, getObservation, getBot) {
    this.port = port;
    this.getObservation = getObservation;
    this.getBot = getBot;
    this.server = null;
    this.socket = null;
    this.lastSeq = -1;
  }

  start() {
    if (this.server) return;

    this.server = net.createServer((socket) => {
      if (this.socket && !this.socket.destroyed) {
        console.log("[conn] new client; closing old socket");
        this.socket.destroy();
      }
      console.log(`[conn] python connected from ${socket.remoteAddress}`);
      this.socket = socket;
      socket.setEncoding("utf8");
      let buffer = "";

      socket.on("data", (data) => {
        buffer += data;
        const lines = buffer.split("\n");
        buffer = lines.pop();
        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed) this.handleMessage(trimmed);
        }
      });

      socket.on("end", () => {
        console.log("[conn] python disconnected");
        this.socket = null;
      });

      socket.on("error", (err) => {
        console.error("[conn] socket error:", err.message);
        this.socket = null;
      });
    });

    this.server.listen(this.port, "0.0.0.0", () => {
      console.log(`[conn] tcp server listening on port ${this.port}`);
    });

    this.server.on("error", (err) => {
      console.error("[conn] server error:", err.message);
    });
  }

  handleMessage(line) {
    let msg;
    try {
      msg = JSON.parse(line);
    } catch (err) {
      console.warn("[conn] invalid json:", line);
      return;
    }

    const bot = this.getBot();
    if (!bot || !bot.entity) {
      this.send({ error: "bot_not_ready" });
      return;
    }

    // Reset command.
    if (msg.reset) {
      releaseAllControls(bot);
      this.lastSeq = -1;
      this.sendObservation(msg.seq || 0, false);
      return;
    }

    const seq = msg.seq ?? this.lastSeq + 1;
    this.lastSeq = seq;

    // Apply survival layer first.
    const survival = checkSurvival(bot);
    if (survival.active) {
      this.sendObservation(seq, true);
      return;
    }

    // Apply RL action.
    if (msg.action) {
      applyAction(bot, msg.action);
    }

    this.sendObservation(seq, false);
  }

  sendObservation(seq, overridden) {
    const payload = this.getObservation(overridden);
    payload.seq = seq;
    this.send(payload);
  }

  send(payload) {
    if (!this.socket || this.socket.destroyed) return;
    try {
      this.socket.write(JSON.stringify(payload) + "\n");
    } catch (err) {
      console.error("[conn] send failed:", err.message);
    }
  }

  close() {
    if (this.socket) {
      this.socket.destroy();
      this.socket = null;
    }
    if (this.server) {
      this.server.close();
      this.server = null;
    }
  }
}

module.exports = { BotConnection };
