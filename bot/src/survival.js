"use strict";

/**
 * Minimal survival override layer.
 *
 * Only intervenes for true instant-death conditions. Everything else is the
 * RL agent's problem to learn.
 */

const pathfinder = require("mineflayer-pathfinder");
const Vec3 = require("vec3");

const { VOID_Y } = require("./constants");
const { isInLava } = require("./observations");

function checkSurvival(bot) {
  if (!bot || !bot.entity) return { active: false, action: null, reason: null };

  const pos = bot.entity.position;

  // 1. Suffocation: head in solid block.
  const headBlock = bot.blockAt({
    x: Math.floor(pos.x),
    y: Math.floor(pos.y + bot.entity.height),
    z: Math.floor(pos.z),
  });
  if (
    headBlock &&
    headBlock.name !== "air" &&
    headBlock.name !== "cave_air" &&
    headBlock.name !== "water" &&
    headBlock.name !== "flowing_water"
  ) {
    if (bot.canDigBlock(headBlock)) {
      bot.dig(headBlock).catch(() => {});
    }
    return { active: true, action: "break_head", reason: "suffocating" };
  }

  // 2. In lava: jump and move toward nearest solid block.
  if (isInLava(bot, pos)) {
    bot.setControlState("jump", true);
    const solid = nearestSolidBlock(bot, pos, 3);
    if (solid && bot.pathfinder) {
      bot.pathfinder.setGoal(
        new pathfinder.goals.GoalNear(
          solid.position.x,
          solid.position.y + 1,
          solid.position.z,
          1
        )
      );
    }
    return { active: true, action: "escape_lava", reason: "in_lava" };
  }

  // 3. Void fall.
  if (pos.y < VOID_Y + 10) {
    bot.setControlState("jump", true);
    return { active: true, action: "jump", reason: "near_void" };
  }

  return { active: false, action: null, reason: null };
}

function nearestSolidBlock(bot, pos, radius) {
  const cx = Math.floor(pos.x);
  const cy = Math.floor(pos.y);
  const cz = Math.floor(pos.z);
  for (let r = 1; r <= radius; r++) {
    for (let dx = -r; dx <= r; dx++) {
      for (let dz = -r; dz <= r; dz++) {
        for (let dy = -1; dy <= 2; dy++) {
          const blockPos = new Vec3(cx + dx, cy + dy, cz + dz);
          const block = bot.blockAt(blockPos);
          if (
            block &&
            block.name !== "air" &&
            block.name !== "cave_air" &&
            block.name !== "lava" &&
            block.name !== "flowing_lava" &&
            block.name !== "water" &&
            block.name !== "flowing_water"
          ) {
            return block;
          }
        }
      }
    }
  }
  return null;
}

module.exports = { checkSurvival };
