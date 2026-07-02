"use strict";

/**
 * Build the observation payload sent to the Python RL agent.
 */

const {
  VOXEL_XZ,
  VOXEL_Y,
  VOXEL_SIZE,
  MAX_ENTITIES,
  HOSTILE_MOBS,
} = require("./constants");

function buildObservation(bot, rewardSignal, overridden, extras = {}) {
  if (!bot || !bot.entity) return null;
  const pos = bot.entity.position;
  const yawDeg = (bot.entity.yaw * 180) / Math.PI;
  const pitchDeg = (bot.entity.pitch * 180) / Math.PI;

  return {
    obs: {
      self: buildSelfObservation(bot, pos, yawDeg, pitchDeg),
      nearby_entities: buildNearbyEntities(bot, pos),
      voxel_grid: buildVoxelGrid(bot, pos),
      environment: buildEnvironment(bot, pos),
    },
    reward_signal: rewardSignal,
    terminated: Boolean(rewardSignal.death),
    overridden: Boolean(overridden),
    death_context: extras.deathContext || null,
    chat_event: extras.chatEvent || null,
  };
}

function buildSelfObservation(bot, pos, yawDeg, pitchDeg) {
  const heldItem = bot.heldItem;
  return {
    health: bot.health || 0,
    food: bot.food || 0,
    armor: armorValue(bot),
    position: [round2(pos.x), round2(pos.y), round2(pos.z)],
    yaw: round2(normalizeYaw(yawDeg)),
    pitch: round2(Math.max(-90, Math.min(90, pitchDeg))),
    held_item: heldItem ? heldItem.name : "air",
    inventory_counts: inventoryCounts(bot),
  };
}

function buildNearbyEntities(bot, pos) {
  if (!bot.entities) return [];
  const entities = [];
  for (const entity of Object.values(bot.entities)) {
    if (entity === bot.entity) continue;
    if (!entity.position) continue;
    const dist = pos.distanceTo(entity.position);
    if (dist > 32) continue;
    const name = (entity.name || entity.username || "unknown").toLowerCase();
    entities.push({
      type: name,
      distance: round2(dist),
      health: entity.health || 0,
      hostile: HOSTILE_MOBS.has(name) || entity.type === "hostile",
      relative_position: [
        round2(entity.position.x - pos.x),
        round2(entity.position.y - pos.y),
        round2(entity.position.z - pos.z),
      ],
    });
  }
  entities.sort((a, b) => a.distance - b.distance);
  return entities.slice(0, MAX_ENTITIES);
}

function buildVoxelGrid(bot, pos) {
  const centerX = Math.floor(pos.x);
  const centerY = Math.floor(pos.y);
  const centerZ = Math.floor(pos.z);
  // Use Uint16 so state IDs up to ~17k don't collide. Python must decode
  // with dtype=np.uint16 and reshape to (VOXEL_XZ, VOXEL_XZ, VOXEL_Y)
  // i.e. (X, Z, Y). Fill order here is dx -> dz -> dy to match that.
  const grid = new Uint16Array(VOXEL_SIZE);
  let idx = 0;

  for (let dx = -Math.floor(VOXEL_XZ / 2); dx <= Math.floor(VOXEL_XZ / 2); dx++) {
    for (let dz = -Math.floor(VOXEL_XZ / 2); dz <= Math.floor(VOXEL_XZ / 2); dz++) {
      for (let dy = -Math.floor(VOXEL_Y / 2); dy <= Math.floor(VOXEL_Y / 2); dy++) {
        const block = bot.blockAt({
          x: centerX + dx,
          y: centerY + dy,
          z: centerZ + dz,
        });
        grid[idx] = block ? block.stateId || 0 : 0;
        idx++;
      }
    }
  }
  return Buffer.from(grid.buffer).toString("base64");
}

function buildEnvironment(bot, pos) {
  return {
    time_of_day: bot.time ? bot.time.timeOfDay || 0 : 6000,
    can_see_sky: canSeeSky(bot, pos),
    in_water: bot.entity.isInWater || false,
    on_ground: bot.entity.onGround || false,
    danger_level: computeDangerLevel(bot, pos),
  };
}

function canSeeSky(bot, pos) {
  const head = bot.blockAt({
    x: Math.floor(pos.x),
    y: Math.floor(pos.y) + 1,
    z: Math.floor(pos.z),
  });
  if (!head || head.name === "air") {
    const above = bot.blockAt({
      x: Math.floor(pos.x),
      y: Math.floor(pos.y) + 5,
      z: Math.floor(pos.z),
    });
    return !above || above.name === "air";
  }
  return false;
}

function computeDangerLevel(bot, pos) {
  let danger = 0;
  if (!bot.entities) return danger;

  let hostileCount = 0;
  for (const entity of Object.values(bot.entities)) {
    if (entity === bot.entity) continue;
    if (!entity.position) continue;
    const dist = pos.distanceTo(entity.position);
    if (dist <= 8) {
      const name = (entity.name || "").toLowerCase();
      if (HOSTILE_MOBS.has(name) || entity.type === "hostile") hostileCount++;
    }
  }
  danger += Math.min(hostileCount, 3) * 0.25;

  const health = bot.health || 20;
  if (health < 6) danger += 0.3;
  else if (health < 10) danger += 0.15;

  if (isInLava(bot, pos)) danger += 0.4;
  if (bot.entity.onFire) danger += 0.2;

  return Math.max(0, Math.min(1, danger));
}

function isInLava(bot, pos) {
  const foot = bot.blockAt({
    x: Math.floor(pos.x),
    y: Math.floor(pos.y),
    z: Math.floor(pos.z),
  });
  return foot && (foot.name === "lava" || foot.name === "flowing_lava");
}

function armorValue(bot) {
  if (!bot.inventory || !bot.inventory.slots) return 0;
  // Armor slots are 5-8 in Mineflayer; count pieces for now.
  let total = 0;
  for (let i = 5; i <= 8; i++) {
    if (bot.inventory.slots[i]) total += 1;
  }
  return total;
}

function inventoryCounts(bot) {
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

function normalizeYaw(yaw) {
  while (yaw < -180) yaw += 360;
  while (yaw > 180) yaw -= 360;
  return yaw;
}

function round2(v) {
  return Math.round(v * 100) / 100;
}

module.exports = { buildObservation, isInLava };
