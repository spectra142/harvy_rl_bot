"use strict";

/**
 * Low-level action execution.
 *
 * The RL agent sends discrete actions that map to control states,
 * camera deltas, hotbar selection, and single-tick interactions.
 */

const Vec3 = require("vec3");

const DEFAULT_ACTION = {
  forward: 0,
  back: 0,
  left: 0,
  right: 0,
  jump: 0,
  sneak: 0,
  sprint: 0,
  attack: 0,
  mine: 0,
  use: 0,
  place_block: 0,
  eat: 0,
  hotbar: null,
  yaw_delta: 0,
  pitch_delta: 0,
};

const FOOD_NAMES = new Set([
  "apple",
  "baked_potato",
  "beetroot",
  "beetroot_soup",
  "bread",
  "carrot",
  "chorus_fruit",
  "cooked_beef",
  "cooked_chicken",
  "cooked_cod",
  "cooked_mutton",
  "cooked_porkchop",
  "cooked_rabbit",
  "cooked_salmon",
  "cookie",
  "dried_kelp",
  "enchanted_golden_apple",
  "golden_apple",
  "golden_carrot",
  "honey_bottle",
  "melon_slice",
  "mushroom_stew",
  "poisonous_potato",
  "potato",
  "pufferfish",
  "pumpkin_pie",
  "rabbit_stew",
  "raw_beef",
  "raw_chicken",
  "raw_cod",
  "raw_mutton",
  "raw_porkchop",
  "raw_rabbit",
  "raw_salmon",
  "rotten_flesh",
  "suspicious_stew",
  "sweet_berries",
  "tropical_fish",
]);

function releaseAllControls(bot) {
  if (!bot) return;
  bot.setControlState("forward", false);
  bot.setControlState("back", false);
  bot.setControlState("left", false);
  bot.setControlState("right", false);
  bot.setControlState("jump", false);
  bot.setControlState("sneak", false);
  bot.setControlState("sprint", false);
}

function applyAction(bot, action) {
  if (!bot || !bot.entity) return;

  const a = { ...DEFAULT_ACTION, ...action };

  // Movement controls.
  bot.setControlState("forward", Boolean(a.forward));
  bot.setControlState("back", Boolean(a.back));
  bot.setControlState("left", Boolean(a.left));
  bot.setControlState("right", Boolean(a.right));
  bot.setControlState("jump", Boolean(a.jump));
  bot.setControlState("sneak", Boolean(a.sneak));
  bot.setControlState("sprint", Boolean(a.sprint));

  // Camera deltas (degrees). Use bot.look() so the server is informed.
  const yawDelta = a.yaw_delta || 0;
  const pitchDelta = a.pitch_delta || 0;
  if (yawDelta !== 0 || pitchDelta !== 0) {
    const newYaw = bot.entity.yaw + (yawDelta * Math.PI) / 180;
    const newPitch = Math.max(
      -Math.PI / 2,
      Math.min(Math.PI / 2, bot.entity.pitch + (pitchDelta * Math.PI) / 180)
    );
    bot.look(newYaw, newPitch, true).catch(() => {});
  }

  // Hotbar selection.
  if (a.hotbar !== null && a.hotbar >= 0 && a.hotbar <= 8) {
    bot.setQuickBarSlot(a.hotbar);
  }

  // Single-tick interactions.
  if (a.attack) {
    const target = nearestAttackableEntity(bot, 4.5);
    if (target) bot.attack(target);
  }

  if (a.mine) {
    const block = bot.blockAtCursor(4);
    if (block && bot.canDigBlock(block)) {
      bot.dig(block).catch(() => {});
    }
  }

  if (a.place_block) {
    placeHeldBlock(bot);
  } else if (a.eat) {
    eatHeldFood(bot);
  } else if (a.use) {
    const block = bot.blockAtCursor(4);
    if (block) {
      bot.activateBlock(block).catch(() => {});
    } else {
      bot.activateItem();
    }
  }
}

function nearestAttackableEntity(bot, maxDistance) {
  if (!bot || !bot.entities) return null;
  let nearest = null;
  let nearestDist = Infinity;
  for (const entity of Object.values(bot.entities)) {
    if (entity === bot.entity) continue;
    if (!entity.position) continue;
    const dist = bot.entity.position.distanceTo(entity.position);
    if (dist < nearestDist && dist <= maxDistance) {
      nearest = entity;
      nearestDist = dist;
    }
  }
  return nearest;
}

function placeHeldBlock(bot) {
  const held = bot.heldItem;
  if (!held) return;
  // Heuristic: items that are not placeable blocks will fail bot.placeBlock
  // quickly, which is fine. We don't maintain a full block list here.
  const ref = bot.blockAtCursor(4);
  if (!ref) return;

  const faces = [
    new Vec3(0, 1, 0),
    new Vec3(0, -1, 0),
    new Vec3(1, 0, 0),
    new Vec3(-1, 0, 0),
    new Vec3(0, 0, 1),
    new Vec3(0, 0, -1),
  ];

  for (const face of faces) {
    const targetPos = ref.position.plus(face);
    const targetBlock = bot.blockAt(targetPos);
    if (targetBlock && targetBlock.name === "air") {
      bot.placeBlock(ref, face).catch(() => {});
      return;
    }
  }
}

function eatHeldFood(bot) {
  const held = bot.heldItem;
  if (!held || !FOOD_NAMES.has(held.name)) return;
  bot.consume().catch(() => {});
}

module.exports = { applyAction, releaseAllControls, DEFAULT_ACTION };
