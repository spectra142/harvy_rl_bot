"use strict";

/**
 * Shared constants for the Mineflayer bot.
 */

// Voxel grid around the bot: 11x11 in XZ, 7 in Y.
const VOXEL_XZ = 11;
const VOXEL_Y = 7;
const VOXEL_SIZE = VOXEL_XZ * VOXEL_XZ * VOXEL_Y; // 847

// Entity / block list limits.
const MAX_ENTITIES = 16;

// Hostile mob list used for danger calculations.
const HOSTILE_MOBS = new Set([
  "zombie", "skeleton", "creeper", "spider", "enderman", "witch", "slime",
  "cave_spider", "silverfish", "blaze", "ghast", "magma_cube", "husk",
  "stray", "drowned", "phantom", "pillager", "vindicator", "evoker",
  "ravager", "vex", "guardian", "elder_guardian", "hoglin", "piglin_brute",
  "zombified_piglin", "wither_skeleton", "wither", "ender_dragon",
  "shulker", "warden",
]);

// Emergency thresholds.
const VOID_Y = -64;

module.exports = {
  VOXEL_XZ,
  VOXEL_Y,
  VOXEL_SIZE,
  MAX_ENTITIES,
  HOSTILE_MOBS,
  VOID_Y,
};
