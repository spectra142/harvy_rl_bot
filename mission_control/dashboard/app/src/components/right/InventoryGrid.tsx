import { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { HardHat, Shield, Shirt, Footprints } from 'lucide-react';
import { useDashboardStore } from '@/stores/dashboardStore';
import type { InventorySlot } from '@/stores/dashboardStore';

// ─── Item color map ──────────────────────────────────────────────────────────
const itemColorMap: Record<string, string> = {
  'Diamond Sword': '#3C44AA',
  'Diamond Pickaxe': '#3C44AA',
  'Diamond Axe': '#3C44AA',
  'Diamond Shovel': '#3C44AA',
  'Diamond Hoe': '#3C44AA',
  'Diamond': '#3C44AA',
  'Diamond Helmet': '#3C44AA',
  'Diamond Chestplate': '#3C44AA',
  'Diamond Leggings': '#3C44AA',
  'Diamond Boots': '#3C44AA',
  'Iron Sword': '#94a3b8',
  'Iron Axe': '#94a3b8',
  'Iron Pickaxe': '#94a3b8',
  'Iron Ingot': '#94a3b8',
  'Iron Helmet': '#94a3b8',
  'Iron Chestplate': '#94a3b8',
  'Iron Leggings': '#94a3b8',
  'Iron Boots': '#94a3b8',
  'Gold Ingot': '#F9B233',
  'Golden Apple': '#F9B233',
  'Oak Log': '#8B6914',
  'Oak Planks': '#B8956A',
  'Cobblestone': '#78716c',
  'Stone': '#78716c',
  'Cooked Porkchop': '#dc7f4a',
  'Cooked Beef': '#b05a2e',
  'Bread': '#d4a843',
  'Torch': '#F9FF3E',
  'Redstone Dust': '#B02E26',
  'Coal': '#1e293b',
  'Emerald': '#5D8C4A',
  'Lapis Lazuli': '#1d4ed8',
  'Obsidian': '#312e81',
  'Ender Pearl': '#10b981',
  'Water Bucket': '#3b82f6',
  'Lava Bucket': '#ea580c',
};

function getItemColor(itemName: string | null): string {
  if (!itemName) return '#475569';
  return itemColorMap[itemName] || '#64748b';
}

function formatItemLabel(itemName: string | null): string {
  if (!itemName) return '';
  return itemName;
}

// ─── Armor slot config ───────────────────────────────────────────────────────
const armorConfig = [
  { key: 'helmet' as const, icon: HardHat, label: 'Helmet' },
  { key: 'chestplate' as const, icon: Shield, label: 'Chest' },
  { key: 'leggings' as const, icon: Shirt, label: 'Legs' },
  { key: 'boots' as const, icon: Footprints, label: 'Boots' },
];

// ─── Slot component ──────────────────────────────────────────────────────────
interface SlotProps {
  slot: InventorySlot | null;
  size?: number;
  isSelected?: boolean;
  isDropTarget?: boolean;
  onDragStart?: (e: React.DragEvent) => void;
  onDragOver?: (e: React.DragEvent) => void;
  onDragLeave?: (e: React.DragEvent) => void;
  onDrop?: (e: React.DragEvent) => void;
  draggable?: boolean;
  armorIcon?: React.ComponentType<{ className?: string; style?: React.CSSProperties }>;
  className?: string;
}

function Slot({
  slot,
  size = 36,
  isSelected = false,
  isDropTarget = false,
  onDragStart,
  onDragOver,
  onDragLeave,
  onDrop,
  draggable = false,
  armorIcon: ArmorIcon,
  className = '',
}: SlotProps) {
  const [tooltipVisible, setTooltipVisible] = useState(false);

  const hasItem = slot && slot.item;
  const itemColor = getItemColor(slot?.item || null);
  const itemLabel = formatItemLabel(slot?.item || null);

  return (
    <div
      className="relative inline-flex items-center justify-center"
      style={{ width: size, height: size }}
      onMouseEnter={() => hasItem && setTooltipVisible(true)}
      onMouseLeave={() => setTooltipVisible(false)}
    >
      {/* Slot background */}
      <div
        className={`
          absolute inset-0 rounded-sm border
          ${isSelected ? 'border-[#F9FF3E] border-2' : ''}
          ${isDropTarget ? 'border-diamond-blue border-2' : ''}
          ${!isSelected && !isDropTarget ? 'border-slate-600' : ''}
          bg-slate-800
        `}
        style={{
          backgroundImage: !hasItem
            ? 'radial-gradient(circle at center, rgba(30,41,59,0.8) 0%, rgba(15,23,42,1) 100%)'
            : undefined,
        }}
      />

      {/* Drag surface */}
      <div
        className={`
          absolute inset-0 z-10 flex items-center justify-center
          ${draggable ? 'cursor-grab active:cursor-grabbing' : ''}
          ${className}
        `}
        draggable={draggable}
        onDragStart={onDragStart}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
      >
        {hasItem ? (
          <>
            {/* Item colored square */}
            <div
              className="rounded-sm"
              style={{
                width: size * 0.55,
                height: size * 0.55,
                backgroundColor: itemColor,
                boxShadow: `0 0 6px ${itemColor}66`,
              }}
            />
            {/* Item count */}
            {slot && slot.count > 1 && (
              <span
                className="absolute font-mono text-mono-xs text-white"
                style={{ bottom: 1, right: 3, fontSize: 10, lineHeight: 1 }}
              >
                {slot.count}
              </span>
            )}
          </>
        ) : ArmorIcon ? (
          <ArmorIcon
            className="text-slate-600"
            style={{ width: size * 0.5, height: size * 0.5 }}
          />
        ) : null}
      </div>

      {/* Tooltip */}
      {tooltipVisible && hasItem && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.15 }}
          className="absolute z-50 pointer-events-none"
          style={{
            bottom: size + 6,
            left: '50%',
            transform: 'translateX(-50%)',
            width: 'max-content',
          }}
        >
          <div
            className="bg-slate-800/95 border border-slate-600 rounded px-3 py-2"
            style={{ boxShadow: '0 4px 12px rgba(0,0,0,0.4)' }}
          >
            <div className="font-mono text-mono-sm text-slate-100 font-semibold">
              {itemLabel}
            </div>
            {slot && slot.count > 0 && (
              <div className="font-mono text-mono-xs text-slate-400">
                x{slot.count}
              </div>
            )}
            {slot && slot.durability !== undefined && slot.maxDurability && (
              <div className="font-mono text-mono-xs text-slate-500 mt-0.5">
                {slot.durability} / {slot.maxDurability}
              </div>
            )}
            {slot && slot.enchantments && slot.enchantments.length > 0 && (
              <div className="mt-1">
                {slot.enchantments.map((ench, i) => (
                  <div
                    key={i}
                    className="font-mono text-mono-xs text-mc-purple"
                  >
                    {ench}
                  </div>
                ))}
              </div>
            )}
          </div>
        </motion.div>
      )}
    </div>
  );
}

// ─── Main component ──────────────────────────────────────────────────────────
export default function InventoryGrid() {
  const inventory = useDashboardStore((s) => s.inventory);
  const armor = useDashboardStore((s) => s.armor);
  const offhand = useDashboardStore((s) => s.offhand);
  const selectedHotbar = useDashboardStore((s) => s.selectedHotbar);
  const moveItem = useDashboardStore((s) => s.moveItem);

  const [draggingIndex, setDraggingIndex] = useState<number | null>(null);
  const [dropTargetIndex, setDropTargetIndex] = useState<number | null>(null);

  const handleDragStart = useCallback(
    (index: number) => (e: React.DragEvent) => {
      setDraggingIndex(index);
      e.dataTransfer.effectAllowed = 'move';
      // Set a small drag image by creating a transparent one
      const canvas = document.createElement('canvas');
      canvas.width = 1;
      canvas.height = 1;
      e.dataTransfer.setDragImage(canvas, 0, 0);
    },
    []
  );

  const handleDragOver = useCallback(
    (index: number) => (e: React.DragEvent) => {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'move';
      if (draggingIndex !== null && draggingIndex !== index) {
        setDropTargetIndex(index);
      }
    },
    [draggingIndex]
  );

  const handleDragLeave = useCallback(() => {
    setDropTargetIndex(null);
  }, []);

  const handleDrop = useCallback(
    (index: number) => (e: React.DragEvent) => {
      e.preventDefault();
      if (draggingIndex !== null && draggingIndex !== index) {
        moveItem(draggingIndex, index);
      }
      setDraggingIndex(null);
      setDropTargetIndex(null);
    },
    [draggingIndex, moveItem]
  );

  const handleDragEnd = useCallback(() => {
    setDraggingIndex(null);
    setDropTargetIndex(null);
  }, []);

  // Build slot arrays
  const hotbarSlots = inventory.slice(0, 9);
  const storageSlots = inventory.slice(9, 36);
  const filledSlots = inventory.filter((s) => s.item).length;

  return (
    <div className="h-full flex flex-col bg-slate-900 border-b border-slate-700">
      {/* Header */}
      <div className="h-8 bg-slate-800 border-b border-slate-700 flex items-center justify-between px-3 border-l-[3px] border-l-gold">
        <span className="text-ui-sm text-slate-400 uppercase tracking-wider">
          INVENTORY
        </span>
        <span className="font-mono text-mono-xs text-slate-500">
          {filledSlots}/40 slots
        </span>
      </div>

      {/* Inventory body */}
      <div className="flex-1 p-2 flex flex-col gap-2 overflow-hidden">
        {/* ── Storage Grid (27 slots = 9x3) ── */}
        <div className="flex gap-1 flex-1">
          {/* Armor column on left */}
          <div className="flex flex-col gap-[3px] pr-1 justify-start pt-1">
            {armorConfig.map(({ key, icon }) => (
              <Slot
                key={key}
                slot={armor[key]}
                size={36}
                armorIcon={icon}
              />
            ))}
          </div>

          {/* 9x3 Storage grid */}
          <div className="flex-1 grid grid-cols-9 gap-[3px] content-start">
            {storageSlots.map((slot, i) => {
              const invIndex = 9 + i;
              return (
                <Slot
                  key={invIndex}
                  slot={slot}
                  size={36}
                  draggable={!!slot.item}
                  isDropTarget={dropTargetIndex === invIndex && draggingIndex !== invIndex}
                  onDragStart={handleDragStart(invIndex)}
                  onDragOver={handleDragOver(invIndex)}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop(invIndex)}
                />
              );
            })}
          </div>
        </div>

        {/* ── Offhand + Hotbar Row ── */}
        <div className="flex items-center gap-2 pt-1 border-t border-slate-800">
          {/* Offhand slot */}
          <div className="flex flex-col items-center gap-[2px]">
            <span className="font-mono text-[9px] text-slate-600 uppercase">
              OFF
            </span>
            <Slot
              slot={offhand}
              size={36}
            />
          </div>

          {/* Divider */}
          <div className="w-px h-10 bg-slate-700 mx-1" />

          {/* Hotbar 9 slots */}
          <div className="flex gap-[3px] flex-1">
            {hotbarSlots.map((slot, i) => (
              <Slot
                key={i}
                slot={slot}
                size={40}
                isSelected={selectedHotbar === i}
                draggable={!!slot.item}
                isDropTarget={dropTargetIndex === i && draggingIndex !== i}
                onDragStart={handleDragStart(i)}
                onDragOver={handleDragOver(i)}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop(i)}
              />
            ))}
          </div>
        </div>
      </div>

      {/* Global drag end handler */}
      {draggingIndex !== null && (
        <div
          className="fixed inset-0 z-40"
          style={{ pointerEvents: 'none' }}
          onDragEnd={handleDragEnd}
        />
      )}
    </div>
  );
}

