import { useState } from 'react';
import { motion } from 'framer-motion';
import { useDashboardStore } from '@/stores/dashboardStore';
import type { ActionProb } from '@/stores/dashboardStore';

// ─── Action Color Map ────────────────────────────────────────────────────────

const ACTION_COLORS: Record<string, { bg: string; hover: string; glow: string }> = {
  move:   { bg: '#5D8C4A', hover: '#6DA35A', glow: 'rgba(93,140,74,0.4)' },
  jump:   { bg: '#D97706', hover: '#E68A1A', glow: 'rgba(217,119,6,0.4)' },
  attack: { bg: '#B02E26', hover: '#C43E36', glow: 'rgba(176,46,38,0.4)' },
  mine:   { bg: '#cbd5e1', hover: '#e2e8f0', glow: 'rgba(203,213,225,0.4)' },
  place:  { bg: '#3C44AA', hover: '#4C54BA', glow: 'rgba(60,68,170,0.4)' },
  look:   { bg: '#8B5CF6', hover: '#9B6CF6', glow: 'rgba(139,92,246,0.4)' },
  craft:  { bg: '#F9B233', hover: '#FAC24D', glow: 'rgba(249,178,51,0.4)' },
  idle:   { bg: '#475569', hover: '#576579', glow: 'rgba(71,85,105,0.4)' },
};

// ─── Action Segment Component ────────────────────────────────────────────────

interface ActionSegmentProps {
  action: ActionProb;
  isHovered: boolean;
  isAnyHovered: boolean;
  onHover: () => void;
  onLeave: () => void;
}

function ActionSegment({ action, isHovered, isAnyHovered, onHover, onLeave }: ActionSegmentProps) {
  const actionKey = action.action.toLowerCase();
  const colors = ACTION_COLORS[actionKey] || ACTION_COLORS.idle;
  const pct = Math.round(action.prob * 100);

  return (
    <motion.div
      className="relative h-full flex items-center justify-center overflow-hidden cursor-pointer"
      style={{
        backgroundColor: isHovered ? colors.hover : colors.bg,
        opacity: isAnyHovered && !isHovered ? 0.5 : 1,
        boxShadow: isHovered ? `0 0 12px ${colors.glow}` : 'none',
        transition: 'background-color 150ms ease, opacity 150ms ease, box-shadow 150ms ease',
      }}
      animate={{ width: `${pct}%` }}
      transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] as [number, number, number, number] }}
      onMouseEnter={onHover}
      onMouseLeave={onLeave}
      title={`${action.action}: ${(action.prob * 100).toFixed(1)}%`}
    >
      {pct >= 8 && (
        <span className="font-mono text-mono-xs text-white/90 truncate px-1">
          {action.action.toUpperCase()}
        </span>
      )}

      {/* Tooltip on hover */}
      {isHovered && (
        <div className="absolute -top-8 left-1/2 -translate-x-1/2 bg-slate-800 border border-slate-600 rounded px-2 py-1 whitespace-nowrap z-20 pointer-events-none shadow-lg">
          <span className="font-mono text-mono-xs text-quartz-white">
            {action.action}: {(action.prob * 100).toFixed(1)}%
          </span>
        </div>
      )}
    </motion.div>
  );
}

// ─── Main Action Distribution Panel ──────────────────────────────────────────

export default function ActionDistribution() {
  const currentActionProbs = useDashboardStore((s) => s.currentActionProbs);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  // Normalize to ensure they sum to 100%
  const total = currentActionProbs.reduce((sum, a) => sum + a.prob, 0);
  const normalized = total > 0
    ? currentActionProbs.map((a) => ({ ...a, prob: a.prob / total }))
    : currentActionProbs;

  return (
    <div className="flex flex-col h-[15%] border-t border-slate-700">
      {/* Panel Header */}
      <div className="h-7 bg-slate-800 border-b border-slate-700 flex items-center px-3 border-l-[3px] border-l-mc-purple">
        <span className="font-ui text-ui-sm text-slate-400 uppercase tracking-wider">
          ACTION DISTRIBUTION
        </span>
        <span className="ml-auto font-mono text-mono-xs text-slate-500">
          Last Step
        </span>
      </div>

      {/* Bar */}
      <div className="flex-1 flex flex-col justify-center px-3 py-2 bg-slate-900">
        <div className="flex w-full h-6 rounded-sm overflow-hidden">
          {normalized.map((action, i) => (
            <ActionSegment
              key={action.action}
              action={action}
              isHovered={hoveredIndex === i}
              isAnyHovered={hoveredIndex !== null}
              onHover={() => setHoveredIndex(i)}
              onLeave={() => setHoveredIndex(null)}
            />
          ))}
        </div>

        {/* Legend row */}
        <div className="flex justify-between mt-1.5 px-0.5">
          {normalized.map((action) => {
            const actionKey = action.action.toLowerCase();
            const colors = ACTION_COLORS[actionKey] || ACTION_COLORS.idle;
            return (
              <div key={action.action} className="flex items-center gap-1">
                <div
                  className="w-1.5 h-1.5 rounded-full"
                  style={{ backgroundColor: colors.bg }}
                />
                <span className="font-mono text-mono-xs text-slate-500">
                  {action.action}
                </span>
                <span className="font-mono text-mono-xs text-slate-400 ml-0.5">
                  {(action.prob * 100).toFixed(0)}%
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
