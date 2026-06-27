import { motion } from 'framer-motion';
import { Users, Gauge, Hash, Swords } from 'lucide-react';
import { useDashboardStore } from '@/stores/dashboardStore';

export default function ServerStatus() {
  const { playerCount, maxPlayers, tps, worldSeed, difficulty } = useDashboardStore();

  // TPS color logic
  let tpsColor = '#5D8C4A'; // green >= 18
  if (tps < 15) {
    tpsColor = '#B02E26'; // red
  } else if (tps < 18) {
    tpsColor = '#F9FF3E'; // amber
  }

  // Difficulty color logic
  let difficultyColor = '#5D8C4A'; // Easy/Peaceful
  const diffLower = difficulty.toLowerCase();
  if (diffLower === 'hard' || diffLower === 'hardcore') {
    difficultyColor = '#B02E26';
  } else if (diffLower === 'normal') {
    difficultyColor = '#F9FF3E';
  }

  return (
    <div className="bg-slate-800 border border-slate-700 rounded p-2.5 mx-3 mb-3">
      {/* Header with LIVE indicator */}
      <div className="flex items-center justify-between mb-2">
        <span className="font-sans text-ui-xs text-slate-500 uppercase tracking-wider">
          SERVER STATUS
        </span>
        <div className="flex items-center gap-1.5">
          <motion.div
            className="w-1.5 h-1.5 rounded-full bg-grass-green"
            animate={{ opacity: [0.4, 1, 0.4], scale: [1, 1.2, 1] }}
            transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
            style={{ boxShadow: '0 0 6px #5D8C4A' }}
          />
          <span className="font-sans text-ui-xs text-grass-green uppercase tracking-wider">
            LIVE
          </span>
        </div>
      </div>

      {/* 2x2 Grid */}
      <div className="grid grid-cols-2 gap-2">
        {/* Players */}
        <div className="flex flex-col gap-0.5">
          <div className="flex items-center gap-1">
            <Users className="w-3 h-3 text-slate-500" />
            <span className="font-sans text-ui-xs text-slate-500 uppercase tracking-wider">
              Players
            </span>
          </div>
          <span className="font-mono text-mono-lg text-quartz-white">
            {playerCount} / {maxPlayers}
          </span>
        </div>

        {/* TPS */}
        <div className="flex flex-col gap-0.5">
          <div className="flex items-center gap-1">
            <Gauge className="w-3 h-3 text-slate-500" />
            <span className="font-sans text-ui-xs text-slate-500 uppercase tracking-wider">
              TPS
            </span>
          </div>
          <span
            className="font-mono text-mono-lg transition-colors duration-200"
            style={{ color: tpsColor }}
          >
            {tps.toFixed(1)}
          </span>
        </div>

        {/* World Seed */}
        <div className="flex flex-col gap-0.5">
          <div className="flex items-center gap-1">
            <Hash className="w-3 h-3 text-slate-500" />
            <span className="font-sans text-ui-xs text-slate-500 uppercase tracking-wider">
              World Seed
            </span>
          </div>
          <span className="font-mono text-mono-sm text-slate-400 truncate" title={worldSeed}>
            {worldSeed}
          </span>
        </div>

        {/* Difficulty */}
        <div className="flex flex-col gap-0.5">
          <div className="flex items-center gap-1">
            <Swords className="w-3 h-3 text-slate-500" />
            <span className="font-sans text-ui-xs text-slate-500 uppercase tracking-wider">
              Difficulty
            </span>
          </div>
          <span
            className="font-mono text-mono-sm transition-colors duration-200"
            style={{ color: difficultyColor }}
          >
            {difficulty}
          </span>
        </div>
      </div>
    </div>
  );
}
