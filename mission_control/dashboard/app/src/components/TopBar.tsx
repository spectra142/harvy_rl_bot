import { motion } from 'framer-motion';
import { Clock, MapPin, Globe, Server } from 'lucide-react';
import { useDashboardStore } from '@/stores/dashboardStore';
import CRTToggle from './CRTToggle';

export default function TopBar() {
  const {
    isConnected,
    isHandshaking,
    serverName,
    ping,
    inGameTime,
    inGameDay,
    botPosition,
    dimension,
  } = useDashboardStore();

  // Status orb color & animation
  let orbColor = '#B02E26'; // redstone-red (disconnected)
  let orbAnimation = 'animate-pulse-red';
  if (isConnected) {
    orbColor = '#5D8C4A'; // grass-green
    orbAnimation = '';
  } else if (isHandshaking) {
    orbColor = '#F9FF3E'; // amber
    orbAnimation = 'animate-pulse-amber';
  }

  // Dimension color
  const dimensionColor =
    dimension === 'overworld'
      ? 'text-grass-green'
      : dimension === 'nether'
        ? 'text-redstone-red'
        : 'text-mc-purple';

  // Ping badge color
  const pingColor =
    ping < 50
      ? 'bg-grass-green text-slate-950'
      : ping <= 150
        ? 'bg-amber text-slate-950'
        : 'bg-redstone-red text-white';

  return (
    <motion.header
      initial={{ y: -48, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.3, ease: 'easeOut', delay: 0.1 }}
      className="fixed top-0 left-0 right-0 h-12 bg-slate-900 border-b border-slate-700 flex items-center justify-between px-4 z-50"
    >
      {/* ── Left: Bot Status ── */}
      <div className="flex items-center gap-2">
        {/* Status orb */}
        <div
          className={`w-2.5 h-2.5 rounded-full ${orbAnimation}`}
          style={{ backgroundColor: orbColor, boxShadow: `0 0 8px ${orbColor}` }}
        />
        {/* Bot avatar */}
        <img
          src="./bot-avatar.png"
          alt="Bot"
          className="w-7 h-7 rounded"
          style={{ imageRendering: 'pixelated' }}
        />
        {/* Bot name */}
        <span className="font-mono text-mono-lg text-quartz-white tracking-tight">
          RL-BOT-01
        </span>
      </div>

      {/* ── Center: Time + Coords + Dimension ── */}
      <div className="flex items-center gap-6">
        {/* In-game time */}
        <div className="flex items-center gap-1.5">
          <Clock className="w-3.5 h-3.5 text-slate-400" />
          <span className="font-mono text-mono-base text-slate-300">{inGameTime}</span>
          <span className="font-mono text-mono-sm text-slate-500">(Day {inGameDay})</span>
        </div>

        {/* XYZ Coordinates */}
        <div className="flex items-center gap-1.5">
          <MapPin className="w-3.5 h-3.5 text-slate-400" />
          <span className="font-mono text-mono-sm bg-slate-800 rounded px-2 py-0.5 text-redstone-red">
            X: {botPosition.x}
          </span>
          <span className="font-mono text-mono-sm bg-slate-800 rounded px-2 py-0.5 text-grass-green">
            Y: {botPosition.y}
          </span>
          <span className="font-mono text-mono-sm bg-slate-800 rounded px-2 py-0.5 text-diamond-blue">
            Z: {botPosition.z}
          </span>
        </div>

        {/* Dimension */}
        <div className="flex items-center gap-1.5">
          <Globe className="w-3.5 h-3.5 text-slate-400" />
          <span className={`font-mono text-mono-sm ${dimensionColor}`}>
            minecraft:{dimension}
          </span>
        </div>
      </div>

      {/* ── Right: Server + CRT + Settings ── */}
      <div className="flex items-center gap-4">
        {/* Server info */}
        <div className="flex items-center gap-1.5">
          <Server className="w-3.5 h-3.5 text-slate-400" />
          <span className="font-mono text-mono-sm text-slate-300">{serverName}</span>
          <span className={`font-mono text-mono-xs rounded px-1.5 py-0.5 ${pingColor}`}>
            {ping}ms
          </span>
        </div>

        {/* CRT Toggle */}
        <CRTToggle />

        {/* Settings icon */}
        <button className="text-slate-500 hover:text-slate-300 hover:scale-110 transition-all duration-150">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" />
            <circle cx="12" cy="12" r="3" />
          </svg>
        </button>
      </div>
    </motion.header>
  );
}
