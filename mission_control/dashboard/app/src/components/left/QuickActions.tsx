import { useState } from 'react';
import { motion } from 'framer-motion';
import { Pause, Play, RefreshCw, Home, Trash2, Eye, Square } from 'lucide-react';
import { useDashboardStore } from '@/stores/dashboardStore';

interface ActionChip {
  id: string;
  label: string;
  icon: React.ReactNode;
  command: string;
}

const ACTIONS: ActionChip[] = [
  { id: 'start', label: 'START', icon: <Play className="w-4 h-4" />, command: '!train' },
  { id: 'pause', label: 'PAUSE', icon: <Pause className="w-4 h-4" />, command: '!pause' },
  { id: 'stop', label: 'STOP', icon: <Square className="w-4 h-4" />, command: '!stop' },
  { id: 'reset', label: 'RESET', icon: <RefreshCw className="w-4 h-4" />, command: '!reset' },
  { id: 'respawn', label: 'RESPAWN', icon: <RefreshCw className="w-4 h-4" />, command: '/respawn' },
  { id: 'spawn', label: 'SPAWN', icon: <Home className="w-4 h-4" />, command: '!spawn' },
  { id: 'clear', label: 'CLEAR', icon: <Trash2 className="w-4 h-4" />, command: '/clear' },
  { id: 'nv', label: 'NV', icon: <Eye className="w-4 h-4" />, command: '/effect give @p minecraft:night_vision 999' },
];

export default function QuickActions() {
  const { executeCommand } = useDashboardStore();
  const [activeFlash, setActiveFlash] = useState<string | null>(null);

  const handleClick = (action: ActionChip) => {
    executeCommand(action.command);

    // Flash feedback
    setActiveFlash(action.id);
    setTimeout(() => setActiveFlash(null), 200);
  };

  return (
    <div className="flex items-center gap-1.5 px-3 py-2 bg-slate-900 border-y border-slate-700 flex-wrap">
      {ACTIONS.map((action) => (
        <motion.button
          key={action.id}
          whileTap={{ scale: 0.97 }}
          whileHover={{ scale: 1.02 }}
          onClick={() => handleClick(action)}
          className={`
            flex items-center gap-1.5 px-3 py-1.5 rounded
            font-mono text-mono-sm
            transition-colors duration-150
            ${activeFlash === action.id
              ? action.id === 'pause'
                ? 'bg-amber text-slate-950'
                : action.id === 'stop'
                  ? 'bg-redstone-red text-white'
                  : ['start', 'reset', 'nv', 'respawn'].includes(action.id)
                    ? 'bg-diamond-blue text-white'
                    : 'bg-slate-700 text-slate-200'
              : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
            }
          `}
          style={{
            border: '1px solid #334155',
          }}
        >
          {action.icon}
          <span className="uppercase">{action.label}</span>
        </motion.button>
      ))}
    </div>
  );
}
