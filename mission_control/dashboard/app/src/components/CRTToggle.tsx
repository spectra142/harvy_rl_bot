import { Monitor } from 'lucide-react';
import { motion } from 'framer-motion';
import { useDashboardStore } from '@/stores/dashboardStore';

export default function CRTToggle() {
  const { crtMode, toggleCrtMode } = useDashboardStore();

  return (
    <div className="flex items-center gap-1.5">
      <Monitor className="w-3.5 h-3.5 text-slate-400" />
      <span className="text-ui-xs text-slate-500 uppercase tracking-wider">CRT</span>
      <button
        onClick={toggleCrtMode}
        className={`
          relative w-5 h-2.5 rounded-full transition-colors duration-150
          ${crtMode ? 'bg-grass-green/60' : 'bg-slate-700'}
        `}
        aria-label="Toggle CRT mode"
      >
        <motion.div
          className="absolute top-0.5 left-0.5 w-1.5 h-1.5 rounded-full"
          style={{
            backgroundColor: crtMode ? '#5D8C4A' : '#64748b',
          }}
          animate={{
            x: crtMode ? 8 : 0,
          }}
          transition={{ duration: 0.15, ease: 'easeInOut' }}
        />
      </button>
    </div>
  );
}
