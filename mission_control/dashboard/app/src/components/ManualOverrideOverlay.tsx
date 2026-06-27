import { motion, AnimatePresence } from 'framer-motion';
import { useEffect, useCallback } from 'react';

interface ManualOverrideOverlayProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function ManualOverrideOverlay({ isOpen, onClose }: ManualOverrideOverlayProps) {
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    },
    [onClose],
  );

  useEffect(() => {
    if (isOpen) {
      window.addEventListener('keydown', handleKeyDown);
    }
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen, handleKeyDown]);

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 bg-[rgba(2,6,23,0.8)] z-[50]"
            onClick={onClose}
          />

          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.25, type: 'spring', stiffness: 300, damping: 25 }}
            className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[400px] bg-slate-900/95 border-2 border-diamond-blue rounded-lg z-[60] p-6"
          >
            {/* Title */}
            <h2 className="text-center text-ui-lg text-diamond-blue font-semibold mb-6 tracking-wide">
              MANUAL OVERRIDE
            </h2>

            {/* WASD Keys */}
            <div className="flex flex-col items-center gap-1 mb-6">
              <KeyButton label="W" />
              <div className="flex gap-1">
                <KeyButton label="A" />
                <KeyButton label="S" />
                <KeyButton label="D" />
              </div>
            </div>

            {/* Mouse Look Area */}
            <div className="flex justify-center mb-6">
              <div className="w-[120px] h-[120px] rounded-full border-2 border-slate-600 flex items-center justify-center relative cursor-crosshair">
                <div className="absolute w-2 h-2 bg-slate-400 rounded-full" />
                <span className="text-ui-xs text-slate-500 mt-16">MOUSE LOOK</span>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex justify-center gap-3 mb-6">
              <ActionButton label="LMB Attack" color="#B02E26" />
              <ActionButton label="RMB Place" color="#3C44AA" />
              <ActionButton label="Space Jump" color="#5D8C4A" />
            </div>

            {/* Close hint */}
            <p className="text-center text-ui-xs text-slate-500">
              Press <kbd className="bg-slate-800 px-1.5 py-0.5 rounded text-slate-300 font-mono text-mono-xs">ESC</kbd> to close
            </p>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}

function KeyButton({ label }: { label: string }) {
  return (
    <motion.button
      whileTap={{ scale: 0.9, y: 2 }}
      className="w-12 h-12 bg-slate-800 border border-slate-600 rounded-md flex items-center justify-center text-quartz-white font-mono text-mono-xl font-bold shadow-[0_4px_0_#334155] active:shadow-none active:translate-y-1 transition-all"
    >
      {label}
    </motion.button>
  );
}

function ActionButton({ label, color }: { label: string; color: string }) {
  return (
    <motion.button
      whileTap={{ scale: 0.95 }}
      className="px-3 py-2 rounded text-slate-950 text-ui-xs font-semibold uppercase"
      style={{ backgroundColor: color }}
    >
      {label}
    </motion.button>
  );
}
