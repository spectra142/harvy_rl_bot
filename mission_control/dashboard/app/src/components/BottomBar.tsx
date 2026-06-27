import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Play,
  Pause,
  RotateCcw,
  Gamepad2,
  Save,
  FolderOpen,
  Octagon,
} from 'lucide-react';
import { useDashboardStore } from '@/stores/dashboardStore';
import ManualOverrideOverlay from './ManualOverrideOverlay';

export default function BottomBar() {
  const {
    isTraining,
    isPaused,
    startTraining,
    pauseTraining,
    resumeTraining,
    resetEnvironment,
    emergencyStopArmed,
    armEmergencyStop,
  } = useDashboardStore();

  const [manualOverrideOpen, setManualOverrideOpen] = useState(false);
  const [, setStopClicks] = useState(0);

  const handleStartTraining = () => {
    if (isTraining && !isPaused) {
      // already training, do nothing
      return;
    }
    if (isPaused) {
      resumeTraining();
    } else {
      startTraining();
    }
  };

  const handlePause = () => {
    if (!isTraining || isPaused) return;
    pauseTraining();
  };

  const handleEmergencyStop = () => {
    if (!emergencyStopArmed) {
      armEmergencyStop();
      setStopClicks(1);
    } else {
      // Second click - execute emergency stop
      setStopClicks(0);
      // Dispatch full-screen red flash via a custom event
      window.dispatchEvent(new CustomEvent('emergency-stop'));
      // Add terminal entry via store
      useDashboardStore.getState().addTerminalEntry({
        text: '!!! EMERGENCY STOP TRIGGERED !!!',
        type: 'error',
      });
      // Reset training state
      useDashboardStore.setState({
        isTraining: false,
        isPaused: false,
      });
    }
  };

  const handleReset = () => {
    resetEnvironment();
  };

  const handleSaveCheckpoint = () => {
    useDashboardStore.getState().addTerminalEntry({
      text: 'Checkpoint saved successfully',
      type: 'success',
    });
  };

  const handleLoadCheckpoint = () => {
    useDashboardStore.getState().addTerminalEntry({
      text: 'Checkpoint loaded',
      type: 'success',
    });
  };

  // Training button label
  const trainingLabel = isTraining && !isPaused ? 'TRAINING...' : 'START TRAINING';

  return (
    <>
      <motion.nav
        initial={{ y: 56, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.3, ease: 'easeOut', delay: 0.5 }}
        className="fixed bottom-0 left-0 right-0 h-14 bg-slate-900 border-t-2 border-slate-700 flex items-center justify-between px-4 z-50"
      >
        {/* Start Training */}
        <motion.button
          whileTap={{ scale: 0.97 }}
          onClick={handleStartTraining}
          className={`
            flex items-center gap-2 px-4 h-9 rounded font-sans text-ui-base uppercase font-semibold
            transition-all duration-150 w-[160px] justify-center
            ${isTraining && !isPaused
              ? 'bg-grass-green/80 text-slate-950 shadow-[0_0_16px_rgba(93,140,74,0.6)]'
              : 'bg-grass-green text-slate-950 shadow-[0_0_16px_rgba(93,140,74,0.4)] hover:shadow-[0_0_20px_rgba(93,140,74,0.6)] animate-[pulse-green_2s_ease-in-out_infinite]'
            }
          `}
        >
          <Play className="w-4 h-4" />
          {trainingLabel}
        </motion.button>

        {/* Pause */}
        <motion.button
          whileTap={{ scale: 0.97 }}
          onClick={handlePause}
          disabled={!isTraining || isPaused}
          className={`
            flex items-center gap-2 px-4 h-9 rounded font-sans text-ui-base uppercase font-semibold
            transition-all duration-150 w-[100px] justify-center
            ${isPaused || !isTraining
              ? 'bg-slate-800 text-slate-600 border border-slate-700 cursor-not-allowed'
              : 'bg-amber text-slate-950 hover:shadow-[0_0_12px_rgba(249,255,62,0.4)]'
            }
          `}
        >
          <Pause className="w-4 h-4" />
          PAUSE
        </motion.button>

        {/* Reset Environment */}
        <motion.button
          whileTap={{ scale: 0.97 }}
          onClick={handleReset}
          className="flex items-center gap-2 px-4 h-9 rounded font-sans text-ui-base uppercase font-semibold bg-[#D97706] text-slate-950 hover:bg-[#F59E0B] transition-all duration-150 w-[120px] justify-center"
        >
          <RotateCcw className="w-4 h-4" />
          RESET ENV
        </motion.button>

        {/* Manual Override */}
        <motion.button
          whileTap={{ scale: 0.97 }}
          onClick={() => setManualOverrideOpen(true)}
          className={`
            flex items-center gap-2 px-4 h-9 rounded font-sans text-ui-base uppercase font-semibold
            transition-all duration-150 w-[110px] justify-center
            ${manualOverrideOpen
              ? 'bg-diamond-blue text-white border border-white'
              : 'bg-diamond-blue text-white hover:shadow-[0_0_12px_rgba(60,68,170,0.5)]'
            }
          `}
        >
          <Gamepad2 className="w-4 h-4" />
          MANUAL
        </motion.button>

        {/* Save Checkpoint */}
        <motion.button
          whileTap={{ scale: 0.97 }}
          onClick={handleSaveCheckpoint}
          className="flex items-center gap-2 px-4 h-9 rounded font-sans text-ui-base uppercase font-semibold bg-slate-800 text-slate-300 border border-slate-600 hover:bg-slate-700 transition-all duration-150 w-[90px] justify-center"
        >
          <Save className="w-4 h-4" />
          SAVE
        </motion.button>

        {/* Load Checkpoint */}
        <motion.button
          whileTap={{ scale: 0.97 }}
          onClick={handleLoadCheckpoint}
          className="flex items-center gap-2 px-4 h-9 rounded font-sans text-ui-base uppercase font-semibold bg-slate-800 text-slate-300 border border-slate-600 hover:bg-slate-700 transition-all duration-150 w-[90px] justify-center"
        >
          <FolderOpen className="w-4 h-4" />
          LOAD
        </motion.button>

        {/* Emergency Stop */}
        <div className="flex flex-col items-center">
          <motion.button
            whileTap={{ scale: 0.95 }}
            onClick={handleEmergencyStop}
            className={`
              flex items-center gap-2 px-4 h-9 rounded font-sans text-ui-base uppercase font-bold
              transition-all duration-150 w-[100px] justify-center
              ${emergencyStopArmed
                ? 'bg-redstone-red text-white border-2 border-white shadow-[0_0_20px_rgba(176,46,38,0.8)] animate-pulse'
                : 'bg-redstone-red text-white shadow-[0_0_20px_rgba(176,46,38,0.5)] hover:shadow-[0_0_28px_rgba(176,46,38,0.7)]'
              }
            `}
          >
            <Octagon className="w-4 h-4" />
            STOP
          </motion.button>
          {emergencyStopArmed && (
            <motion.span
              initial={{ opacity: 0, y: -4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="text-ui-xs text-redstone-red mt-1 absolute -bottom-5"
            >
              Click again to confirm
            </motion.span>
          )}
        </div>
      </motion.nav>

      {/* Manual Override Overlay */}
      <ManualOverrideOverlay
        isOpen={manualOverrideOpen}
        onClose={() => setManualOverrideOpen(false)}
      />
    </>
  );
}
