import { motion } from 'framer-motion';
import DashboardLayout from '@/components/DashboardLayout';
import { ServerConnection, ServerStatus, QuickActions, CommandTerminal } from '@/components/left';
import { POVFeed, MiniMap, RLCharts, ActionDistribution } from '@/components/center';
import { InventoryGrid, HyperparameterPanel, DeathEventLog } from '@/components/right';

export default function App() {
  return (
    <DashboardLayout>
      {/* Left Column — 30% — Connection & Command Zone */}
      <motion.div
        data-zone="left"
        initial={{ x: -20, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ duration: 0.3, ease: 'easeOut', delay: 0.2 }}
        className="w-[30%] h-full border-r border-slate-700 overflow-hidden flex flex-col"
      >
        <ServerConnection />
        <ServerStatus />
        <QuickActions />
        <CommandTerminal />
      </motion.div>

      {/* Center Column — 40% — Intelligence & Visualization Zone */}
      <motion.div
        data-zone="center"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.3, ease: 'easeOut', delay: 0.3 }}
        className="flex-1 h-full overflow-hidden flex flex-col"
      >
        <POVFeed />
        <MiniMap />
        <RLCharts />
        <ActionDistribution />
      </motion.div>

      {/* Right Column — 30% — Inventory & Hyperparameter Zone */}
      <motion.div
        data-zone="right"
        initial={{ x: 20, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ duration: 0.3, ease: 'easeOut', delay: 0.4 }}
        className="w-[30%] h-full border-l border-slate-700 overflow-hidden flex flex-col"
      >
        <InventoryGrid />
        <HyperparameterPanel />
        <DeathEventLog />
      </motion.div>
    </DashboardLayout>
  );
}
