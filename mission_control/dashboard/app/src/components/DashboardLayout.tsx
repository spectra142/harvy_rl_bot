import { type ReactNode, useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useDashboardStore } from '@/stores/dashboardStore';
import TopBar from './TopBar';
import BottomBar from './BottomBar';

interface DashboardLayoutProps {
  children: ReactNode;
}

export default function DashboardLayout({ children }: DashboardLayoutProps) {
  const crtMode = useDashboardStore((s) => s.crtMode);

  return (
    <div className="relative w-screen h-screen overflow-hidden bg-slate-950">
      {/* Top Bar */}
      <TopBar />

      {/* Main Content Area */}
      <div
        className="flex w-full"
        style={{ height: 'calc(100vh - 48px - 56px)', marginTop: 48 }}
      >
        {children}
      </div>

      {/* Bottom Bar */}
      <BottomBar />

      {/* CRT Overlay */}
      {crtMode && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.4 }}
          className="fixed inset-0 pointer-events-none z-[100] crt-active"
          aria-hidden="true"
        />
      )}

      {/* Emergency Stop Flash Overlay */}
      <EmergencyStopFlash />
    </div>
  );
}

function EmergencyStopFlash() {
  const [flash, setFlash] = useState(false);

  useEffect(() => {
    const handler = () => {
      setFlash(true);
      setTimeout(() => setFlash(false), 600);
    };
    window.addEventListener('emergency-stop' as keyof WindowEventMap, handler as EventListener);
    return () => window.removeEventListener('emergency-stop' as keyof WindowEventMap, handler as EventListener);
  }, []);

  if (!flash) return null;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: [0, 0.2, 0] }}
      transition={{ duration: 0.6 }}
      className="fixed inset-0 pointer-events-none z-[70] bg-redstone-red"
      aria-hidden="true"
    />
  );
}
