import { motion } from 'framer-motion';
import { Plug, Unplug } from 'lucide-react';
import { useDashboardStore } from '@/stores/dashboardStore';

const MC_VERSIONS = [
  '1.8', '1.9', '1.10', '1.11', '1.12', '1.13', '1.14',
  '1.15', '1.16', '1.17', '1.18', '1.19', '1.20', '1.20+',
];

export default function ServerConnection() {
  const store = useDashboardStore();

  const handleToggleConnection = () => {
    if (store.isConnected) {
      store.disconnect();
    } else {
      const wsUrl = `ws://${store.ipAddress}:${store.port}/ws`;
      store.connect(wsUrl);
    }
  };

  // Button state
  let buttonLabel = 'CONNECT';
  let buttonBg = '#5D8C4A';
  let buttonIcon = <Plug className="w-4 h-4" />;
  let isPulsing = false;

  if (store.isHandshaking) {
    buttonLabel = 'CONNECTING...';
    buttonBg = '#F9FF3E';
    buttonIcon = <Plug className="w-4 h-4" />;
    isPulsing = true;
  } else if (store.isConnected) {
    buttonLabel = 'DISCONNECT';
    buttonBg = '#B02E26';
    buttonIcon = <Unplug className="w-4 h-4" />;
  }

  return (
    <div className="bg-slate-900 border border-slate-700 rounded-md flex flex-col overflow-hidden">
      {/* Panel Header */}
      <div
        className="h-8 bg-slate-800 border-b border-slate-700 flex items-center px-3"
        style={{ borderLeftWidth: 3, borderLeftColor: '#3C44AA' }}
      >
        <span className="font-sans text-ui-sm text-slate-400 uppercase tracking-wider">
          SERVER CONNECTION
        </span>
      </div>

      {/* Form Fields */}
      <div className="p-3 flex flex-col gap-2">
        {/* Server Name */}
        <div className="flex flex-col gap-1">
          <label className="font-sans text-ui-xs text-slate-500 uppercase tracking-wider">
            Server Name
          </label>
          <input
            type="text"
            placeholder="My RL Server"
            value={store.serverName}
            onChange={(e) => useDashboardStore.setState({ serverName: e.target.value })}
            className="bg-slate-800 border border-slate-600 rounded px-3 py-2 font-mono text-mono-sm text-quartz-white placeholder-slate-500 focus:outline-none focus:border-diamond-blue focus:shadow-[0_0_0_2px_rgba(60,68,170,0.2)] transition-all h-9"
          />
        </div>

        {/* Bridge IP */}
        <div className="flex flex-col gap-1">
          <label className="font-sans text-ui-xs text-slate-500 uppercase tracking-wider">
            Bridge IP
          </label>
          <input
            type="text"
            placeholder="localhost"
            value={store.ipAddress}
            onChange={(e) => useDashboardStore.setState({ ipAddress: e.target.value })}
            className="bg-slate-800 border border-slate-600 rounded px-3 py-2 font-mono text-mono-sm text-quartz-white placeholder-slate-500 focus:outline-none focus:border-diamond-blue focus:shadow-[0_0_0_2px_rgba(60,68,170,0.2)] transition-all h-9"
          />
        </div>

        {/* Bridge Port */}
        <div className="flex flex-col gap-1">
          <label className="font-sans text-ui-xs text-slate-500 uppercase tracking-wider">
            Bridge Port
          </label>
          <input
            type="text"
            placeholder="9877"
            value={store.port}
            onChange={(e) => useDashboardStore.setState({ port: e.target.value })}
            className="bg-slate-800 border border-slate-600 rounded px-3 py-2 font-mono text-mono-sm text-quartz-white placeholder-slate-500 focus:outline-none focus:border-diamond-blue focus:shadow-[0_0_0_2px_rgba(60,68,170,0.2)] transition-all h-9 w-24"
          />
        </div>

        {/* Minecraft Version */}
        <div className="flex flex-col gap-1">
          <label className="font-sans text-ui-xs text-slate-500 uppercase tracking-wider">
            Minecraft Version
          </label>
          <select
            value={store.mcVersion}
            onChange={(e) => useDashboardStore.setState({ mcVersion: e.target.value })}
            className="bg-slate-800 border border-slate-600 rounded px-3 py-2 font-mono text-mono-sm text-quartz-white focus:outline-none focus:border-diamond-blue focus:shadow-[0_0_0_2px_rgba(60,68,170,0.2)] transition-all h-9 appearance-none cursor-pointer"
            style={{ backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%2364748b' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m6 9 6 6 6-6'/%3E%3C/svg%3E")`, backgroundRepeat: 'no-repeat', backgroundPosition: 'right 12px center' }}
          >
            {MC_VERSIONS.map((v) => (
              <option key={v} value={v} className="bg-slate-800 text-quartz-white">
                {v}
              </option>
            ))}
          </select>
        </div>

        {/* Connect/Disconnect Button */}
        <motion.button
          whileTap={{ scale: 0.97 }}
          onClick={handleToggleConnection}
          animate={isPulsing ? { opacity: [0.4, 1, 0.4] } : { opacity: 1 }}
          transition={isPulsing ? { duration: 1.2, repeat: Infinity, ease: 'easeInOut' } : {}}
          className="flex items-center justify-center gap-2 w-full h-9 rounded font-sans text-ui-base uppercase font-semibold transition-all duration-150 mt-1"
          style={{
            backgroundColor: buttonBg,
            color: store.isHandshaking ? '#0f172a' : store.isConnected ? '#ffffff' : '#0f172a',
            boxShadow: store.isConnected
              ? '0 0 12px rgba(176,46,38,0.3)'
              : '0 0 12px rgba(93,140,74,0.3)',
          }}
        >
          {buttonIcon}
          {buttonLabel}
        </motion.button>
      </div>
    </div>
  );
}
