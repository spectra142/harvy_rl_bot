import { useState, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown } from 'lucide-react';
import { useDashboardStore } from '@/stores/dashboardStore';

// ─── Slider config ───────────────────────────────────────────────────────────
interface SliderConfig {
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
  defaultValue: number;
  color: string;
  format: (v: number) => string;
}

const sliders: SliderConfig[] = [
  {
    key: 'learningRate',
    label: 'Learning Rate',
    min: 0.0001,
    max: 0.01,
    step: 0.0001,
    defaultValue: 0.001,
    color: '#3C44AA',
    format: (v) => v.toFixed(4),
  },
  {
    key: 'gamma',
    label: 'Discount Factor (\u03B3)',
    min: 0.90,
    max: 0.999,
    step: 0.001,
    defaultValue: 0.99,
    color: '#5D8C4A',
    format: (v) => v.toFixed(3),
  },
  {
    key: 'batchSize',
    label: 'Batch Size',
    min: 16,
    max: 256,
    step: 16,
    defaultValue: 64,
    color: '#F9FF3E',
    format: (v) => v.toString(),
  },
  {
    key: 'replayBufferSize',
    label: 'Replay Buffer',
    min: 1000,
    max: 50000,
    step: 1000,
    defaultValue: 10000,
    color: '#22d3ee',
    format: (v) => v.toLocaleString(),
  },
  {
    key: 'targetNetworkUpdateFreq',
    label: 'Target Update Freq',
    min: 100,
    max: 5000,
    step: 100,
    defaultValue: 1000,
    color: '#B02E26',
    format: (v) => v.toLocaleString(),
  },
  {
    key: 'epsilonDecay',
    label: 'Epsilon Decay',
    min: 0.900,
    max: 0.9999,
    step: 0.0001,
    defaultValue: 0.995,
    color: '#8B5CF6',
    format: (v) => v.toFixed(4),
  },
];

// ─── Single Slider Row ───────────────────────────────────────────────────────
interface SliderRowProps {
  config: SliderConfig;
  value: number;
  onChange: (value: number) => void;
}

function SliderRow({ config, value, onChange }: SliderRowProps) {
  const { label, min, max, step, color, format } = config;
  const rangeRef = useRef<HTMLInputElement>(null);

  const percentage = ((value - min) / (max - min)) * 100;

  // Compute step precision for rounding
  const stepStr = step.toString();
  const decimalIndex = stepStr.indexOf('.');
  const precision = decimalIndex >= 0 ? stepStr.length - decimalIndex - 1 : 0;

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      let v = parseFloat(e.target.value);
      // Round to step precision
      v = parseFloat(v.toFixed(precision));
      onChange(v);
    },
    [onChange, precision]
  );

  return (
    <div className="flex items-center gap-3 py-1.5">
      {/* Label */}
      <span
        className="text-ui-xs text-slate-300 uppercase shrink-0"
        style={{ width: 96 }}
      >
        {label}
      </span>

      {/* Slider */}
      <div className="flex-1 relative flex items-center" style={{ height: 20 }}>
        <input
          ref={rangeRef}
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={handleChange}
          className="w-full h-1 appearance-none bg-transparent cursor-pointer slider-input"
          style={{
            // Use CSS custom properties for styling via inline approach
            ['--slider-percentage' as string]: `${percentage}%`,
            ['--slider-color' as string]: color,
          }}
        />
        {/* Custom track background */}
        <div
          className="absolute inset-x-0 top-1/2 -translate-y-1/2 h-1 rounded-full pointer-events-none"
          style={{ backgroundColor: '#334155' }}
        />
        {/* Custom filled track */}
        <div
          className="absolute left-0 top-1/2 -translate-y-1/2 h-1 rounded-full pointer-events-none"
          style={{
            width: `${percentage}%`,
            backgroundColor: color,
          }}
        />
        {/* Custom thumb */}
        <div
          className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 pointer-events-none"
          style={{
            left: `${percentage}%`,
            width: 14,
            height: 14,
            borderRadius: '50%',
            backgroundColor: '#e2e8f0',
            border: '2px solid #94a3b8',
            boxShadow: '0 1px 3px rgba(0,0,0,0.3)',
          }}
        />
      </div>

      {/* Value display */}
      <span
        className="font-mono text-mono-sm text-right shrink-0"
        style={{ width: 56, color: '#3C44AA' }}
      >
        {format(value)}
      </span>
    </div>
  );
}

// ─── Main component ──────────────────────────────────────────────────────────
export default function HyperparameterPanel() {
  const [collapsed, setCollapsed] = useState(false);

  const learningRate = useDashboardStore((s) => s.learningRate);
  const gamma = useDashboardStore((s) => s.gamma);
  const batchSize = useDashboardStore((s) => s.batchSize);
  const replayBufferSize = useDashboardStore((s) => s.replayBufferSize);
  const targetNetworkUpdateFreq = useDashboardStore(
    (s) => s.targetNetworkUpdateFreq
  );
  const epsilonDecay = useDashboardStore((s) => s.epsilonDecay);
  const setHyperparam = useDashboardStore((s) => s.setHyperparam);

  const values: Record<string, number> = {
    learningRate,
    gamma,
    batchSize,
    replayBufferSize,
    targetNetworkUpdateFreq,
    epsilonDecay,
  };

  const handleChange = useCallback(
    (key: string) => (value: number) => {
      setHyperparam(key, value);
    },
    [setHyperparam]
  );

  const handleReset = useCallback(() => {
    sliders.forEach((s) => {
      setHyperparam(s.key, s.defaultValue);
    });
  }, [setHyperparam]);

  return (
    <div className="bg-slate-900 border-b border-slate-700">
      {/* Header bar */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="w-full h-8 bg-slate-800 border-b border-slate-700 flex items-center justify-between px-3 cursor-pointer border-l-[3px] border-l-mc-purple hover:bg-slate-750 transition-colors"
      >
        <span className="text-ui-sm text-slate-400 uppercase tracking-wider">
          HYPERPARAMETERS
        </span>
        <motion.div
          animate={{ rotate: collapsed ? -90 : 0 }}
          transition={{ duration: 0.2 }}
        >
          <ChevronDown className="w-3.5 h-3.5 text-slate-500" />
        </motion.div>
      </button>

      {/* Collapsible body */}
      <AnimatePresence initial={false}>
        {!collapsed && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: 'easeOut' }}
            className="overflow-hidden"
          >
            <div className="px-3 py-2">
              {sliders.map((config) => (
                <SliderRow
                  key={config.key}
                  config={config}
                  value={values[config.key] ?? config.defaultValue}
                  onChange={handleChange(config.key)}
                />
              ))}

              {/* Action buttons */}
              <div className="flex gap-2 pt-2 border-t border-slate-800 mt-1">
                <button
                  onClick={handleReset}
                  className="flex-1 h-7 bg-slate-800 hover:bg-slate-700 border border-slate-600 rounded text-ui-xs text-slate-300 uppercase transition-colors active:scale-[0.98]"
                >
                  Reset Defaults
                </button>
                <button
                  onClick={() => {
                    // Save config to localStorage
                    const config: Record<string, number> = {};
                    sliders.forEach((s) => {
                      config[s.key] = values[s.key] ?? s.defaultValue;
                    });
                    localStorage.setItem(
                      'rl-bot-hyperparams',
                      JSON.stringify(config)
                    );
                    useDashboardStore
                      .getState()
                      .addTerminalEntry({
                        text: 'Hyperparameter config saved',
                        type: 'success',
                      });
                  }}
                  className="flex-1 h-7 bg-slate-800 hover:bg-slate-700 border border-slate-600 rounded text-ui-xs text-slate-300 uppercase transition-colors active:scale-[0.98]"
                >
                  Save Config
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
