import { useState, useMemo } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  type ChartOptions,
  type ChartData,
} from 'chart.js';
import { useDashboardStore } from '@/stores/dashboardStore';
import type { DataPoint } from '@/stores/dashboardStore';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

// ─── Shared dark theme chart defaults ────────────────────────────────────────

const commonOptions: ChartOptions<'line'> = {
  responsive: true,
  maintainAspectRatio: false,
  animation: {
    duration: 300,
  },
  interaction: {
    mode: 'index',
    intersect: false,
  },
  plugins: {
    legend: {
      display: false,
    },
    tooltip: {
      backgroundColor: '#1e293b',
      titleColor: '#94a3b8',
      bodyColor: '#E3E3E5',
      borderColor: '#475569',
      borderWidth: 1,
      cornerRadius: 4,
      padding: 8,
      titleFont: {
        family: '"JetBrains Mono", monospace',
        size: 10,
      },
      bodyFont: {
        family: '"JetBrains Mono", monospace',
        size: 11,
      },
    },
  },
  scales: {
    x: {
      grid: {
        color: 'rgba(255,255,255,0.05)',
        lineWidth: 1,
      },
      ticks: {
        color: '#94a3b8',
        font: {
          family: '"JetBrains Mono", monospace',
          size: 9,
        },
        maxTicksLimit: 8,
      },
      border: {
        display: false,
      },
    },
    y: {
      grid: {
        color: 'rgba(255,255,255,0.05)',
        lineWidth: 1,
      },
      ticks: {
        color: '#94a3b8',
        font: {
          family: '"JetBrains Mono", monospace',
          size: 9,
        },
        maxTicksLimit: 5,
      },
      border: {
        display: false,
      },
    },
  },
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

function computeMovingAverage(data: DataPoint[], window: number): DataPoint[] {
  const result: DataPoint[] = [];
  for (let i = 0; i < data.length; i++) {
    const start = Math.max(0, i - window + 1);
    const slice = data.slice(start, i + 1);
    const avg = slice.reduce((sum, d) => sum + d.y, 0) / slice.length;
    result.push({ x: data[i].x, y: avg });
  }
  return result;
}

function useLastN(data: DataPoint[], n: number): DataPoint[] {
  return useMemo(() => {
    if (data.length <= n) return data;
    return data.slice(data.length - n);
  }, [data, n]);
}

// ─── Episode Reward Chart ────────────────────────────────────────────────────

function EpisodeRewardChart() {
  const rewardHistory = useDashboardStore((s) => s.rewardHistory);
  const [showAvg, setShowAvg] = useState(false);
  const data = useLastN(rewardHistory, 50);

  const avgData = useMemo(() => {
    if (!showAvg || data.length < 2) return [];
    return computeMovingAverage(data, 10);
  }, [data, showAvg]);

  const chartData: ChartData<'line'> = useMemo(
    () => ({
      labels: data.map((d) => `E${d.x}`),
      datasets: [
        {
          label: 'Reward',
          data: data.map((d) => d.y),
          borderColor: '#5D8C4A',
          backgroundColor: 'rgba(93,140,74,0.1)',
          borderWidth: 2,
          fill: true,
          pointRadius: 0,
          pointHoverRadius: 3,
          tension: 0.2,
        },
        ...(showAvg && avgData.length > 0
          ? [
              {
                label: 'AVG',
                data: avgData.map((d) => d.y),
                borderColor: '#ffffff',
                backgroundColor: 'transparent',
                borderWidth: 1,
                borderDash: [4, 4] as [number, number],
                fill: false,
                pointRadius: 0,
                pointHoverRadius: 0,
                tension: 0.3,
              },
            ]
          : []),
      ],
    }),
    [data, avgData, showAvg]
  );

  return (
    <div className="flex-1 flex flex-col min-h-0 px-2 py-1">
      <div className="flex items-center justify-between mb-0.5">
        <span className="font-ui text-ui-xs text-slate-400 uppercase tracking-wider">
          EPISODE REWARD
        </span>
        <button
          onClick={() => setShowAvg((v) => !v)}
          className={`font-mono text-mono-xs px-1.5 py-0.5 rounded transition-colors duration-150 ${
            showAvg
              ? 'bg-slate-600 text-white'
              : 'bg-slate-800 text-slate-500 hover:bg-slate-700 hover:text-slate-300'
          }`}
        >
          AVG
        </button>
      </div>
      <div className="flex-1 min-h-0">
        <Line data={chartData} options={commonOptions} />
      </div>
    </div>
  );
}

// ─── Loss Decay Chart ────────────────────────────────────────────────────────

function LossDecayChart() {
  const lossHistory = useDashboardStore((s) => s.lossHistory);
  const [logScale, setLogScale] = useState(false);
  const data = useLastN(lossHistory, 50);

  const chartData: ChartData<'line'> = useMemo(
    () => ({
      labels: data.map((d) => `S${d.x}`),
      datasets: [
        {
          label: 'Loss',
          data: data.map((d) => d.y),
          borderColor: '#B02E26',
          backgroundColor: 'transparent',
          borderWidth: 2,
          fill: false,
          pointRadius: 0,
          pointHoverRadius: 3,
          tension: 0.2,
        },
      ],
    }),
    [data]
  );

  const options: ChartOptions<'line'> = useMemo(
    () => ({
      ...commonOptions,
      scales: {
        ...commonOptions.scales,
        y: {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any -- Chart.js scale options are a complex union; casting is required to override the y-axis type while preserving common options.
          ...(commonOptions.scales as any).y,
          type: logScale ? 'logarithmic' : 'linear',
          min: logScale ? undefined : 0,
        },
      },
    }),
    [logScale]
  );

  return (
    <div className="flex-1 flex flex-col min-h-0 px-2 py-1 border-t border-slate-800">
      <div className="flex items-center justify-between mb-0.5">
        <span className="font-ui text-ui-xs text-slate-400 uppercase tracking-wider">
          LOSS DECAY
        </span>
        <button
          onClick={() => setLogScale((v) => !v)}
          className={`font-mono text-mono-xs px-1.5 py-0.5 rounded transition-colors duration-150 ${
            logScale
              ? 'bg-slate-600 text-white'
              : 'bg-slate-800 text-slate-500 hover:bg-slate-700 hover:text-slate-300'
          }`}
        >
          LOG
        </button>
      </div>
      <div className="flex-1 min-h-0">
        <Line data={chartData} options={options} />
      </div>
    </div>
  );
}

// ─── Epsilon Chart ───────────────────────────────────────────────────────────

function EpsilonChart() {
  const epsilonHistory = useDashboardStore((s) => s.epsilonHistory);
  const data = useLastN(epsilonHistory, 50);

  const chartData: ChartData<'line'> = useMemo(
    () => ({
      labels: data.map((d) => `S${d.x}`),
      datasets: [
        {
          label: 'Epsilon',
          data: data.map((d) => d.y),
          borderColor: '#3C44AA',
          backgroundColor: 'rgba(60,68,170,0.1)',
          borderWidth: 2,
          fill: true,
          pointRadius: 0,
          pointHoverRadius: 3,
          tension: 0.2,
        },
      ],
    }),
    [data]
  );

  const options: ChartOptions<'line'> = useMemo(
    () => ({
      ...commonOptions,
      scales: {
        ...commonOptions.scales,
        y: {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any -- Chart.js scale options are a complex union; casting is required to override the y-axis type while preserving common options.
          ...(commonOptions.scales as any).y,
          min: 0,
          max: 1,
        },
      },
    }),
    []
  );

  return (
    <div className="flex-1 flex flex-col min-h-0 px-2 py-1 border-t border-slate-800">
      <div className="flex items-center justify-between mb-0.5">
        <span className="font-ui text-ui-xs text-slate-400 uppercase tracking-wider">
          EPSILON (EXPLORATION)
        </span>
      </div>
      <div className="flex-1 min-h-0">
        <Line data={chartData} options={options} />
      </div>
    </div>
  );
}

// ─── Main RL Charts Panel ────────────────────────────────────────────────────

export default function RLCharts() {
  return (
    <div className="flex flex-col h-[30%] border-t border-slate-700">
      {/* Panel Header */}
      <div className="h-7 bg-slate-800 border-b border-slate-700 flex items-center px-3 border-l-[3px] border-l-cyan">
        <span className="font-ui text-ui-sm text-slate-400 uppercase tracking-wider">
          RL METRICS
        </span>
      </div>

      {/* Charts */}
      <div className="flex-1 flex flex-col min-h-0 bg-slate-900">
        <EpisodeRewardChart />
        <LossDecayChart />
        <EpsilonChart />
      </div>
    </div>
  );
}
