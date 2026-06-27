import { useMemo } from 'react';
import { Heart, Drumstick } from 'lucide-react';
import { useDashboardStore } from '@/stores/dashboardStore';

const MAX_HEARTS = 10;
const MAX_HUNGER = 10;

export default function POVFeed() {
  const botHealth = useDashboardStore((s) => s.botHealth);
  const botHunger = useDashboardStore((s) => s.botHunger);
  const botXP = useDashboardStore((s) => s.botXP);
  const botEquipped = useDashboardStore((s) => s.botEquipped);
  const crtMode = useDashboardStore((s) => s.crtMode);

  // Calculate health hearts (0-20 health maps to 0-10 hearts)
  const fullHearts = Math.floor(botHealth / 2);
  const halfHeart = botHealth % 2 >= 1;
  const emptyHearts = MAX_HEARTS - fullHearts - (halfHeart ? 1 : 0);

  // Calculate hunger icons (0-20 hunger maps to 0-10 icons)
  const fullHunger = Math.floor(botHunger / 2);
  const halfHunger = botHunger % 2 >= 1;
  const emptyHunger = MAX_HUNGER - fullHunger - (halfHunger ? 1 : 0);

  // XP level and progress
  const xpLevel = Math.floor(botXP);
  const xpProgress = (botXP % 1) * 100;

  // Health color transitions from red to amber to green based on health
  const healthColor = useMemo(() => {
    if (botHealth <= 6) return '#B02E26';
    if (botHealth <= 12) return '#D97706';
    return '#5D8C4A';
  }, [botHealth]);

  return (
    <div className="flex flex-col h-[35%]">
      {/* Panel Header */}
      <div className="h-7 bg-slate-800 border-b border-slate-700 flex items-center px-3 border-l-[3px] border-l-grass-green">
        <span className="font-ui text-ui-sm text-slate-400 uppercase tracking-wider">
          POV FEED
        </span>
        <span className="ml-auto font-mono text-mono-xs text-slate-500">
          FPS: 20
        </span>
      </div>

      {/* Feed Container */}
      <div className="flex-1 relative overflow-hidden">
        {/* POV stream */}
        <iframe
          src={import.meta.env.VITE_POV_URL || 'http://localhost:3007'}
          className="absolute inset-0 w-full h-full border-0"
          title="Bot POV"
          allow="autoplay"
        />

        {/* Dark vignette overlay */}
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background:
              'radial-gradient(ellipse at center, transparent 40%, rgba(0,0,0,0.4) 100%)',
          }}
        />

        {/* CRT scanline overlay for POV */}
        {crtMode && (
          <div className="absolute inset-0 pointer-events-none z-20 pov-scanline" />
        )}

        {/* HUD Overlay */}
        <div
          className={`absolute inset-0 pointer-events-none z-10 ${
            crtMode ? 'pov-crt-glow' : ''
          }`}
        >
          {/* Bot name tag - top center */}
          <div className="absolute top-2 left-1/2 -translate-x-1/2 bg-slate-800/80 rounded-full px-3 py-0.5">
            <span className="font-mono text-mono-xs text-slate-300">RL-Bot</span>
          </div>

          {/* Crosshair - centered */}
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2">
            {/* Top arm */}
            <div
              className="absolute left-1/2 -translate-x-1/2 bg-white/80"
              style={{
                width: 2,
                height: 12,
                bottom: 6,
              }}
            />
            {/* Bottom arm */}
            <div
              className="absolute left-1/2 -translate-x-1/2 bg-white/80"
              style={{
                width: 2,
                height: 12,
                top: 6,
              }}
            />
            {/* Left arm */}
            <div
              className="absolute top-1/2 -translate-y-1/2 bg-white/80"
              style={{
                width: 12,
                height: 2,
                right: 6,
              }}
            />
            {/* Right arm */}
            <div
              className="absolute top-1/2 -translate-y-1/2 bg-white/80"
              style={{
                width: 12,
                height: 2,
                left: 6,
              }}
            />
          </div>

          {/* Health bar - top-right */}
          <div className="absolute top-2.5 right-2.5 flex flex-col items-end gap-1">
            <div className="flex gap-px">
              {/* Full hearts */}
              {Array.from({ length: fullHearts }).map((_, i) => (
                <Heart
                  key={`hf-${i}`}
                  className="w-3 h-3"
                  fill={healthColor}
                  stroke={healthColor}
                />
              ))}
              {/* Half heart */}
              {halfHeart ? (
                <div className="relative w-3 h-3">
                  <Heart
                    className="absolute inset-0 w-3 h-3"
                    fill="#1e293b"
                    stroke="#1e293b"
                  />
                  <div className="absolute inset-0 overflow-hidden w-1.5">
                    <Heart
                      className="w-3 h-3"
                      fill={healthColor}
                      stroke={healthColor}
                    />
                  </div>
                </div>
              ) : null}
              {/* Empty hearts */}
              {Array.from({ length: emptyHearts }).map((_, i) => (
                <Heart
                  key={`he-${i}`}
                  className="w-3 h-3"
                  fill="#1e293b"
                  stroke="#475569"
                />
              ))}
            </div>

            {/* Hunger bar - below health */}
            <div className="flex gap-px">
              {Array.from({ length: fullHunger }).map((_, i) => (
                <Drumstick
                  key={`hg-${i}`}
                  className="w-3 h-3"
                  fill="#8B6914"
                  stroke="#8B6914"
                />
              ))}
              {halfHunger ? (
                <div className="relative w-3 h-3">
                  <Drumstick
                    className="absolute inset-0 w-3 h-3"
                    fill="#1e293b"
                    stroke="#1e293b"
                  />
                  <div className="absolute inset-0 overflow-hidden w-1.5">
                    <Drumstick
                      className="w-3 h-3"
                      fill="#8B6914"
                      stroke="#8B6914"
                    />
                  </div>
                </div>
              ) : null}
              {Array.from({ length: emptyHunger }).map((_, i) => (
                <Drumstick
                  key={`he-${i}`}
                  className="w-3 h-3"
                  fill="#1e293b"
                  stroke="#475569"
                />
              ))}
            </div>

            {/* XP Level text */}
            <span
              className="font-mono text-mono-sm"
              style={{ color: '#7FFF00' }}
            >
              Lv {xpLevel}
            </span>
          </div>

          {/* XP bar - thin line above hotbar area */}
          <div className="absolute bottom-10 left-1/2 -translate-x-1/2 w-48 h-1 bg-slate-800/60 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-300"
              style={{
                width: `${xpProgress}%`,
                backgroundColor: '#7FFF00',
              }}
            />
          </div>

          {/* Equipped item - bottom-right */}
          <div className="absolute bottom-2.5 right-2.5 bg-slate-950/70 rounded px-2 py-1 flex items-center gap-1.5">
            <div className="w-5 h-5 bg-slate-700 rounded-sm flex items-center justify-center">
              <span className="text-mono-xs text-slate-400">?</span>
            </div>
            <span className="font-mono text-mono-sm text-quartz-white">
              {botEquipped}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
