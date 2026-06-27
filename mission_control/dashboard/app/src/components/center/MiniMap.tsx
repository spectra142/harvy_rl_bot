import { useRef, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { useDashboardStore } from '@/stores/dashboardStore';

const MAP_RADIUS = 200; // blocks visible around bot
const BOT_ARROW_SIZE = 8;

export default function MiniMap() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const botPosition = useDashboardStore((s) => s.botPosition);
  const botAngle = useDashboardStore((s) => s.botAngle);
  const entities = useDashboardStore((s) => s.entities);
  const exploredChunks = useDashboardStore((s) => s.exploredChunks);

  // Draw the mini-map
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const size = canvas.width;
    const centerX = size / 2;
    const centerY = size / 2;
    const scale = size / (MAP_RADIUS * 2);

    // Clear
    ctx.clearRect(0, 0, size, size);

    // Draw terrain background
    const terrainImg = new Image();
    terrainImg.src = './mini-map-terrain.jpg';
    if (terrainImg.complete) {
      ctx.drawImage(terrainImg, 0, 0, size, size);
    } else {
      // Fallback background
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, size, size);
    }

    // Grid overlay - chunk boundaries every 50px
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= size; i += 50) {
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i, size);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, i);
      ctx.lineTo(size, i);
      ctx.stroke();
    }

    // Explored chunks
    exploredChunks.forEach((chunkKey) => {
      const [cx, cz] = chunkKey.split(',').map(Number);
      const screenX = centerX + (cx * 16 - botPosition.x) * scale;
      const screenY = centerY + (cz * 16 - botPosition.z) * scale;
      if (
        screenX > -16 * scale &&
        screenX < size + 16 * scale &&
        screenY > -16 * scale &&
        screenY < size + 16 * scale
      ) {
        ctx.fillStyle = 'rgba(100,100,100,0.3)';
        ctx.fillRect(screenX, screenY, 16 * scale, 16 * scale);
      }
    });

    // Entity dots
    entities.forEach((entity) => {
      const ex = centerX + (entity.x - botPosition.x) * scale;
      const ez = centerY + (entity.z - botPosition.z) * scale;

      if (ex < 0 || ex > size || ez < 0 || ez > size) return;

      let color = '#8B5CF6';
      let radius = 3;
      let isPlayer = false;

      switch (entity.type) {
        case 'hostile':
          color = '#B02E26';
          radius = 4;
          break;
        case 'passive':
          color = '#5D8C4A';
          radius = 3;
          break;
        case 'player':
          color = '#3C44AA';
          radius = 5;
          isPlayer = true;
          break;
        case 'item':
          color = '#F9FF3E';
          radius = 2;
          break;
        case 'poi':
          color = '#F9B233';
          radius = 3;
          break;
      }

      ctx.beginPath();
      ctx.arc(ex, ez, radius, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();

      if (isPlayer) {
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    });

    // Bot arrow at center (always centered)
    const angleRad = (botAngle * Math.PI) / 180;
    ctx.save();
    ctx.translate(centerX, centerY);
    ctx.rotate(angleRad);

    // Arrow shape
    ctx.beginPath();
    ctx.moveTo(0, -BOT_ARROW_SIZE);
    ctx.lineTo(-BOT_ARROW_SIZE * 0.7, BOT_ARROW_SIZE * 0.7);
    ctx.lineTo(BOT_ARROW_SIZE * 0.7, BOT_ARROW_SIZE * 0.7);
    ctx.closePath();
    ctx.fillStyle = '#3C44AA';
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    ctx.restore();
  }, [botPosition, botAngle, entities, exploredChunks]);

  // Set up canvas size and animation loop
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const resize = () => {
      const rect = container.getBoundingClientRect();
      const size = Math.floor(Math.min(rect.width, rect.height));
      canvas.width = size;
      canvas.height = size;
      draw();
    };

    resize();
    window.addEventListener('resize', resize);

    // Update at 2Hz
    const interval = setInterval(draw, 500);

    return () => {
      window.removeEventListener('resize', resize);
      clearInterval(interval);
    };
  }, [draw]);

  return (
    <div className="flex flex-col h-[20%]">
      {/* Panel Header */}
      <div className="h-7 bg-slate-800 border-b border-slate-700 flex items-center px-3 border-l-[3px] border-l-diamond-blue">
        <span className="font-ui text-ui-sm text-slate-400 uppercase tracking-wider">
          MINI-MAP
        </span>
      </div>

      {/* Map Container */}
      <div className="flex-1 p-2 flex items-center justify-center bg-slate-900">
        <motion.div
          ref={containerRef}
          className="relative aspect-square h-full rounded-md overflow-hidden border border-slate-700"
          style={{
            boxShadow: 'inset 0 2px 8px rgba(0,0,0,0.4)',
          }}
          animate={{
            scale: [1, 1.005, 1],
          }}
          transition={{
            duration: 1.5,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        >
          <canvas
            ref={canvasRef}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
            }}
          />

          {/* Coordinates overlay */}
          <div className="absolute bottom-1 left-1.5 pointer-events-none">
            <span className="font-mono text-mono-xs text-slate-400">
              X: {Math.round(botPosition.x)} Z: {Math.round(botPosition.z)}
            </span>
          </div>

          {/* Compass overlay */}
          <div className="absolute top-1 right-1.5 pointer-events-none flex flex-col items-center">
            <span className="font-mono text-mono-xs text-slate-400 font-bold">N</span>
            <div className="flex gap-1">
              <span className="font-mono text-mono-xs text-slate-500">W</span>
              <span className="font-mono text-mono-xs text-slate-500">E</span>
            </div>
            <span className="font-mono text-mono-xs text-slate-500">S</span>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
