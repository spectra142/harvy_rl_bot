import { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Skull, Gem, Trophy } from 'lucide-react';
import { useDashboardStore } from '@/stores/dashboardStore';
import type { GameEvent } from '@/stores/dashboardStore';

// ─── Filter config ───────────────────────────────────────────────────────────
const filters = [
  { key: 'all' as const, label: 'ALL' },
  { key: 'death' as const, label: 'DEATHS' },
  { key: 'discovery' as const, label: 'DISCOVERIES' },
  { key: 'milestone' as const, label: 'MILESTONES' },
] as const;

type FilterKey = (typeof filters)[number]['key'];

// ─── Icon map ────────────────────────────────────────────────────────────────
const eventIconMap = {
  death: { Icon: Skull, color: '#B02E26', borderColor: '#B02E26' },
  discovery: { Icon: Gem, color: '#3C44AA', borderColor: '#3C44AA' },
  milestone: { Icon: Trophy, color: '#F9B233', borderColor: '#F9B233' },
};

// ─── Timestamp formatter ─────────────────────────────────────────────────────
function formatTimestamp(ts: number): string {
  const d = new Date(ts);
  const h = d.getHours().toString().padStart(2, '0');
  const m = d.getMinutes().toString().padStart(2, '0');
  const s = d.getSeconds().toString().padStart(2, '0');
  return `${h}:${m}:${s}`;
}

// ─── Event Entry ─────────────────────────────────────────────────────────────
interface EventEntryProps {
  event: GameEvent;
}

function EventEntry({ event }: EventEntryProps) {
  const { Icon, color, borderColor } = eventIconMap[event.type];
  const ts = formatTimestamp(event.timestamp);

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: -16 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.25, ease: 'easeOut' }}
      className="relative flex items-start gap-2.5 px-3 py-2 border-b border-slate-800 hover:bg-slate-800/50 transition-colors cursor-default group"
    >
      {/* Left colored border */}
      <div
        className="absolute left-0 top-1.5 bottom-1.5 w-[3px] rounded-full"
        style={{ backgroundColor: borderColor }}
      />

      {/* Icon */}
      <Icon
        className="shrink-0 mt-0.5"
        style={{ width: 14, height: 14, color }}
      />

      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="font-mono text-mono-sm text-slate-200 leading-tight">
          {event.title}
        </div>
        {event.description && (
          <div className="font-mono text-mono-xs text-slate-500 mt-0.5">
            {event.description}
          </div>
        )}
        {event.position && (
          <div className="font-mono text-mono-xs text-slate-600 mt-0.5">
            at {event.position.x}, {event.position.y}, {event.position.z}
          </div>
        )}
      </div>

      {/* Timestamp */}
      <span className="font-mono text-mono-xs text-slate-500 shrink-0 self-start pt-0.5">
        {ts}
      </span>
    </motion.div>
  );
}

// ─── Mock events initializer ─────────────────────────────────────────────────
function getMockEvents(): GameEvent[] {
  const now = Date.now();
  return [
    {
      id: 'evt-init-1',
      type: 'death',
      title: 'Died to Zombie',
      description: 'at 142, 64, -89',
      position: { x: 142, y: 64, z: -89 },
      timestamp: now - 120000,
    },
    {
      id: 'evt-init-2',
      type: 'discovery',
      title: 'Found Diamond Ore',
      description: 'at 138, 12, -95',
      position: { x: 138, y: 12, z: -95 },
      timestamp: now - 300000,
    },
    {
      id: 'evt-init-3',
      type: 'milestone',
      title: 'Episode 47 completed',
      description: 'reward: 1,247',
      timestamp: now - 480000,
    },
    {
      id: 'evt-init-4',
      type: 'death',
      title: 'Died to Lava',
      description: 'at 200, 32, -50',
      position: { x: 200, y: 32, z: -50 },
      timestamp: now - 720000,
    },
    {
      id: 'evt-init-5',
      type: 'discovery',
      title: 'Found Iron Ore',
      description: 'at 145, 45, -80',
      position: { x: 145, y: 45, z: -80 },
      timestamp: now - 900000,
    },
    {
      id: 'evt-init-6',
      type: 'milestone',
      title: 'Episode 50 completed',
      description: 'reward: 2,104',
      timestamp: now - 1200000,
    },
    {
      id: 'evt-init-7',
      type: 'discovery',
      title: 'Crafted Diamond Pickaxe',
      description: '',
      timestamp: now - 1500000,
    },
    {
      id: 'evt-init-8',
      type: 'death',
      title: 'Died to Fall Damage',
      description: 'at 300, 64, -200',
      position: { x: 300, y: 64, z: -200 },
      timestamp: now - 1800000,
    },
  ];
}

// ─── Main component ──────────────────────────────────────────────────────────
export default function DeathEventLog() {
  const [activeFilter, setActiveFilter] = useState<FilterKey>('all');
  const events = useDashboardStore((s) => s.events);
  const addEvent = useDashboardStore((s) => s.addEvent);
  const listRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const prevEventsLength = useRef(events.length);

  // Initialize with mock events if empty
  useEffect(() => {
    const store = useDashboardStore.getState();
    if (store.events.length === 0) {
      const mockEvents = getMockEvents();
      mockEvents.forEach((e) =>
        addEvent({
          type: e.type,
          title: e.title,
          description: e.description,
          position: e.position,
        })
      );
    }
  }, [addEvent]);

  // Filtered events
  const filteredEvents =
    activeFilter === 'all'
      ? events
      : events.filter((e) => e.type === activeFilter);

  // Auto-scroll to bottom on new events
  useEffect(() => {
    if (events.length > prevEventsLength.current && autoScroll && listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
    prevEventsLength.current = events.length;
  }, [events.length, autoScroll]);

  // Handle scroll: pause auto-scroll if user scrolls up
  const handleScroll = useCallback(() => {
    if (!listRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = listRef.current;
    const nearBottom = scrollHeight - scrollTop - clientHeight < 20;
    setAutoScroll(nearBottom);
  }, []);

  return (
    <div className="h-full flex flex-col bg-slate-900">
      {/* Header bar */}
      <div className="h-8 bg-slate-800 border-b border-slate-700 flex items-center px-3 border-l-[3px] border-l-redstone-red">
        <span className="text-ui-sm text-slate-400 uppercase tracking-wider">
          EVENT LOG
        </span>
      </div>

      {/* Filter tabs */}
      <div className="flex gap-1 px-3 py-1.5 border-b border-slate-800">
        {filters.map((f) => {
          const isActive = activeFilter === f.key;
          return (
            <button
              key={f.key}
              onClick={() => setActiveFilter(f.key)}
              className={`
                px-2.5 py-1 rounded text-ui-xs uppercase tracking-wider
                transition-colors duration-150 cursor-pointer
                ${
                  isActive
                    ? 'bg-slate-700 text-quartz-white'
                    : 'text-slate-400 hover:text-slate-300 bg-transparent'
                }
              `}
              style={isActive ? { borderBottom: '2px solid #3C44AA' } : undefined}
            >
              {f.label}
            </button>
          );
        })}
      </div>

      {/* Event list */}
      <div
        ref={listRef}
        className="flex-1 overflow-y-auto"
        onScroll={handleScroll}
      >
        <AnimatePresence mode="popLayout" initial={false}>
          {filteredEvents.length === 0 ? (
            <div className="flex items-center justify-center h-20">
              <span className="text-ui-xs text-slate-600 uppercase">
                No events
              </span>
            </div>
          ) : (
            filteredEvents.map((event) => (
              <EventEntry key={event.id} event={event} />
            ))
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
