import { useState, useRef, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { X } from 'lucide-react';
import { useDashboardStore, type TerminalEntry } from '@/stores/dashboardStore';

// ─── Known commands for tab completion ─────────────────────────────────────

const KNOWN_COMMANDS = [
  '/gamemode', '/tp', '/give', '/kill', '/time', '/weather', '/clear', '/help', '/respawn',
  '!train', '!pause', '!resume', '!reset', '!explore', '!mine', '!goto', '!craft', '!status', '!save', '!load', '!spawn',
];

// ─── Helpers ─────────────────────────────────────────────────────────────────

function formatTimestamp(ts: number): string {
  const d = new Date(ts);
  const h = d.getHours().toString().padStart(2, '0');
  const m = d.getMinutes().toString().padStart(2, '0');
  const s = d.getSeconds().toString().padStart(2, '0');
  return `[${h}:${m}:${s}]`;
}

function getEntryColor(type: TerminalEntry['type']): string {
  switch (type) {
    case 'bot_response':
      return '#E3E3E5'; // quartz-white
    case 'rl_metric':
      return '#22d3ee'; // cyan
    case 'hyperparam':
      return '#22d3ee'; // cyan
    case 'error':
      return '#B02E26'; // redstone-red
    case 'success':
      return '#5D8C4A'; // grass-green
    case 'server_event':
      return '#F9FF3E'; // amber
    case 'command':
      return '#94a3b8'; // slate-400
    default:
      return '#E3E3E5';
  }
}

// ─── Component ───────────────────────────────────────────────────────────────

export default function CommandTerminal() {
  const store = useDashboardStore();
  const { terminalHistory, executeCommand, clearTerminal, isConnected } = store;

  // Local state
  const [input, setInput] = useState('');
  const [commandHistory, setCommandHistory] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [suggestionIndex, setSuggestionIndex] = useState(0);
  const [showSuggestions, setShowSuggestions] = useState(false);

  // Refs
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const wasAtBottom = useRef(true);

  // ─── Scroll tracking ─────────────────────────────────────────────────────

  const handleScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const threshold = 20;
    wasAtBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < threshold;
  }, []);

  // Auto-scroll to bottom when new entries arrive (if user was at bottom)
  useEffect(() => {
    if (wasAtBottom.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [terminalHistory.length]);

  // ─── Input handlers ──────────────────────────────────────────────────────

  const handleSubmit = () => {
    const trimmed = input.trim();
    if (!trimmed) return;

    // Save to command history
    setCommandHistory((prev) => {
      const next = [trimmed, ...prev];
      return next.slice(0, 50); // max 50
    });
    setHistoryIndex(-1);
    setShowSuggestions(false);

    executeCommand(trimmed);
    setInput('');
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    // Tab completion
    if (e.key === 'Tab') {
      e.preventDefault();
      if (!showSuggestions || suggestions.length === 0) {
        const matches = KNOWN_COMMANDS.filter((cmd) =>
          cmd.toLowerCase().startsWith(input.toLowerCase())
        );
        if (matches.length > 0) {
          setSuggestions(matches);
          setSuggestionIndex(0);
          setShowSuggestions(true);
          setInput(matches[0]);
        }
      } else {
        const nextIndex = (suggestionIndex + 1) % suggestions.length;
        setSuggestionIndex(nextIndex);
        setInput(suggestions[nextIndex]);
      }
      return;
    }

    // Hide suggestions on other keys
    if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') {
      setShowSuggestions(false);
    }

    // Enter
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSubmit();
      return;
    }

    // Arrow Up - previous command
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (commandHistory.length === 0) return;
      const nextIndex = historyIndex + 1;
      if (nextIndex < commandHistory.length) {
        setHistoryIndex(nextIndex);
        setInput(commandHistory[nextIndex]);
      }
      return;
    }

    // Arrow Down - next command
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (historyIndex <= 0) {
        setHistoryIndex(-1);
        setInput('');
      } else {
        const nextIndex = historyIndex - 1;
        setHistoryIndex(nextIndex);
        setInput(commandHistory[nextIndex]);
      }
      return;
    }
  };

  // ─── Suggestion click ────────────────────────────────────────────────────

  const handleSuggestionClick = (suggestion: string) => {
    setInput(suggestion);
    setShowSuggestions(false);
    inputRef.current?.focus();
  };

  // ─── Render ──────────────────────────────────────────────────────────────

  return (
    <div className="flex-1 bg-slate-900 border border-slate-700 rounded-md flex flex-col overflow-hidden min-h-0">
      {/* Panel Header */}
      <div
        className="h-8 bg-slate-800 border-b border-slate-700 flex items-center justify-between px-3 shrink-0"
        style={{ borderLeftWidth: 3, borderLeftColor: '#F9FF3E' }}
      >
        <div className="flex items-center gap-2">
          <span className="font-sans text-ui-sm text-slate-400 uppercase tracking-wider">
            TERMINAL
          </span>
          {/* Connection status dot */}
          <div
            className="w-1.5 h-1.5 rounded-full"
            style={{
              backgroundColor: isConnected ? '#5D8C4A' : '#B02E26',
              boxShadow: isConnected
                ? '0 0 6px #5D8C4A'
                : '0 0 6px #B02E26',
            }}
          />
        </div>
        <motion.button
          whileTap={{ scale: 0.9 }}
          onClick={clearTerminal}
          className="text-slate-500 hover:text-slate-300 transition-colors duration-150 p-0.5"
          title="Clear terminal"
        >
          <X className="w-3 h-3" />
        </motion.button>
      </div>

      {/* Terminal Output Area */}
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto bg-slate-950 p-2 font-mono text-mono-sm leading-relaxed scrollbar-thin min-h-0"
      >
        {terminalHistory.length === 0 ? (
          <div className="text-slate-600 italic">No messages...</div>
        ) : (
          terminalHistory.map((entry) => (
            <TerminalLine key={entry.id} entry={entry} />
          ))
        )}
      </div>

      {/* Input Bar */}
      <div className="shrink-0 h-9 bg-slate-900 border-t-2 border-slate-700 flex items-center px-2 relative">
        {/* Suggestion box */}
        {showSuggestions && suggestions.length > 0 && (
          <div className="absolute bottom-full left-0 right-0 mb-1 bg-slate-800 border border-slate-600 rounded overflow-hidden z-10">
            {suggestions.map((s, i) => (
              <div
                key={s}
                onClick={() => handleSuggestionClick(s)}
                className={`px-3 py-1 font-mono text-mono-sm cursor-pointer transition-colors ${
                  i === suggestionIndex
                    ? 'bg-slate-700 text-quartz-white'
                    : 'text-slate-300 hover:bg-slate-700'
                }`}
              >
                {s}
              </div>
            ))}
          </div>
        )}

        {/* Prompt */}
        <span className="text-amber font-mono text-mono-base mr-1 select-none">&gt;</span>

        {/* Input */}
        <input
          ref={inputRef}
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type command..."
          className="flex-1 bg-transparent text-quartz-white font-mono text-mono-sm placeholder-slate-600 focus:outline-none h-full"
          spellCheck={false}
          autoComplete="off"
        />
      </div>
    </div>
  );
}

// ─── Terminal Line sub-component ─────────────────────────────────────────────

function TerminalLine({ entry }: { entry: TerminalEntry }) {
  const color = getEntryColor(entry.type);
  const timestamp = formatTimestamp(entry.timestamp);

  // For commands, show the "> command" format without timestamp prefix
  if (entry.type === 'command') {
    return (
      <div className="py-0.5" style={{ color: '#94a3b8' }}>
        {entry.text}
      </div>
    );
  }

  return (
    <div className="py-0.5" style={{ color }}>
      <span className="text-slate-500 mr-1">{timestamp}</span>
      <span>{entry.text}</span>
    </div>
  );
}
