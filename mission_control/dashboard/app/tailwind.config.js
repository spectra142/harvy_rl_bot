/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ["class"],
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive) / <alpha-value>)",
          foreground: "hsl(var(--destructive-foreground) / <alpha-value>)",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        sidebar: {
          DEFAULT: "hsl(var(--sidebar-background))",
          foreground: "hsl(var(--sidebar-foreground))",
          primary: "hsl(var(--sidebar-primary))",
          "primary-foreground": "hsl(var(--sidebar-primary-foreground))",
          accent: "hsl(var(--sidebar-accent))",
          "accent-foreground": "hsl(var(--sidebar-accent-foreground))",
          border: "hsl(var(--sidebar-border))",
          ring: "hsl(var(--sidebar-ring))",
        },
        'grass-green': '#5D8C4A',
        'redstone-red': '#B02E26',
        'diamond-blue': '#3C44AA',
        'amber': '#F9FF3E',
        'quartz-white': '#E3E3E5',
        'cyan': '#22d3ee',
        'gold': '#F9B233',
        'mc-purple': '#8B5CF6',
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      fontSize: {
        'mono-xs': ['11px', { lineHeight: '1.3', fontWeight: '400', letterSpacing: '0.02em' }],
        'mono-sm': ['12px', { lineHeight: '1.3', fontWeight: '400', letterSpacing: '0.01em' }],
        'mono-base': ['13px', { lineHeight: '1.3', fontWeight: '500', letterSpacing: '0' }],
        'mono-lg': ['14px', { lineHeight: '1.3', fontWeight: '600', letterSpacing: '0' }],
        'mono-xl': ['16px', { lineHeight: '1.3', fontWeight: '700', letterSpacing: '-0.01em' }],
        'mono-2xl': ['20px', { lineHeight: '1.3', fontWeight: '700', letterSpacing: '-0.02em' }],
        'ui-xs': ['10px', { lineHeight: '1.4', fontWeight: '500', letterSpacing: '0.05em' }],
        'ui-sm': ['11px', { lineHeight: '1.4', fontWeight: '500', letterSpacing: '0.03em' }],
        'ui-base': ['12px', { lineHeight: '1.4', fontWeight: '600', letterSpacing: '0.02em' }],
        'ui-lg': ['14px', { lineHeight: '1.4', fontWeight: '600', letterSpacing: '0.01em' }],
        'ui-xl': ['16px', { lineHeight: '1.4', fontWeight: '700', letterSpacing: '0' }],
        'ui-2xl': ['20px', { lineHeight: '1.4', fontWeight: '700', letterSpacing: '-0.01em' }],
      },
      spacing: {
        'space-0': '0px',
        'space-1': '2px',
        'space-2': '4px',
        'space-3': '6px',
        'space-4': '8px',
        'space-5': '10px',
        'space-6': '12px',
        'space-8': '16px',
        'space-10': '20px',
        'space-12': '24px',
        'space-16': '32px',
      },
      borderRadius: {
        xl: "calc(var(--radius) + 4px)",
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
        xs: "calc(var(--radius) - 6px)",
      },
      boxShadow: {
        xs: "0 1px 2px 0 rgb(0 0 0 / 0.05)",
      },
      keyframes: {
        "accordion-down": {
          from: { height: "0" },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: "0" },
        },
        "caret-blink": {
          "0%,70%,100%": { opacity: "1" },
          "20%,50%": { opacity: "0" },
        },
        "pulse-green": {
          "0%, 100%": { boxShadow: "0 0 4px #5D8C4A", transform: "scale(1)" },
          "50%": { boxShadow: "0 0 12px #5D8C4A", transform: "scale(1.15)" },
        },
        "pulse-amber": {
          "0%, 100%": { boxShadow: "0 0 4px #F9FF3E", transform: "scale(1)" },
          "50%": { boxShadow: "0 0 12px #F9FF3E", transform: "scale(1.3)" },
        },
        "pulse-red": {
          "0%, 100%": { boxShadow: "0 0 4px #B02E26", transform: "scale(1)" },
          "50%": { boxShadow: "0 0 12px #B02E26", transform: "scale(1.15)" },
        },
        "scanline": {
          "0%": { transform: "translateY(0)" },
          "100%": { transform: "translateY(4px)" },
        },
        "crt-flicker": {
          "0%, 100%": { opacity: "0.01" },
          "50%": { opacity: "0.03" },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
        "caret-blink": "caret-blink 1.25s ease-out infinite",
        "pulse-green": "pulse-green 1.2s ease-in-out infinite",
        "pulse-amber": "pulse-amber 0.8s ease-in-out infinite",
        "pulse-red": "pulse-red 1.5s ease-in-out infinite",
        "scanline": "scanline 0.1s linear infinite",
        "crt-flicker": "crt-flicker 50ms infinite",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
}
