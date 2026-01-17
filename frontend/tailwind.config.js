/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Light backgrounds - clean and modern
        'gst': {
          'light': '#ffffff',       // Pure white
          'bg': '#f8fafc',          // Soft gray background
          'card': '#ffffff',        // Card background
          'border': '#e2e8f0',      // Light borders
          'hover': '#f1f5f9',       // Hover state
        },
        // Primary: Robinhood lime green
        'lime': {
          50: '#f0fdf0',
          100: '#dcfce7',
          200: '#bbf7d0',
          300: '#86efac',
          400: '#4ade80',
          500: '#00c853',           // Primary Robinhood green
          600: '#00a843',           // Hover state
          700: '#15803d',
          800: '#166534',
          900: '#14532d',
        },
        // Secondary: Schwab blue
        'schwab': {
          50: '#eff6ff',
          100: '#dbeafe',
          200: '#bfdbfe',
          300: '#93c5fd',
          400: '#60a5fa',
          500: '#00a3e0',           // Schwab light blue
          600: '#0284c7',
          700: '#0369a1',
          800: '#075985',
          900: '#0c4a6e',
        },
        // Accent colors
        'profit': '#00c853',        // Lime green for gains
        'loss': '#ef4444',          // Red for losses
        'warning': '#f59e0b',       // Amber orange
        'info': '#00a3e0',          // Schwab blue
      },
      fontFamily: {
        'sans': ['Inter', 'SF Pro Display', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
        'mono': ['SF Mono', 'JetBrains Mono', 'Fira Code', 'monospace'],
      },
      backgroundImage: {
        'gradient-brand': 'linear-gradient(135deg, #00c853 0%, #00a3e0 100%)',
      },
      boxShadow: {
        'card': '0 1px 3px rgba(0, 0, 0, 0.08), 0 1px 2px rgba(0, 0, 0, 0.06)',
        'card-hover': '0 4px 12px rgba(0, 0, 0, 0.1)',
        'glow-green': '0 0 20px rgba(0, 200, 83, 0.2)',
        'glow-blue': '0 0 20px rgba(0, 163, 224, 0.2)',
      }
    },
  },
  plugins: [],
}
