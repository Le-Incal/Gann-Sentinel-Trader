/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'gst': {
          'dark': '#0a0f1a',
          'darker': '#060a12',
          'card': '#111827',
          'border': '#1f2937',
          'accent': '#3b82f6',
          'accent-hover': '#2563eb',
          'green': '#10b981',
          'red': '#ef4444',
          'yellow': '#f59e0b',
          'purple': '#8b5cf6',
        }
      },
      fontFamily: {
        'mono': ['JetBrains Mono', 'Fira Code', 'monospace'],
      }
    },
  },
  plugins: [],
}
