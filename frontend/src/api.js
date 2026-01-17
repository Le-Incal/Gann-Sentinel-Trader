/**
 * API client for Gann Sentinel Trader
 * 
 * For production, set VITE_API_URL to your Railway backend URL
 * For development, the Vite proxy handles /api routes
 */

const API_BASE = import.meta.env.VITE_API_URL || '';

/**
 * Fetch wrapper with error handling
 */
async function fetchAPI(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;
  
  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error(`API request failed: ${endpoint}`, error);
    throw error;
  }
}

/**
 * Public API endpoints (no authentication required)
 */
export const api = {
  // Health check
  getHealth: () => fetchAPI('/health'),

  // Dashboard data
  getDashboard: () => fetchAPI('/api/public/dashboard'),

  // Portfolio
  getPortfolio: () => fetchAPI('/api/public/portfolio'),
  getPortfolioHistory: (days = 30) => fetchAPI(`/api/public/portfolio/history?days=${days}`),

  // Positions
  getPositions: () => fetchAPI('/api/public/positions'),

  // Trades
  getTrades: (limit = 50) => fetchAPI(`/api/public/trades?limit=${limit}`),
  getPendingTrades: () => fetchAPI('/api/public/trades/pending'),

  // Signals
  getSignals: (limit = 50) => fetchAPI(`/api/public/signals?limit=${limit}`),

  // Scan Cycles
  getScanCycles: (limit = 20) => fetchAPI(`/api/public/scan_cycles?limit=${limit}`),
  getScanCycle: (cycleId) => fetchAPI(`/api/public/scan_cycles/${cycleId}`),

  // Debates
  getDebates: (limit = 20) => fetchAPI(`/api/public/debates?limit=${limit}`),
  getDebate: (sessionId) => fetchAPI(`/api/public/debates/${sessionId}`),

  // System status
  getSystemStatus: () => fetchAPI('/api/public/system'),

  // Cost tracking
  getCostSummary: (days = 7) => fetchAPI(`/api/public/costs?days=${days}`),
};

export default api;
