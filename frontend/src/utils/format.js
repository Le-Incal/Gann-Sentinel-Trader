/**
 * Formatting utilities for the GST Dashboard
 */

/**
 * Format a number as currency
 */
export function formatCurrency(value, currency = 'USD') {
  if (value === null || value === undefined) return '$0.00'
  
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value)
}

/**
 * Format a number as percentage
 */
export function formatPercent(value) {
  if (value === null || value === undefined) return '0.00%'
  
  return new Intl.NumberFormat('en-US', {
    style: 'percent',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value / 100)
}

/**
 * Format a date string
 */
export function formatDate(dateString) {
  if (!dateString) return '-'
  
  const date = new Date(dateString)
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    hour12: true,
  }).format(date)
}

/**
 * Format a date as relative time (e.g., "5 minutes ago")
 */
export function formatTimeAgo(dateString) {
  if (!dateString) return '-'
  
  const date = new Date(dateString)
  const now = new Date()
  const seconds = Math.floor((now - date) / 1000)
  
  if (seconds < 60) return 'just now'
  
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  
  const days = Math.floor(hours / 24)
  if (days < 7) return `${days}d ago`
  
  return formatDate(dateString)
}

/**
 * Format large numbers with K/M/B suffixes
 */
export function formatCompact(value) {
  if (value === null || value === undefined) return '0'
  
  return new Intl.NumberFormat('en-US', {
    notation: 'compact',
    compactDisplay: 'short',
  }).format(value)
}

/**
 * Truncate text with ellipsis
 */
export function truncate(text, maxLength = 100) {
  if (!text) return ''
  if (text.length <= maxLength) return text
  return text.slice(0, maxLength) + '...'
}

/**
 * Format conviction score as display text
 */
export function formatConviction(score) {
  if (score === null || score === undefined) return '-'
  if (score >= 80) return `${score} (High)`
  if (score >= 60) return `${score} (Medium)`
  return `${score} (Low)`
}

/**
 * Get color class for conviction score
 */
export function getConvictionColor(score) {
  if (score === null || score === undefined) return 'text-gray-400'
  if (score >= 80) return 'text-gst-green'
  if (score >= 60) return 'text-gst-yellow'
  return 'text-gst-red'
}

/**
 * Format side (buy/sell) with color class
 */
export function getSideClass(side) {
  return side?.toLowerCase() === 'buy' ? 'text-gst-green' : 'text-gst-red'
}
