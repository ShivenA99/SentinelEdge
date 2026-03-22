import { useState } from 'react'
import {
  ChevronDown,
  ChevronUp,
  Phone,
  AlertTriangle,
  Shield,
  Clock,
} from 'lucide-react'

// -------- Types --------
export interface CallHistoryEntry {
  id: string
  callType: string
  callLabel: string
  duration: number         // seconds
  peakScore: number
  finalScore: number
  outcome: 'Blocked' | 'Dismissed' | 'Completed'
  timestamp: number        // Date.now()
  totalSentences: number
  topFeatures: string[]
}

interface CallHistoryProps {
  entries: CallHistoryEntry[]
}

// -------- Helpers --------
function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60)
  const secs = seconds % 60
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
}

function relativeTime(timestamp: number): string {
  const diff = Math.floor((Date.now() - timestamp) / 1000)
  if (diff < 5) return 'just now'
  if (diff < 60) return `${diff}s ago`
  const mins = Math.floor(diff / 60)
  if (mins < 60) return `${mins} minute${mins !== 1 ? 's' : ''} ago`
  const hours = Math.floor(mins / 60)
  return `${hours} hour${hours !== 1 ? 's' : ''} ago`
}

function getOutcomeStyle(outcome: string): { bg: string; text: string } {
  switch (outcome) {
    case 'Blocked':
      return { bg: 'bg-alert/10', text: 'text-alert' }
    case 'Dismissed':
      return { bg: 'bg-warning/10', text: 'text-warning' }
    case 'Completed':
    default:
      return { bg: 'bg-safe/10', text: 'text-safe' }
  }
}

function getScoreColor(score: number): string {
  if (score < 0.3) return 'text-safe'
  if (score < 0.5) return 'text-warning'
  if (score < 0.75) return 'text-orange-400'
  return 'text-alert'
}

function getScoreBgColor(score: number): string {
  if (score < 0.3) return 'bg-safe/10'
  if (score < 0.5) return 'bg-warning/10'
  if (score < 0.75) return 'bg-orange-500/10'
  return 'bg-alert/10'
}

function getCallIcon(callType: string) {
  switch (callType) {
    case 'irs_scam':
    case 'tech_support':
    case 'bank_fraud':
      return <AlertTriangle className="w-4 h-4" />
    case 'legitimate':
      return <Shield className="w-4 h-4" />
    default:
      return <Phone className="w-4 h-4" />
  }
}

function getCallIconColor(callType: string): string {
  switch (callType) {
    case 'irs_scam':
      return 'text-alert bg-alert/10'
    case 'tech_support':
      return 'text-orange-400 bg-orange-500/10'
    case 'bank_fraud':
      return 'text-warning bg-warning/10'
    case 'legitimate':
      return 'text-safe bg-safe/10'
    default:
      return 'text-gray-400 bg-gray-700/30'
  }
}

// -------- Component --------
export default function CallHistory({ entries }: CallHistoryProps) {
  const [isExpanded, setIsExpanded] = useState(true)
  const [selectedId, setSelectedId] = useState<string | null>(null)

  const selectedEntry = entries.find(e => e.id === selectedId) ?? null

  return (
    <div className="glass-card overflow-hidden">
      {/* Collapsible header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between px-4 py-4 hover:bg-white/[0.02] transition-colors sm:px-6"
      >
        <div className="flex items-center gap-2.5">
          <Clock className="w-4 h-4 text-brand-teal" />
          <h3 className="text-sm font-semibold text-gray-200">Recent Calls</h3>
          {entries.length > 0 && (
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-brand-teal/10 text-brand-teal font-medium tabular-nums">
              {entries.length}
            </span>
          )}
        </div>
        {isExpanded ? (
          <ChevronUp className="w-4 h-4 text-gray-500" />
        ) : (
          <ChevronDown className="w-4 h-4 text-gray-500" />
        )}
      </button>

      {/* Body */}
      {isExpanded && (
        <div className="border-t border-dark-border/30">
          {entries.length === 0 ? (
            /* Empty state */
            <div className="flex flex-col items-center justify-center py-10 px-6">
              <Phone className="w-8 h-8 text-gray-700 mb-3" />
              <p className="text-sm text-gray-500 font-medium">No calls yet.</p>
              <p className="text-xs text-gray-600 mt-1">Start a sample call to see results here.</p>
            </div>
          ) : (
            <div className="divide-y divide-dark-border/20">
              {entries.map((entry) => (
                <div key={entry.id}>
                  {/* Row */}
                  <button
                    onClick={() => setSelectedId(selectedId === entry.id ? null : entry.id)}
                    className="w-full flex flex-col items-start gap-3 px-4 py-3.5 text-left hover:bg-white/[0.02] transition-colors sm:px-6 lg:flex-row lg:items-center lg:gap-4"
                  >
                    {/* Call type icon */}
                    <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${getCallIconColor(entry.callType)}`}>
                      {getCallIcon(entry.callType)}
                    </div>

                    {/* Call info */}
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-gray-200 truncate">
                        {entry.callLabel}
                      </p>
                      <p className="text-[10px] text-gray-500 mt-0.5">
                        {relativeTime(entry.timestamp)}
                      </p>
                    </div>

                    {/* Duration */}
                    <div className="flex items-center gap-1.5 text-xs text-gray-400 lg:flex-shrink-0">
                      <Clock className="w-3 h-3" />
                      <span className="font-mono tabular-nums">{formatDuration(entry.duration)}</span>
                    </div>

                    {/* Final score badge */}
                    <div className={`flex-shrink-0 px-2.5 py-1 rounded-lg ${getScoreBgColor(entry.finalScore)}`}>
                      <span className={`text-xs font-bold tabular-nums ${getScoreColor(entry.finalScore)}`}>
                        {(entry.finalScore * 100).toFixed(0)}%
                      </span>
                    </div>

                    {/* Outcome badge */}
                    <div>
                      <span className={`text-[10px] font-semibold px-2 py-1 rounded-full ${getOutcomeStyle(entry.outcome).bg} ${getOutcomeStyle(entry.outcome).text}`}>
                        {entry.outcome}
                      </span>
                    </div>

                    {/* Expand indicator */}
                    {selectedId === entry.id ? (
                      <ChevronUp className="w-3.5 h-3.5 text-gray-600 flex-shrink-0" />
                    ) : (
                      <ChevronDown className="w-3.5 h-3.5 text-gray-600 flex-shrink-0" />
                    )}
                  </button>

                  {/* Expanded detail */}
                  {selectedId === entry.id && (
                    <div className="px-4 pb-4 animate-fade-in sm:px-6">
                      <div className="rounded-xl border border-dark-border/20 bg-dark-bg/50 p-4 space-y-3 lg:ml-12">
                        {/* Summary row */}
                        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
                          <div>
                            <p className="text-[10px] text-gray-600 uppercase tracking-wider mb-1">Total Sentences</p>
                            <p className="text-sm font-bold text-gray-200 tabular-nums">{entry.totalSentences}</p>
                          </div>
                          <div>
                            <p className="text-[10px] text-gray-600 uppercase tracking-wider mb-1">Peak Score</p>
                            <p className={`text-sm font-bold tabular-nums ${getScoreColor(entry.peakScore)}`}>
                              {(entry.peakScore * 100).toFixed(1)}%
                            </p>
                          </div>
                          <div>
                            <p className="text-[10px] text-gray-600 uppercase tracking-wider mb-1">Final Score</p>
                            <p className={`text-sm font-bold tabular-nums ${getScoreColor(entry.finalScore)}`}>
                              {(entry.finalScore * 100).toFixed(1)}%
                            </p>
                          </div>
                        </div>

                        {/* Top features */}
                        {entry.topFeatures.length > 0 && (
                          <div>
                            <p className="text-[10px] text-gray-600 uppercase tracking-wider mb-1.5">Top Features</p>
                            <div className="flex flex-wrap gap-1.5">
                              {entry.topFeatures.map((feature) => (
                                <span key={feature} className="text-[10px] px-2 py-0.5 bg-brand-teal/10 text-brand-teal rounded-full">
                                  {feature.replace(/_/g, ' ')}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
