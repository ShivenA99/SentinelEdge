import { BarChart3 } from 'lucide-react'

interface FeatureBreakdownProps {
  features: Record<string, number>
}

function getBarColor(value: number): string {
  if (value < 0.3) return 'bg-brand-teal'
  if (value < 0.5) return 'bg-warning'
  if (value < 0.75) return 'bg-orange-500'
  return 'bg-alert'
}

function getBarGlow(value: number): string {
  if (value < 0.3) return ''
  if (value < 0.75) return ''
  return 'shadow-sm shadow-alert/20'
}

function formatFeatureName(name: string): string {
  return name
    .replace(/_/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase())
}

export default function FeatureBreakdown({ features }: FeatureBreakdownProps) {
  // Sort features by value descending, filter out zero values
  const sortedFeatures = Object.entries(features)
    .filter(([, value]) => value > 0)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 8) // Show top 8

  return (
    <div>
      {/* Header */}
      <div className="flex items-center gap-2 mb-4">
        <BarChart3 className="w-4 h-4 text-brand-teal" />
        <h3 className="text-sm font-semibold text-gray-200">Feature Breakdown</h3>
      </div>

      {/* Feature bars */}
      <div className="space-y-2.5">
        {sortedFeatures.length === 0 ? (
          <div className="flex items-center justify-center py-12 text-gray-600">
            <div className="text-center">
              <BarChart3 className="w-8 h-8 mx-auto mb-2 opacity-30" />
              <p className="text-sm">No features detected</p>
              <p className="text-xs mt-1 opacity-60">Analysis data appears during calls</p>
            </div>
          </div>
        ) : (
          sortedFeatures.map(([name, value]) => (
            <div key={name} className="group">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-gray-400 group-hover:text-gray-200 transition-colors truncate max-w-[140px]">
                  {formatFeatureName(name)}
                </span>
                <span className="text-xs font-mono text-gray-500 tabular-nums">
                  {(value * 100).toFixed(0)}%
                </span>
              </div>
              <div className="h-2 bg-dark-bg rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-700 ease-out ${getBarColor(value)} ${getBarGlow(value)}`}
                  style={{ width: `${Math.max(value * 100, 2)}%` }}
                />
              </div>
            </div>
          ))
        )}
      </div>

      {/* Legend footer */}
      {sortedFeatures.length > 0 && (
        <div className="mt-4 pt-3 border-t border-dark-border/30">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-gray-600">
              {sortedFeatures.length} active indicator{sortedFeatures.length !== 1 ? 's' : ''}
            </span>
            <div className="flex gap-2">
              {[
                { label: 'Low', color: 'bg-brand-teal' },
                { label: 'Med', color: 'bg-warning' },
                { label: 'High', color: 'bg-orange-500' },
                { label: 'Crit', color: 'bg-alert' },
              ].map(item => (
                <div key={item.label} className="flex items-center gap-1">
                  <div className={`w-1.5 h-1.5 rounded-full ${item.color}`} />
                  <span className="text-[9px] text-gray-600">{item.label}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
