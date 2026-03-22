import { BarChart3, Info } from 'lucide-react'
import type { CSSProperties } from 'react'

interface FeatureBreakdownProps {
  features: Record<string, number>
}

const FEATURE_EXPLANATIONS: Record<string, string> = {
  authority_claim: 'The caller is presenting themselves as a trusted institution or official authority.',
  coercion: 'Language is pressuring the recipient into acting before they can think or verify.',
  credential_request: 'The conversation is attempting to obtain passwords, PINs, codes, or other secrets.',
  fear_induction: 'The caller is using fear to reduce skepticism and speed up compliance.',
  financial_terms: 'Money, payments, balances, or banking language is becoming central to the call.',
  gift_card_mention: 'Requesting gift cards is a strong scam indicator because they are hard to trace or reverse.',
  mfa_bypass: 'The caller is trying to capture one-time codes or verification steps meant to protect the user.',
  pii_request: 'The caller is asking for sensitive personal information that should not be shared casually.',
  remote_access: 'The caller is steering the user toward remote-control tools or device access.',
  specific_amount: 'Using a precise dollar amount can make a scam feel more legitimate and urgent.',
  specific_detail: 'Specific details are being used to create a false sense of credibility.',
  tech_jargon: 'Technical language may be used to intimidate or confuse the recipient.',
  threat_language: 'Threats of punishment, loss, or legal action are a major fraud signal.',
  time_pressure: 'The caller is creating a countdown or deadline to force a rushed decision.',
  trust_request: 'The caller is explicitly pushing the recipient to trust them or follow instructions without verification.',
  unusual_payment: 'The requested payment method is uncommon for legitimate businesses or institutions.',
  urgency: 'The tone suggests immediate action is required, which is common in scam escalation.',
}

function getSeverity(value: number) {
  if (value < 0.3) {
    return {
      label: 'Low',
      color: '#14B8A6',
      glow: 'rgba(20, 184, 166, 0.16)',
      track: 'rgba(20, 184, 166, 0.12)',
    }
  }
  if (value < 0.5) {
    return {
      label: 'Medium',
      color: '#FBBF24',
      glow: 'rgba(251, 191, 36, 0.18)',
      track: 'rgba(251, 191, 36, 0.12)',
    }
  }
  if (value < 0.75) {
    return {
      label: 'High',
      color: '#FB923C',
      glow: 'rgba(251, 146, 60, 0.2)',
      track: 'rgba(251, 146, 60, 0.12)',
    }
  }
  return {
    label: 'Critical',
    color: '#F87171',
    glow: 'rgba(248, 113, 113, 0.22)',
    track: 'rgba(248, 113, 113, 0.12)',
  }
}

function formatFeatureName(name: string): string {
  return name
    .replace(/_/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase())
}

export default function FeatureBreakdown({ features }: FeatureBreakdownProps) {
  const sortedFeatures = Object.entries(features)
    .filter(([, value]) => value > 0)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 8)

  return (
    <div>
      <div className="mb-4 flex items-center gap-2">
        <BarChart3 className="w-4 h-4 text-brand-teal" />
        <h3 className="text-sm font-semibold text-gray-200">Feature Breakdown</h3>
      </div>

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
          sortedFeatures.map(([name, value], index) => {
            const severity = getSeverity(value)
            const description = FEATURE_EXPLANATIONS[name] ?? 'This signal contributes to the overall fraud risk score.'

            return (
              <div
                key={name}
                className="feature-bar-container group relative"
                style={
                  {
                    '--feature-color': severity.color,
                    '--feature-glow': severity.glow,
                    '--feature-track': severity.track,
                    animationDelay: `${index * 70}ms`,
                  } as CSSProperties
                }
              >
                <div className="mb-1 flex items-center justify-between gap-3">
                  <div className="flex min-w-0 items-center gap-2">
                    <span className="truncate text-xs text-gray-400 transition-colors duration-300 group-hover:text-gray-100">
                      {formatFeatureName(name)}
                    </span>
                    <div className="feature-tooltip-trigger relative flex-shrink-0">
                      <Info className="h-3.5 w-3.5 text-gray-600 transition-colors duration-300 group-hover:text-gray-300" />
                      <div className="feature-tooltip pointer-events-none absolute left-1/2 top-[calc(100%+10px)] z-20 w-56 -translate-x-1/2 rounded-xl border border-white/10 bg-slate-950/95 p-3 text-[11px] leading-relaxed text-gray-200 shadow-[0_18px_45px_rgba(2,6,23,0.5)] backdrop-blur-md">
                        <p className="font-semibold text-white">{formatFeatureName(name)}</p>
                        <p className="mt-1 text-gray-300">{description}</p>
                        <div className="mt-2 flex items-center gap-1.5 text-[10px] uppercase tracking-[0.18em] text-gray-500">
                          <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: severity.color }} />
                          {severity.label} Signal
                        </div>
                      </div>
                    </div>
                  </div>
                  <span className="font-mono text-xs tabular-nums text-gray-500">
                    {(value * 100).toFixed(0)}%
                  </span>
                </div>

                <div className="feature-track h-2 overflow-hidden rounded-full bg-dark-bg">
                  <div
                    className="feature-bar h-full rounded-full"
                    style={{ width: `${Math.max(value * 100, 2)}%` }}
                  />
                </div>
              </div>
            )
          })
        )}
      </div>

      {sortedFeatures.length > 0 && (
        <div className="mt-4 border-t border-dark-border/30 pt-3">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-gray-600">
              {sortedFeatures.length} active indicator{sortedFeatures.length !== 1 ? 's' : ''}
            </span>
            <div className="flex gap-2">
              {[
                { label: 'Low', color: '#14B8A6' },
                { label: 'Med', color: '#FBBF24' },
                { label: 'High', color: '#FB923C' },
                { label: 'Crit', color: '#F87171' },
              ].map(item => (
                <div key={item.label} className="flex items-center gap-1">
                  <div className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: item.color }} />
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
