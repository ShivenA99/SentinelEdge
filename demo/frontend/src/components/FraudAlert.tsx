import { ShieldAlert, AlertTriangle, Info, ShieldX } from 'lucide-react'

interface FraudAlertProps {
  riskLevel: 'medium' | 'high' | 'critical'
  score: number
  reasons: string[]
  onBlock: () => void
  onDismiss: () => void
}

export default function FraudAlert({
  riskLevel,
  score,
  reasons,
  onBlock,
  onDismiss,
}: FraudAlertProps) {
  const configs = {
    critical: {
      bg: 'bg-gradient-to-b from-alert/95 to-alert-dark/95',
      border: 'border-alert-light',
      icon: <ShieldX className="w-10 h-10 text-white drop-shadow-lg" />,
      title: 'FRAUD DETECTED',
      subtitle: 'This call exhibits strong fraud indicators',
      containerClass: 'alert-overlay alert-overlay-critical',
      badgeClass: 'bg-white/20 text-white',
    },
    high: {
      bg: 'bg-gradient-to-b from-warning/95 to-amber-700/95',
      border: 'border-warning-light',
      icon: <AlertTriangle className="w-10 h-10 text-white drop-shadow-lg" />,
      title: 'HIGH RISK CALL',
      subtitle: 'Suspicious patterns detected',
      containerClass: 'alert-overlay',
      badgeClass: 'bg-white/20 text-white',
    },
    medium: {
      bg: 'bg-gradient-to-b from-dark-card/98 to-dark-surface/98',
      border: 'border-warning/50',
      icon: <Info className="w-6 h-6 text-warning" />,
      title: 'Caution',
      subtitle: 'Some suspicious indicators present',
      containerClass: 'alert-banner',
      badgeClass: 'bg-warning/20 text-warning',
    },
  }

  const config = configs[riskLevel]
  const percentage = Math.round(score * 100)

  if (riskLevel === 'medium') {
    return (
      <div className={`mx-3 mt-2 ${config.containerClass}`}>
        <div className={`${config.bg} alert-banner-card backdrop-blur-xl rounded-xl border ${config.border} p-3 shadow-[0_18px_45px_rgba(15,23,42,0.4)]`}>
          <div className="flex items-center gap-3">
            {config.icon}
            <div className="flex-1">
              <p className="text-xs font-semibold text-warning">{config.title}</p>
              <p className="text-[10px] text-gray-400">{config.subtitle}</p>
            </div>
            <div className={`px-2 py-0.5 rounded-full ${config.badgeClass} text-[10px] font-bold`}>
              {percentage}%
            </div>
          </div>
          {reasons.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1">
              {reasons.slice(0, 3).map((reason, i) => (
                <span key={i} className="text-[9px] px-1.5 py-0.5 bg-warning/10 text-warning/80 rounded-full">
                  {reason}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    )
  }

  // Full overlay for high/critical
  return (
    <div className={`w-full h-full ${config.containerClass}`}>
      <div className={`alert-overlay-panel w-full h-full ${config.bg} backdrop-blur-xl border-y ${config.border} flex flex-col items-center justify-center px-6`}>
        {/* Risk icon */}
        <div className="relative mb-4">
          <div className={`w-20 h-20 rounded-full ${
            riskLevel === 'critical' ? 'bg-white/10' : 'bg-white/10'
          } flex items-center justify-center`}>
            {config.icon}
          </div>
          {riskLevel === 'critical' && (
            <div className="absolute inset-0 w-20 h-20 rounded-full border-2 border-white/30 animate-ping" />
          )}
        </div>

        {/* Title */}
        <h3 className="text-xl font-bold text-white tracking-wide mb-1">
          {config.title}
        </h3>
        <p className="text-xs text-white/70 mb-3 text-center">
          {config.subtitle}
        </p>

        {/* Confidence score */}
        <div className={`${config.badgeClass} px-4 py-1.5 rounded-full text-sm font-bold mb-4`}>
          {percentage}% Confidence
        </div>

        {/* Reasons */}
        {reasons.length > 0 && (
          <div className="w-full max-w-[220px] mb-6 space-y-1.5">
            {reasons.slice(0, 4).map((reason, i) => (
              <div key={i} className="flex items-center gap-2 text-[11px] text-white/80">
                <div className="w-1 h-1 rounded-full bg-white/60 flex-shrink-0" />
                <span className="capitalize">{reason}</span>
              </div>
            ))}
          </div>
        )}

        {/* Action buttons */}
        <div className="w-full max-w-[220px] space-y-2">
          <button
            onClick={onBlock}
            className="w-full py-2.5 rounded-xl bg-white text-alert font-bold text-sm flex items-center justify-center gap-2 hover:bg-gray-100 transition-colors active:scale-[0.98]"
          >
            <ShieldAlert className="w-4 h-4" />
            Block Caller
          </button>
          <button
            onClick={onDismiss}
            className="w-full py-2.5 rounded-xl bg-white/10 text-white/80 font-medium text-sm hover:bg-white/20 transition-colors active:scale-[0.98]"
          >
            Dismiss
          </button>
        </div>

        {/* SentinelEdge branding */}
        <p className="text-[8px] text-white/30 mt-4 tracking-widest uppercase">
          Protected by SentinelEdge
        </p>
      </div>
    </div>
  )
}
