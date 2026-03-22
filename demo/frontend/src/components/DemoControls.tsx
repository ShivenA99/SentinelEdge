import { Play, Mic, MicOff, AlertTriangle, Phone, Shield } from 'lucide-react'

interface DemoControlsProps {
  onSelectCall: (callId: string) => void
  onToggleMic: () => void
  isCallActive: boolean
  isMicActive: boolean
  availableCalls: Array<{ id: string; description: string }>
}

const callIcons: Record<string, { icon: React.ReactNode; color: string }> = {
  irs_scam: {
    icon: <AlertTriangle className="w-3.5 h-3.5" />,
    color: 'hover:bg-alert/10 hover:text-alert hover:border-alert/30',
  },
  tech_support: {
    icon: <AlertTriangle className="w-3.5 h-3.5" />,
    color: 'hover:bg-orange-500/10 hover:text-orange-400 hover:border-orange-500/30',
  },
  bank_fraud: {
    icon: <AlertTriangle className="w-3.5 h-3.5" />,
    color: 'hover:bg-warning/10 hover:text-warning hover:border-warning/30',
  },
  legitimate: {
    icon: <Shield className="w-3.5 h-3.5" />,
    color: 'hover:bg-safe/10 hover:text-safe hover:border-safe/30',
  },
}

export default function DemoControls({
  onSelectCall,
  onToggleMic,
  isCallActive,
  isMicActive,
  availableCalls,
}: DemoControlsProps) {
  return (
    <div className="glass-card p-4 sm:p-5">
      <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
        {/* Left: sample calls */}
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center">
          <div className="flex items-center gap-2 mr-2">
            <Phone className="w-4 h-4 text-brand-teal" />
            <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
              Sample Calls
            </span>
          </div>
          <div className="flex flex-wrap gap-2">
            {availableCalls.map((call) => {
              const config = callIcons[call.id] || {
                icon: <Play className="w-3.5 h-3.5" />,
                color: 'hover:bg-brand-teal/10 hover:text-brand-teal hover:border-brand-teal/30',
              }
              return (
                <button
                  key={call.id}
                  onClick={() => onSelectCall(call.id)}
                  disabled={isCallActive}
                  className={`
                    control-button px-3 py-2 text-xs font-medium
                    ${isCallActive ? '' : config.color}
                  `}
                >
                  <Play className="w-3 h-3" />
                  {call.description}
                </button>
              )
            })}
          </div>
        </div>

        {/* Right: mic toggle */}
        <div className="flex flex-wrap items-center gap-3 xl:justify-end">
          <div className="hidden h-6 w-px bg-dark-border/30 xl:block" />
          <button
            onClick={onToggleMic}
            disabled={isCallActive}
            className={`
              control-button px-4 py-2 text-xs font-medium
              ${isMicActive
                ? 'bg-brand-teal/10 text-brand-teal border-brand-teal/30'
                : !isCallActive
                  ? 'hover:bg-brand-teal/10 hover:text-brand-teal hover:border-brand-teal/30'
                  : ''
              }
            `}
          >
            {isMicActive ? (
              <Mic className="w-3.5 h-3.5" />
            ) : (
              <MicOff className="w-3.5 h-3.5" />
            )}
            Live Mic
          </button>

          {/* Status indicator */}
          {isCallActive && (
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-safe/10 border border-safe/20">
              <div className="w-1.5 h-1.5 rounded-full bg-safe animate-pulse" />
              <span className="text-[10px] text-safe font-medium">Call Active</span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
