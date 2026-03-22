import { Play, Mic, MicOff, AlertTriangle, Phone, Shield } from 'lucide-react'

interface DemoControlsProps {
  onSelectCall: (callId: string) => void
  onToggleMic: () => void
  onSelectMicDevice: (deviceValue: string) => void
  isCallActive: boolean
  isMicActive: boolean
  selectedMicDevice: string
  micDevices: Array<{ value: string; label: string }>
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
  onSelectMicDevice,
  isCallActive,
  isMicActive,
  selectedMicDevice,
  micDevices,
  availableCalls,
}: DemoControlsProps) {
  return (
    <div className="glass-card p-4">
      <div className="flex items-center justify-between">
        {/* Left: sample calls */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 mr-2">
            <Phone className="w-4 h-4 text-brand-teal" />
            <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
              Sample Calls
            </span>
          </div>
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
                  flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium
                  border border-dark-border/50 transition-all duration-200
                  ${isCallActive
                    ? 'opacity-40 cursor-not-allowed bg-dark-card text-gray-600'
                    : `bg-dark-card text-gray-300 ${config.color} active:scale-[0.97]`
                  }
                `}
              >
                <Play className="w-3 h-3" />
                {call.description}
              </button>
            )
          })}
        </div>

        {/* Right: mic toggle */}
        <div className="flex items-center gap-3">
          <div className="h-6 w-px bg-dark-border/30" />
          <div className="flex flex-col items-end gap-1">
            <button
              onClick={onToggleMic}
              disabled={isCallActive}
              className={`
                flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium
                transition-all duration-200 border
                ${isMicActive
                  ? 'bg-brand-teal/10 text-brand-teal border-brand-teal/30'
                  : isCallActive
                    ? 'opacity-40 cursor-not-allowed bg-dark-card text-gray-600 border-dark-border/50'
                    : 'bg-dark-card text-gray-300 border-dark-border/50 hover:bg-brand-teal/10 hover:text-brand-teal hover:border-brand-teal/30'
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

            {!isCallActive && (
              <>
                <span className={`text-[10px] ${isMicActive ? 'text-brand-teal/80' : 'text-gray-500'}`}>
                  {isMicActive
                    ? 'Mic is on for your side of role-play'
                    : 'Scripted caller mode: backend plays sample scam call lines'}
                </span>

                <select
                  value={selectedMicDevice}
                  onChange={(e) => onSelectMicDevice(e.target.value)}
                  className="px-2 py-1 text-[10px] bg-dark-card text-gray-300 border border-dark-border/50 rounded-md min-w-[260px]"
                  disabled={isCallActive || micDevices.length === 0}
                >
                  {micDevices.length === 0 ? (
                    <option value="">No input devices found</option>
                  ) : (
                    micDevices.map((device) => (
                      <option key={device.value} value={device.value}>
                        {device.label}
                      </option>
                    ))
                  )}
                </select>
              </>
            )}
          </div>

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
