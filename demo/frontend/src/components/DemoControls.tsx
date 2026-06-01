import { Play, Mic, MicOff, AlertTriangle, Phone, Shield, Radio, SlidersHorizontal } from 'lucide-react'

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
    <div className="glass-card overflow-hidden p-0">
      <div className="border-b border-white/10 bg-slate-950/40 px-4 py-3 sm:px-5">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-3">
            <div className="grid h-9 w-9 place-items-center rounded-lg border border-brand-teal/25 bg-brand-teal/10">
              <SlidersHorizontal className="h-4 w-4 text-brand-teal-light" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-white">Scenario Console</h3>
              <p className="text-xs text-slate-500">Launch a call, enable live microphone role-play, and inspect model signals.</p>
            </div>
          </div>

          {isCallActive && (
            <div className="inline-flex items-center gap-2 rounded-full border border-safe/20 bg-safe/10 px-3 py-1.5">
              <Radio className="h-3.5 w-3.5 animate-pulse text-safe" />
              <span className="text-[11px] font-semibold uppercase tracking-wider text-safe">Call Active</span>
            </div>
          )}
        </div>
      </div>

      <div className="flex flex-col gap-5 p-4 sm:p-5 xl:flex-row xl:items-start xl:justify-between">
        {/* Left: sample calls */}
        <div className="flex flex-1 flex-col gap-3">
          <div className="flex items-center gap-2">
            <Phone className="w-4 h-4 text-brand-teal" />
            <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
              Sample Calls
            </span>
          </div>
          <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
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
                    control-button min-h-[58px] justify-start px-3 py-3 text-left text-xs font-medium
                    ${isCallActive ? '' : config.color}
                  `}
                >
                  <span className="grid h-8 w-8 flex-shrink-0 place-items-center rounded-lg bg-white/5 text-slate-300">
                    {config.icon}
                  </span>
                  <span className="min-w-0">
                    <span className="block truncate text-slate-200">{call.description}</span>
                    <span className="mt-0.5 block text-[10px] font-normal text-slate-500">
                      {call.id === 'legitimate' ? 'Benign baseline' : 'Risk pattern replay'}
                    </span>
                  </span>
                </button>
              )
            })}
          </div>
        </div>

        {/* Right: mic toggle */}
        <div className="flex min-w-0 flex-col gap-3 rounded-xl border border-white/10 bg-slate-950/30 p-3 xl:w-[330px]">
          <div className="flex items-center justify-between gap-3">
            <span className="text-xs font-semibold uppercase tracking-wider text-gray-400">Input Mode</span>
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
          </div>

            {!isCallActive && (
              <>
                <span className={`text-[11px] leading-4 ${isMicActive ? 'text-brand-teal/80' : 'text-gray-500'}`}>
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
      </div>
    </div>
  )
}
