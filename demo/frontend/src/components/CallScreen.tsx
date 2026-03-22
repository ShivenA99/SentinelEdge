import { useEffect, useState } from 'react'
import {
  Phone,
  PhoneOff,
  Mic,
  MicOff,
  Volume2,
  Grid3X3,
  UserCircle2,
  Wifi,
  Battery,
  Signal,
  Camera,
} from 'lucide-react'
import FraudAlert from './FraudAlert'

interface CallScreenProps {
  callerName: string
  callerNumber: string
  duration: number
  isActive: boolean
  onEndCall: () => void
  onBlock: () => void
  onDismissAlert: () => void
  alertLevel: 'safe' | 'low' | 'medium' | 'high' | 'critical'
  alertReasons: string[]
  fraudScore: number
}

function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60)
  const secs = seconds % 60
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
}

function getCurrentTime(): string {
  const now = new Date()
  return now.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true })
}

export default function CallScreen({
  callerName,
  callerNumber,
  duration,
  isActive,
  onEndCall,
  onBlock,
  onDismissAlert,
  alertLevel,
  alertReasons,
  fraudScore,
}: CallScreenProps) {
  const [isMuted, setIsMuted] = useState(false)
  const [isSpeaker, setIsSpeaker] = useState(false)
  const [isScreenShaking, setIsScreenShaking] = useState(false)

  const showAlert = isActive && (alertLevel === 'high' || alertLevel === 'critical' || alertLevel === 'medium')
  const callerInitials = callerName
    .split(' ')
    .filter(Boolean)
    .slice(0, 2)
    .map(part => part[0]?.toUpperCase())
    .join('')

  useEffect(() => {
    if (!showAlert) return

    setIsScreenShaking(true)
    const timeout = window.setTimeout(() => {
      setIsScreenShaking(false)
    }, 480)

    return () => window.clearTimeout(timeout)
  }, [showAlert, alertLevel])

  return (
    <div className={`w-full h-full flex flex-col bg-gradient-to-b from-dark-bg via-dark-card to-dark-bg ${isScreenShaking ? 'phone-screen-shake' : ''}`}>
      {/* Status bar */}
      <div className="relative z-30 flex items-center justify-between px-7 pt-4 pb-2 text-[11px] text-white/85">
        <span className="min-w-[52px] font-semibold tracking-[0.01em]">{getCurrentTime()}</span>
        <div className="w-[126px]" />
        <div className="flex min-w-[56px] items-center justify-end gap-1.5">
          <Signal className="h-3.5 w-3.5 stroke-[2.2]" />
          <Wifi className="h-3.5 w-3.5 stroke-[2.2]" />
          <Battery className="h-3.5 w-3.5 stroke-[2.2]" />
        </div>
      </div>

      {/* Fraud Alert Overlay */}
      {showAlert && (
        <div className="absolute inset-0 z-20 pt-[42px]">
          <FraudAlert
            riskLevel={alertLevel as 'medium' | 'high' | 'critical'}
            score={fraudScore}
            reasons={alertReasons}
            onBlock={onBlock}
            onDismiss={onDismissAlert}
          />
        </div>
      )}

      {/* Main call content */}
      <div className="flex-1 flex flex-col items-center justify-center px-6 relative">
        {isActive ? (
          <>
            {/* Caller avatar */}
            <div className="relative mb-4">
              <div className={`relative w-[92px] h-[92px] rounded-full flex items-center justify-center overflow-hidden shadow-[0_20px_45px_rgba(15,23,42,0.38)] ${
                alertLevel === 'critical' ? 'ring-2 ring-alert/80 animate-pulse-alert' :
                alertLevel === 'high' ? 'ring-2 ring-warning/80' :
                'ring-2 ring-brand-teal/50'
              }`}>
                <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_28%,rgba(255,255,255,0.28),transparent_28%),linear-gradient(160deg,#8892a6_0%,#646f84_45%,#3b4456_100%)]" />
                <div className="absolute inset-x-5 bottom-0 h-[44px] rounded-t-[28px] bg-white/18 blur-[1px]" />
                <div className="absolute left-1/2 top-[18px] h-[26px] w-[26px] -translate-x-1/2 rounded-full bg-white/28" />
                <div className="absolute left-1/2 top-[38px] h-[34px] w-[54px] -translate-x-1/2 rounded-t-[26px] bg-white/22" />
                <div className="absolute inset-0 bg-gradient-to-b from-white/5 via-transparent to-black/25" />
                <div className="absolute bottom-2 left-1/2 -translate-x-1/2 rounded-full bg-black/28 px-2 py-0.5 backdrop-blur-sm">
                  <span className="text-[10px] font-semibold tracking-[0.2em] text-white/88">{callerInitials || 'CP'}</span>
                </div>
              </div>

              <div className="absolute -top-1 -right-1 rounded-full border border-white/10 bg-black/45 p-1.5 shadow-lg backdrop-blur-md">
                <Camera className="h-3.5 w-3.5 text-white/75" />
              </div>

              {/* SentinelEdge protection badge */}
              <div className="absolute -bottom-1 -right-1 w-6 h-6 rounded-full bg-dark-card border-2 border-brand-teal flex items-center justify-center">
                <div className="w-2 h-2 rounded-full bg-brand-teal" />
              </div>
            </div>

            {/* Caller info */}
            <h2 className="text-lg font-semibold text-white text-center leading-tight">{callerName}</h2>
            <p className="text-xs text-gray-400 mt-0.5 font-mono">{callerNumber}</p>

            {/* Call timer */}
            <div className="mt-3 flex items-center gap-2">
              <div className={`w-1.5 h-1.5 rounded-full ${
                alertLevel === 'critical' ? 'bg-alert animate-pulse' :
                alertLevel === 'high' ? 'bg-warning animate-pulse' :
                'bg-safe animate-pulse'
              }`} />
              <span className="text-sm text-gray-300 font-mono tabular-nums">
                {formatDuration(duration)}
              </span>
            </div>

            {/* SentinelEdge status line */}
            <div className="mt-2 px-3 py-1 rounded-full bg-dark-card/80 border border-dark-border/50">
              <p className="text-[9px] text-gray-500 tracking-wider uppercase flex items-center gap-1">
                <span className={`inline-block w-1 h-1 rounded-full ${
                  alertLevel === 'critical' || alertLevel === 'high' ? 'bg-alert' : 'bg-safe'
                }`} />
                SentinelEdge Active
              </p>
            </div>
          </>
        ) : (
          /* Idle state */
          <div className="text-center">
            <div className="w-20 h-20 rounded-full bg-dark-card flex items-center justify-center mx-auto mb-4 ring-1 ring-dark-border">
              <Phone className="w-10 h-10 text-gray-600" />
            </div>
            <h2 className="text-lg font-medium text-gray-500">No Active Call</h2>
            <p className="text-xs text-gray-600 mt-1">Select a sample call to begin</p>
          </div>
        )}
      </div>

      {/* Bottom action buttons */}
      {isActive && (
        <div className="pb-10 px-6">
          {/* Action row */}
          <div className="grid grid-cols-4 gap-4 mb-6">
            <button
              onClick={() => setIsMuted(!isMuted)}
              className="flex flex-col items-center gap-1"
            >
              <div className={`w-11 h-11 rounded-full flex items-center justify-center transition-colors ${
                isMuted ? 'bg-white text-black' : 'bg-dark-surface/80 text-white'
              }`}>
                {isMuted ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
              </div>
              <span className="text-[9px] text-gray-400">
                {isMuted ? 'unmute' : 'mute'}
              </span>
            </button>

            <button className="flex flex-col items-center gap-1">
              <div className="w-11 h-11 rounded-full bg-dark-surface/80 flex items-center justify-center text-white">
                <Grid3X3 className="w-5 h-5" />
              </div>
              <span className="text-[9px] text-gray-400">keypad</span>
            </button>

            <button
              onClick={() => setIsSpeaker(!isSpeaker)}
              className="flex flex-col items-center gap-1"
            >
              <div className={`w-11 h-11 rounded-full flex items-center justify-center transition-colors ${
                isSpeaker ? 'bg-white text-black' : 'bg-dark-surface/80 text-white'
              }`}>
                <Volume2 className="w-5 h-5" />
              </div>
              <span className="text-[9px] text-gray-400">speaker</span>
            </button>

            <button className="flex flex-col items-center gap-1">
              <div className="w-11 h-11 rounded-full bg-dark-surface/80 flex items-center justify-center text-white">
                <UserCircle2 className="w-5 h-5" />
              </div>
              <span className="text-[9px] text-gray-400">contacts</span>
            </button>
          </div>

          {/* End call button */}
          <div className="flex justify-center">
            <button
              onClick={onEndCall}
              className="w-16 h-16 rounded-full bg-alert flex items-center justify-center hover:bg-alert-dark transition-colors shadow-lg shadow-alert/30 active:scale-95"
            >
              <PhoneOff className="w-7 h-7 text-white rotate-[135deg]" />
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
