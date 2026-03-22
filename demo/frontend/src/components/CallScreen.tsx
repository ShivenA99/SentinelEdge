import { useState } from 'react'
import {
  Phone,
  PhoneOff,
  Mic,
  MicOff,
  Volume2,
  Grid3X3,
  UserCircle,
  Wifi,
  Battery,
  Signal,
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

  const showAlert = isActive && (alertLevel === 'high' || alertLevel === 'critical' || alertLevel === 'medium')

  return (
    <div className="w-full h-full flex flex-col bg-gradient-to-b from-dark-bg via-dark-card to-dark-bg">
      {/* Status bar */}
      <div className="flex items-center justify-between px-6 pt-10 pb-1 text-[10px] text-gray-400">
        <span className="font-medium">{getCurrentTime()}</span>
        <div className="flex items-center gap-1.5">
          <Signal className="w-3 h-3" />
          <Wifi className="w-3 h-3" />
          <Battery className="w-3 h-3" />
        </div>
      </div>

      {/* Fraud Alert Overlay */}
      {showAlert && (
        <div className="absolute inset-0 z-20 pt-10">
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
              <div className={`w-20 h-20 rounded-full flex items-center justify-center ${
                alertLevel === 'critical' ? 'bg-alert/20 ring-2 ring-alert animate-pulse-alert' :
                alertLevel === 'high' ? 'bg-warning/20 ring-2 ring-warning' :
                'bg-brand-teal/20 ring-2 ring-brand-teal/50'
              }`}>
                <UserCircle className={`w-12 h-12 ${
                  alertLevel === 'critical' ? 'text-alert' :
                  alertLevel === 'high' ? 'text-warning' :
                  'text-brand-teal'
                }`} />
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

            {callerName === 'Live Microphone' && (
              <div className="mt-3 px-3 py-2 rounded-lg bg-brand-teal/10 border border-brand-teal/25">
                <p className="text-[10px] text-brand-teal text-center tracking-wide uppercase">
                  Speak now: live mic analysis in progress
                </p>
              </div>
            )}
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
                <UserCircle className="w-5 h-5" />
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
