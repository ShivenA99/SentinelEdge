import { useState, useCallback, useRef, useEffect } from 'react'
import { Shield, Radio, Lock, GitBranch, Activity } from 'lucide-react'
import PhoneSimulator from './components/PhoneSimulator'
import CallScreen from './components/CallScreen'
import TranscriptPanel from './components/TranscriptPanel'
import ScoreGauge from './components/ScoreGauge'
import FeatureBreakdown from './components/FeatureBreakdown'
import DemoControls from './components/DemoControls'
import PrivacyDemo from './components/PrivacyDemo'
import FederatedDashboard from './components/FederatedDashboard'
import CallHistory from './components/CallHistory'
import type { CallHistoryEntry } from './components/CallHistory'
import { useWebSocket } from './hooks/useWebSocket'

// -------- Sample call scripts --------
interface ScriptLine {
  text: string
  score: number
  features: Record<string, number>
}

const SAMPLE_CALLS: Record<string, { description: string; lines: ScriptLine[] }> = {
  irs_scam: {
    description: 'IRS Scam Call',
    lines: [
      { text: "Hello, this is Officer James Wilson from the Internal Revenue Service.", score: 0.15, features: { urgency: 0.1, financial_terms: 0.2, authority_claim: 0.3 } },
      { text: "We have detected a serious issue with your tax filing for the previous year.", score: 0.25, features: { urgency: 0.2, financial_terms: 0.3, authority_claim: 0.3, threat_language: 0.1 } },
      { text: "There is an outstanding balance of $4,789 that must be resolved immediately.", score: 0.42, features: { urgency: 0.5, financial_terms: 0.7, specific_amount: 0.6, time_pressure: 0.4 } },
      { text: "If this is not paid today, a warrant will be issued for your arrest.", score: 0.68, features: { urgency: 0.9, threat_language: 0.8, time_pressure: 0.9, authority_claim: 0.5 } },
      { text: "You need to purchase Google Play gift cards worth $4,789 and read me the codes.", score: 0.85, features: { urgency: 0.8, financial_terms: 0.9, gift_card_mention: 1.0, unusual_payment: 1.0 } },
      { text: "This is the only way to avoid criminal prosecution. Do it now.", score: 0.92, features: { urgency: 1.0, threat_language: 0.95, time_pressure: 1.0, coercion: 0.9, unusual_payment: 0.8 } },
      { text: "Are you at the store yet? You need to hurry, the deadline is in one hour.", score: 0.95, features: { urgency: 1.0, time_pressure: 1.0, coercion: 0.85, unusual_payment: 0.7 } },
    ],
  },
  tech_support: {
    description: 'Tech Support Scam',
    lines: [
      { text: "Hi, this is Mike from Microsoft Technical Support department.", score: 0.12, features: { authority_claim: 0.3, tech_jargon: 0.1 } },
      { text: "We have detected that your computer has been compromised by a virus.", score: 0.28, features: { urgency: 0.3, tech_jargon: 0.4, fear_induction: 0.3 } },
      { text: "Hackers are currently stealing your personal information as we speak.", score: 0.45, features: { urgency: 0.6, fear_induction: 0.7, time_pressure: 0.5 } },
      { text: "I need you to download a remote access tool so I can fix this for you.", score: 0.62, features: { urgency: 0.5, remote_access: 0.9, tech_jargon: 0.5, trust_request: 0.6 } },
      { text: "Please go to this website and enter your computer password when prompted.", score: 0.78, features: { credential_request: 0.95, remote_access: 0.8, trust_request: 0.7 } },
      { text: "Now I also need your bank details to process the security deposit of $299.", score: 0.91, features: { financial_terms: 0.9, credential_request: 0.8, specific_amount: 0.7, trust_request: 0.6 } },
    ],
  },
  bank_fraud: {
    description: 'Bank Fraud Call',
    lines: [
      { text: "Good afternoon, this is the fraud prevention department at First National Bank.", score: 0.10, features: { authority_claim: 0.3, financial_terms: 0.2 } },
      { text: "We noticed some suspicious activity on your account ending in 4821.", score: 0.22, features: { authority_claim: 0.3, financial_terms: 0.4, specific_detail: 0.3 } },
      { text: "Someone attempted to make a purchase of $2,340 at an electronics store.", score: 0.30, features: { financial_terms: 0.5, urgency: 0.3, specific_amount: 0.4 } },
      { text: "To verify your identity, I need you to confirm your full Social Security number.", score: 0.65, features: { credential_request: 0.95, authority_claim: 0.4, pii_request: 0.9 } },
      { text: "I also need your online banking password and the PIN for your debit card.", score: 0.88, features: { credential_request: 1.0, pii_request: 1.0, financial_terms: 0.7 } },
      { text: "Please also share the verification code that was just sent to your phone.", score: 0.93, features: { credential_request: 0.9, mfa_bypass: 1.0, urgency: 0.6 } },
    ],
  },
  legitimate: {
    description: 'Legitimate Call',
    lines: [
      { text: "Hi, this is Sarah from Dr. Thompson's office calling about your appointment.", score: 0.03, features: { authority_claim: 0.05 } },
      { text: "We have you scheduled for next Tuesday at 2:30 PM for your annual checkup.", score: 0.02, features: {} },
      { text: "I wanted to confirm that the time still works for you.", score: 0.01, features: {} },
      { text: "Also, please remember to bring your insurance card and a photo ID.", score: 0.04, features: { pii_request: 0.05 } },
      { text: "If you need to reschedule, you can call us back at the number on our website.", score: 0.02, features: {} },
      { text: "Thank you, and we look forward to seeing you next week. Have a great day!", score: 0.01, features: {} },
    ],
  },
}

const AVAILABLE_CALLS = Object.entries(SAMPLE_CALLS).map(([id, call]) => ({
  id,
  description: call.description,
}))

type AlertLevel = 'safe' | 'low' | 'medium' | 'high' | 'critical'

function getAlertLevel(score: number): AlertLevel {
  if (score < 0.15) return 'safe'
  if (score < 0.3) return 'low'
  if (score < 0.5) return 'medium'
  if (score < 0.75) return 'high'
  return 'critical'
}

type Tab = 'detection' | 'privacy' | 'federated'

interface AudioDevice {
  index: number
  name: string
  max_input_channels: number
  default_samplerate: number
}

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('detection')
  const [isCallActive, setIsCallActive] = useState(false)
  const [isMicActive, setIsMicActive] = useState(true)
  const [callDuration, setCallDuration] = useState(0)
  const [currentCallId, setCurrentCallId] = useState<string | null>(null)
  const [sentences, setSentences] = useState<Array<{ text: string; score: number; index: number }>>([])
  const [currentScore, setCurrentScore] = useState(0)
  const [emaScore, setEmaScore] = useState(0)
  const [features, setFeatures] = useState<Record<string, number>>({})
  const [alertDismissed, setAlertDismissed] = useState(false)
  const [fraudSignalReceived, setFraudSignalReceived] = useState(false)
  const [privacySentences, setPrivacySentences] = useState<Array<{ text: string; score: number; features: Record<string, number> }>>([])
  const [gradientVectors, setGradientVectors] = useState<number[][]>([])
  const [isBackendDriven, setIsBackendDriven] = useState(false)
  const [audioDevices, setAudioDevices] = useState<AudioDevice[]>([])
  const [selectedInputDevice, setSelectedInputDevice] = useState<string>('')
  const [callHistory, setCallHistory] = useState<CallHistoryEntry[]>([])

  const durationRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const speakCallerLine = useCallback((text: string) => {
    if (!('speechSynthesis' in window)) {
      return
    }

    // Keep caller speech readable and avoid queued overlap.
    window.speechSynthesis.cancel()
    const utterance = new SpeechSynthesisUtterance(text)
    utterance.rate = 0.95
    utterance.pitch = 0.95
    window.speechSynthesis.speak(utterance)
  }, [])

  // WebSocket hook for live backend connection
  const wsUrl = currentCallId
    ? (() => {
        const params = new URLSearchParams()
        const isScriptedCall = currentCallId !== 'live_mic'
        if (isScriptedCall || isMicActive) {
          params.set('interactive', '1')
        }
        if (selectedInputDevice) {
          params.set('input_device', selectedInputDevice)
        }
        const qs = params.toString()
        const wsBase = import.meta.env.DEV
          ? 'ws://localhost:8000'
          : `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}`
        return `${wsBase}/ws/call/${currentCallId}${qs ? `?${qs}` : ''}`
      })()
    : ''

  const { connect, disconnect, send, isConnected } = useWebSocket({
    url: wsUrl,
    onMessage: (data) => {
      if (data.type === 'call_start') {
        setIsBackendDriven(true)
        return
      }

      if (data.type === 'sentence') {
        const speaker = data.speaker === 'you' ? 'You' : 'Scammer'
        const isUser = data.speaker === 'you'
        const displayScore = isUser
          ? (data.raw_score ?? 0)
          : (data.ema_score ?? data.raw_score ?? 0)
        const newSentence = {
          text: `[${speaker}] ${data.text}`,
          score: displayScore,
          index: data.index ?? 0,
        }
        setSentences(prev => [...prev, newSentence])
        if (!isUser) {
          setCurrentScore(data.raw_score ?? 0)
          setEmaScore(data.ema_score ?? data.raw_score ?? 0)
          if (data.features) setFeatures(data.features)
        }
        setPrivacySentences(prev => [
          ...prev,
          {
            text: data.text,
            score: displayScore,
            features: data.features ?? {},
          },
        ])

        // In scripted call mode, play the caller sentence aloud.
        if (data.speaker !== 'you' && currentCallId !== 'live_mic') {
          speakCallerLine(data.text)
        }
        return
      }

      if (data.type === 'fraud_detected') {
        setFraudSignalReceived(true)
        setAlertDismissed(false)
        if (typeof data.ema_score === 'number') {
          setEmaScore(data.ema_score)
        }
        if (Array.isArray(data.reasons) && data.reasons.length > 0) {
          const mapped: Record<string, number> = {}
          data.reasons.forEach((reason: string, idx: number) => {
            mapped[`reason_${idx + 1}_${reason.toLowerCase().replace(/[^a-z0-9]+/g, '_')}`] = 1
          })
          setFeatures(mapped)
        }
        return
      }

      if (data.type === 'waiting_for_reply') {
        setSentences(prev => [
          ...prev,
          {
            text: `[System] Your turn: speak now (window ${data.timeout_seconds ?? 15}s)`,
            score: 0,
            index: prev.length,
          },
        ])
        return
      }

      if (data.type === 'user_timeout') {
        setSentences(prev => [
          ...prev,
          {
            text: '[System] No reply detected, continuing call flow.',
            score: 0,
            index: prev.length,
          },
        ])
        return
      }

      if (data.type === 'user_echo_detected') {
        setSentences(prev => [
          ...prev,
          {
            text: `[System] ${data.message}`,
            score: 0,
            index: prev.length,
          },
        ])
        return
      }

      if (data.type === 'call_end' || data.type === 'call_blocked') {
        setIsCallActive(false)
        setCurrentCallId(null)
        setFraudSignalReceived(false)
        if (durationRef.current) {
          clearInterval(durationRef.current)
          durationRef.current = null
        }
        if ('speechSynthesis' in window) {
          window.speechSynthesis.cancel()
        }
        return
      }

      if (data.type === 'error') {
        console.error('Backend error:', data.message)
        setSentences(prev => [
          ...prev,
          {
            text: `[System] ${data.message}`,
            score: 0,
            index: prev.length,
          },
        ])
      }
    },
    onOpen: () => console.log('WebSocket connected'),
    onClose: () => {
      console.log('WebSocket disconnected')
      setIsBackendDriven(false)
    },
    autoConnect: false,
  })

  const startCall = useCallback((callId: string) => {
    if (!SAMPLE_CALLS[callId]) return

    // Reset state
    setSentences([])
    setCurrentScore(0)
    setEmaScore(0)
    setFeatures({})
    setAlertDismissed(false)
    setFraudSignalReceived(false)
    setCallDuration(0)
    setCurrentCallId(callId)
    setIsCallActive(true)
    setPrivacySentences([])
    setGradientVectors([])
    setIsBackendDriven(false)

    // Start duration timer
    if (durationRef.current) {
      clearInterval(durationRef.current)
    }
    durationRef.current = setInterval(() => {
      setCallDuration(prev => prev + 1)
    }, 1000)
  }, [])

  useEffect(() => {
    const loadAudioDevices = async () => {
      try {
        const apiBase = import.meta.env.DEV ? 'http://localhost:8000' : ''
        const resp = await fetch(`${apiBase}/api/audio-devices`)
        const data = await resp.json()
        const devices: AudioDevice[] = Array.isArray(data.devices) ? data.devices : []
        setAudioDevices(devices)
        if (devices.length > 0) {
          setSelectedInputDevice(String(devices[0].index))
        }
      } catch {
        setAudioDevices([])
      }
    }

    void loadAudioDevices()
  }, [])

  const localPlaybackRef = useRef<ReturnType<typeof setTimeout>[]>([])

  useEffect(() => {
    if (!isCallActive || !currentCallId) return

    // Try WebSocket first
    connect()

    // After a short delay, check if backend connected.
    // If not, run local playback with built-in scripts.
    const fallbackTimer = setTimeout(() => {
      if (!isBackendDriven && currentCallId) {
        const call = SAMPLE_CALLS[currentCallId]
        if (!call) return

        let ema = 0
        const alpha = 0.3

        call.lines.forEach((line, i) => {
          const timer = setTimeout(() => {
            ema = i === 0 ? line.score : alpha * line.score + (1 - alpha) * ema
            setSentences(prev => [...prev, { text: line.text, score: line.score, index: i }])
            setCurrentScore(line.score)
            setEmaScore(ema)
            setFeatures(line.features)
            setPrivacySentences(prev => [...prev, { text: line.text, score: line.score, features: line.features }])
            if (ema >= 0.5) setFraudSignalReceived(true)
          }, (i + 1) * 3000)
          localPlaybackRef.current.push(timer)
        })
      }
    }, 2000)

    return () => {
      clearTimeout(fallbackTimer)
      localPlaybackRef.current.forEach(t => clearTimeout(t))
      localPlaybackRef.current = []
    }
  }, [isCallActive, currentCallId, connect, isBackendDriven])

  // Record a completed call into history
  const recordCallHistory = useCallback((outcome: 'Blocked' | 'Dismissed' | 'Completed') => {
    if (!currentCallId) return
    const call = SAMPLE_CALLS[currentCallId]
    if (!call) return

    // Gather top features from all sentences seen so far
    const featureCounts: Record<string, number> = {}
    privacySentences.forEach(s => {
      Object.entries(s.features).forEach(([k, v]) => {
        if (v > 0.3) {
          featureCounts[k] = Math.max(featureCounts[k] ?? 0, v)
        }
      })
    })
    const topFeatures = Object.entries(featureCounts)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 5)
      .map(([k]) => k)

    const peakScore = sentences.reduce((max, s) => Math.max(max, s.score), 0)

    const entry: CallHistoryEntry = {
      id: `${currentCallId}_${Date.now()}`,
      callType: currentCallId,
      callLabel: call.description,
      duration: callDuration,
      peakScore,
      finalScore: emaScore,
      outcome,
      timestamp: Date.now(),
      totalSentences: sentences.length,
      topFeatures,
    }
    setCallHistory(prev => [entry, ...prev])
  }, [currentCallId, callDuration, emaScore, sentences, privacySentences])

  const endCall = useCallback(() => {
    recordCallHistory('Completed')
    setIsCallActive(false)
    setCurrentCallId(null)
    if (durationRef.current) {
      clearInterval(durationRef.current)
      durationRef.current = null
    }
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel()
    }
    disconnect()
  }, [disconnect, recordCallHistory])

  const blockCaller = useCallback(() => {
    send({ action: 'block' })
    endCall()
  }, [endCall, send])

  const dismissAlert = useCallback(() => {
    send({ action: 'dismiss' })
    setAlertDismissed(true)
  }, [send])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (durationRef.current) clearInterval(durationRef.current)
      if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel()
      }
    }
  }, [])

  const alertLevel = getAlertLevel(emaScore)
  const effectiveAlertLevel = alertDismissed || !fraudSignalReceived ? 'safe' : alertLevel
  const callerNames: Record<string, string> = {
    live_mic: 'Live Microphone',
    irs_scam: 'Officer James Wilson',
    tech_support: 'Mike - Microsoft',
    bank_fraud: 'First National Bank',
    legitimate: "Dr. Thompson's Office",
  }
  const callerNumbers: Record<string, string> = {
    live_mic: 'On-device audio stream',
    irs_scam: '+1 (202) 555-0147',
    tech_support: '+1 (800) 555-0199',
    bank_fraud: '+1 (312) 555-0183',
    legitimate: '+1 (415) 555-0126',
  }

  const tabs: { id: Tab; label: string; icon: React.ReactNode }[] = [
    { id: 'detection', label: 'Live Detection', icon: <Radio className="w-4 h-4" /> },
    { id: 'privacy', label: 'Privacy Demo', icon: <Lock className="w-4 h-4" /> },
    { id: 'federated', label: 'Federated Learning', icon: <GitBranch className="w-4 h-4" /> },
  ]
  const isDetectionLoading = isCallActive && sentences.length === 0

  return (
    <div className="min-h-screen bg-dark-bg text-gray-100 font-sans">
      {/* Header */}
      <header className="border-b border-dark-border/50 bg-dark-card/50 backdrop-blur-xl sticky top-0 z-50">
        <div className="mx-auto flex max-w-[1440px] flex-col gap-4 px-4 py-4 sm:px-6 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-center gap-3">
            <div className="relative">
              <Shield className="w-8 h-8 text-brand-teal" />
              <div className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 bg-safe rounded-full animate-pulse" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight">
                <span className="text-brand-teal">Sentinel</span>
                <span className="text-white">Edge</span>
              </h1>
              <p className="text-[10px] text-gray-500 font-medium tracking-widest uppercase">
                Federated Edge AI Fraud Detection
              </p>
            </div>
          </div>

          {/* Tabs */}
          <nav className="flex flex-wrap items-center gap-2">
            {tabs.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`control-button px-3 py-2 text-sm font-medium sm:px-4 ${
                  activeTab === tab.id
                    ? 'border-brand-teal/30 bg-brand-teal/10 text-brand-teal'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-white/5'
                }`}
              >
                {tab.icon}
                {tab.label}
              </button>
            ))}
          </nav>

          {/* Status indicator */}
          <div className="flex items-center gap-2 text-xs lg:justify-end">
            <Activity className={`w-3.5 h-3.5 ${isConnected ? 'text-safe' : 'text-gray-500'}`} />
            <span className={isConnected ? 'text-safe' : 'text-gray-500'}>
              {isConnected ? 'Backend Connected' : isBackendDriven ? 'Reconnecting...' : 'Backend Disconnected'}
            </span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="mx-auto max-w-[1440px] px-4 py-5 sm:px-6 sm:py-6">
        {activeTab === 'detection' && (
          <div className="space-y-6">
            {/* Demo Controls */}
            <DemoControls
              onSelectCall={startCall}
              onToggleMic={() => setIsMicActive(!isMicActive)}
              onSelectMicDevice={setSelectedInputDevice}
              isCallActive={isCallActive}
              isMicActive={isMicActive}
              selectedMicDevice={selectedInputDevice}
              micDevices={audioDevices.map(d => ({ value: String(d.index), label: `${d.index}: ${d.name}` }))}
              availableCalls={AVAILABLE_CALLS}
            />

            {/* Main detection layout */}
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
              {/* Phone Simulator */}
              <div className="flex justify-center lg:col-span-4 lg:justify-start">
                <PhoneSimulator>
                  <CallScreen
                    callerName={currentCallId ? callerNames[currentCallId] ?? 'Unknown' : 'No Active Call'}
                    callerNumber={currentCallId ? callerNumbers[currentCallId] ?? '' : ''}
                    duration={callDuration}
                    isActive={isCallActive}
                    onEndCall={endCall}
                    onBlock={blockCaller}
                    onDismissAlert={dismissAlert}
                    alertLevel={effectiveAlertLevel}
                    alertReasons={
                      Object.entries(features)
                        .filter(([, v]) => v > 0.5)
                        .sort(([, a], [, b]) => b - a)
                        .map(([k]) => k.replace(/_/g, ' '))
                    }
                    fraudScore={emaScore}
                  />
                </PhoneSimulator>
              </div>

              {/* Right side panels */}
              <div className="space-y-6 lg:col-span-8">
                {/* Score + Features row */}
                <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                  <div className="glass-card p-6 glow-teal">
                    {isDetectionLoading ? (
                      <div className="panel-skeleton">
                        <div className="mx-auto h-48 w-48 rounded-full skeleton-block" />
                        <div className="mx-auto h-4 w-32 skeleton-block" />
                        <div className="mx-auto h-3 w-40 skeleton-block" />
                      </div>
                    ) : (
                      <ScoreGauge score={emaScore} label="Fraud Score (EMA)" />
                    )}
                  </div>
                  <div className="glass-card p-6">
                    {isDetectionLoading ? (
                      <div className="panel-skeleton">
                        {[0, 1, 2, 3].map(item => (
                          <div key={item} className="panel-skeleton-card space-y-2">
                            <div className="flex items-center justify-between gap-3">
                              <div className="h-3 w-24 skeleton-block" />
                              <div className="h-3 w-10 skeleton-block" />
                            </div>
                            <div className="h-2 w-full rounded-full skeleton-block" />
                          </div>
                        ))}
                      </div>
                    ) : (
                      <FeatureBreakdown features={features} />
                    )}
                  </div>
                </div>

                {/* Transcript */}
                <div className="glass-card p-6">
                  {isDetectionLoading ? (
                    <div className="panel-skeleton">
                      {[0, 1, 2].map(item => (
                        <div key={item} className="panel-skeleton-card space-y-3">
                          <div className="h-3 w-20 skeleton-block" />
                          <div className="h-3 w-full skeleton-block" />
                          <div className="h-3 w-5/6 skeleton-block" />
                        </div>
                      ))}
                    </div>
                  ) : (
                    <TranscriptPanel
                      sentences={sentences}
                      isStreaming={isCallActive}
                    />
                  )}
                </div>
              </div>
            </div>

            {/* Call History - collapsible section below main content */}
            <CallHistory entries={callHistory} />
          </div>
        )}

        {activeTab === 'privacy' && (
          <PrivacyDemo
            sentences={privacySentences}
            gradientVectors={gradientVectors}
            isCallActive={isCallActive}
          />
        )}

        {activeTab === 'federated' && (
          <FederatedDashboard />
        )}
      </main>
    </div>
  )
}
