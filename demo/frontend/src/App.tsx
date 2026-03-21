import { useState, useCallback, useRef, useEffect } from 'react'
import { Shield, Radio, Lock, GitBranch, Activity } from 'lucide-react'
import PhoneSimulator from './components/PhoneSimulator'
import CallScreen from './components/CallScreen'
import TranscriptPanel from './components/TranscriptPanel'
import ScoreGauge from './components/ScoreGauge'
import FeatureBreakdown from './components/FeatureBreakdown'
import FraudAlert from './components/FraudAlert'
import DemoControls from './components/DemoControls'
import PrivacyDemo from './components/PrivacyDemo'
import FederatedDashboard from './components/FederatedDashboard'
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

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('detection')
  const [isCallActive, setIsCallActive] = useState(false)
  const [isMicActive, setIsMicActive] = useState(false)
  const [callDuration, setCallDuration] = useState(0)
  const [currentCallId, setCurrentCallId] = useState<string | null>(null)
  const [sentences, setSentences] = useState<Array<{ text: string; score: number; index: number }>>([])
  const [currentScore, setCurrentScore] = useState(0)
  const [emaScore, setEmaScore] = useState(0)
  const [features, setFeatures] = useState<Record<string, number>>({})
  const [alertDismissed, setAlertDismissed] = useState(false)
  const [privacySentences, setPrivacySentences] = useState<Array<{ text: string; score: number; features: Record<string, number> }>>([])
  const [gradientVectors, setGradientVectors] = useState<number[][]>([])

  const durationRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const playbackRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const lineIndexRef = useRef(0)

  // WebSocket hook for live backend connection
  const { connect, disconnect, isConnected } = useWebSocket({
    url: currentCallId ? `ws://localhost:8000/ws/call/${currentCallId}` : '',
    onMessage: (data) => {
      if (data.type === 'transcript') {
        const newSentence = {
          text: data.text,
          score: data.fraud_score,
          index: sentences.length,
        }
        setSentences(prev => [...prev, newSentence])
        setCurrentScore(data.fraud_score)
        setEmaScore(data.ema_score ?? data.fraud_score)
        if (data.features) setFeatures(data.features)
      }
    },
    onOpen: () => console.log('WebSocket connected'),
    onClose: () => console.log('WebSocket disconnected'),
    autoConnect: false,
  })

  // EMA computation for local playback
  const computeEma = useCallback((newScore: number, prevEma: number, alpha = 0.3) => {
    return alpha * newScore + (1 - alpha) * prevEma
  }, [])

  // Generate fake gradient vector for privacy demo
  const generateGradientVector = useCallback(() => {
    const size = 12
    return Array.from({ length: size }, () => {
      const val = (Math.random() - 0.5) * 0.01
      return Number(val.toFixed(6))
    })
  }, [])

  const startCall = useCallback((callId: string) => {
    const call = SAMPLE_CALLS[callId]
    if (!call) return

    // Reset state
    setSentences([])
    setCurrentScore(0)
    setEmaScore(0)
    setFeatures({})
    setAlertDismissed(false)
    setCallDuration(0)
    setCurrentCallId(callId)
    setIsCallActive(true)
    lineIndexRef.current = 0
    setPrivacySentences([])
    setGradientVectors([])

    // Try connecting to backend WebSocket
    try {
      connect()
    } catch {
      // Fallback: local playback mode
    }

    // Start duration timer
    durationRef.current = setInterval(() => {
      setCallDuration(prev => prev + 1)
    }, 1000)

    // Play lines with realistic timing (local mode)
    let runningEma = 0
    const playLine = (index: number) => {
      if (index >= call.lines.length) return

      const line = call.lines[index]
      runningEma = 0.3 * line.score + 0.7 * runningEma

      setSentences(prev => [...prev, { text: line.text, score: line.score, index }])
      setCurrentScore(line.score)
      setEmaScore(runningEma)
      setFeatures(line.features)
      setPrivacySentences(prev => [...prev, { text: line.text, score: line.score, features: line.features }])
      setGradientVectors(prev => [...prev, generateGradientVector()])

      lineIndexRef.current = index + 1

      // Next line after 2.5-4 seconds (variable for realism)
      const delay = 2500 + Math.random() * 1500
      playbackRef.current = setTimeout(() => playLine(index + 1), delay)
    }

    // Start first line after 1 second
    playbackRef.current = setTimeout(() => playLine(0), 1000)
  }, [connect, generateGradientVector])

  const endCall = useCallback(() => {
    setIsCallActive(false)
    setCurrentCallId(null)
    if (durationRef.current) clearInterval(durationRef.current)
    if (playbackRef.current) clearTimeout(playbackRef.current)
    disconnect()
  }, [disconnect])

  const blockCaller = useCallback(() => {
    endCall()
  }, [endCall])

  const dismissAlert = useCallback(() => {
    setAlertDismissed(true)
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (durationRef.current) clearInterval(durationRef.current)
      if (playbackRef.current) clearTimeout(playbackRef.current)
    }
  }, [])

  const alertLevel = getAlertLevel(emaScore)
  const callerNames: Record<string, string> = {
    irs_scam: 'Officer James Wilson',
    tech_support: 'Mike - Microsoft',
    bank_fraud: 'First National Bank',
    legitimate: "Dr. Thompson's Office",
  }
  const callerNumbers: Record<string, string> = {
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

  return (
    <div className="min-h-screen bg-dark-bg text-gray-100 font-sans">
      {/* Header */}
      <header className="border-b border-dark-border/50 bg-dark-card/50 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-[1440px] mx-auto px-6 py-4 flex items-center justify-between">
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
          <nav className="flex items-center gap-1">
            {tabs.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-2 text-sm font-medium transition-all duration-200 rounded-lg ${
                  activeTab === tab.id
                    ? 'bg-brand-teal/10 text-brand-teal'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-white/5'
                }`}
              >
                {tab.icon}
                {tab.label}
              </button>
            ))}
          </nav>

          {/* Status indicator */}
          <div className="flex items-center gap-2 text-xs">
            <Activity className={`w-3.5 h-3.5 ${isConnected ? 'text-safe' : 'text-gray-500'}`} />
            <span className={isConnected ? 'text-safe' : 'text-gray-500'}>
              {isConnected ? 'Backend Connected' : 'Local Playback Mode'}
            </span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-[1440px] mx-auto px-6 py-6">
        {activeTab === 'detection' && (
          <div className="space-y-6">
            {/* Demo Controls */}
            <DemoControls
              onSelectCall={startCall}
              onToggleMic={() => setIsMicActive(!isMicActive)}
              isCallActive={isCallActive}
              isMicActive={isMicActive}
              availableCalls={AVAILABLE_CALLS}
            />

            {/* Main detection layout */}
            <div className="grid grid-cols-12 gap-6">
              {/* Phone Simulator */}
              <div className="col-span-4 flex justify-center">
                <PhoneSimulator>
                  <CallScreen
                    callerName={currentCallId ? callerNames[currentCallId] ?? 'Unknown' : 'No Active Call'}
                    callerNumber={currentCallId ? callerNumbers[currentCallId] ?? '' : ''}
                    duration={callDuration}
                    isActive={isCallActive}
                    onEndCall={endCall}
                    onBlock={blockCaller}
                    onDismissAlert={dismissAlert}
                    alertLevel={alertDismissed ? 'safe' : alertLevel}
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
              <div className="col-span-8 space-y-6">
                {/* Score + Features row */}
                <div className="grid grid-cols-2 gap-6">
                  <div className="glass-card p-6 glow-teal">
                    <ScoreGauge score={emaScore} label="Fraud Score (EMA)" />
                  </div>
                  <div className="glass-card p-6">
                    <FeatureBreakdown features={features} />
                  </div>
                </div>

                {/* Transcript */}
                <div className="glass-card p-6">
                  <TranscriptPanel
                    sentences={sentences}
                    isStreaming={isCallActive}
                  />
                </div>
              </div>
            </div>
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
