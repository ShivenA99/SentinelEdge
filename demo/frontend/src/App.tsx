import { useState, useCallback, useRef, useEffect } from 'react'
import { Shield, Radio, Lock, Activity, Cpu, Gauge, Server, Zap, Send, Loader2 } from 'lucide-react'
import PhoneSimulator from './components/PhoneSimulator'
import CallScreen from './components/CallScreen'
import TranscriptPanel from './components/TranscriptPanel'
import ScoreGauge from './components/ScoreGauge'
import FeatureBreakdown from './components/FeatureBreakdown'
import DemoControls from './components/DemoControls'
import PrivacyDemo from './components/PrivacyDemo'
import CallHistory from './components/CallHistory'
import type { CallHistoryEntry } from './components/CallHistory'
import { useWebSocket } from './hooks/useWebSocket'
import PrivacyDashboard from './components/PrivacyDashboard'

// -------- Sample call scripts --------
interface ScriptLine {
  text: string
  score: number
  features: Record<string, number>
}

interface AvailableCall {
  id: string
  description: string
  caller?: string
  caller_name?: string
}

interface LatencyStats {
  source: string
  backend: string
  n: number
  p50_ms: number
  p95_ms: number
}

interface CustomScoreSentence {
  index: number
  text: string
  raw_score: number
  ema_score: number
  risk_level: string
  should_alert: boolean
  reasons: string[]
  features: Record<string, number>
  inference_ms: number
}

interface CustomScoreResult {
  sentences: CustomScoreSentence[]
  final_score: number
  peak_score: number
  mean_score: number
  latency: {
    n: number
    p50_ms: number
    p95_ms: number
    total_ms: number
  }
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
  crypto_investment: {
    description: 'Crypto Investment Scam',
    lines: [
      { text: "This is Rachel from Digital Asset Partners calling about your portfolio invitation.", score: 0.16, features: { authority_claim: 0.2, financial_terms: 0.35 } },
      { text: "Our private crypto fund has guaranteed returns if you act before the market closes.", score: 0.48, features: { financial_terms: 0.8, urgency: 0.6, time_pressure: 0.5 } },
      { text: "Transfer five thousand dollars today and I can lock your account bonus.", score: 0.74, features: { financial_terms: 0.9, specific_amount: 0.8, time_pressure: 0.8 } },
      { text: "Do not discuss this with your bank because they may block the investment window.", score: 0.86, features: { coercion: 0.8, trust_request: 0.7, financial_terms: 0.8 } },
    ],
  },
  grandparent: {
    description: 'Grandparent Scam',
    lines: [
      { text: "Grandma, it is me, and I am in serious trouble after a car accident.", score: 0.28, features: { urgency: 0.5, fear_induction: 0.5 } },
      { text: "Please do not tell Mom or Dad because I am embarrassed and scared.", score: 0.52, features: { coercion: 0.7, urgency: 0.5 } },
      { text: "The lawyer needs three thousand dollars in gift cards before I can be released.", score: 0.83, features: { unusual_payment: 1.0, gift_card_mention: 1.0, specific_amount: 0.8 } },
      { text: "Read the card numbers to the lawyer right now so they can close the case.", score: 0.91, features: { urgency: 0.9, credential_request: 0.7, unusual_payment: 0.9 } },
    ],
  },
  amazon_refund: {
    description: 'Amazon Refund Scam',
    lines: [
      { text: "This is Amazon billing support about a mistaken refund on your account.", score: 0.18, features: { authority_claim: 0.3, financial_terms: 0.35 } },
      { text: "We accidentally credited you two thousand dollars and need remote access to correct it.", score: 0.66, features: { remote_access: 0.9, financial_terms: 0.8, specific_amount: 0.7 } },
      { text: "Log in to your bank while I am connected so I can verify the reversal.", score: 0.88, features: { credential_request: 0.9, remote_access: 0.9, trust_request: 0.8 } },
      { text: "If you disconnect, the refund will be marked as theft on your account.", score: 0.93, features: { threat_language: 0.9, urgency: 0.9, coercion: 0.8 } },
    ],
  },
  utility_shutoff: {
    description: 'Utility Shutoff Scam',
    lines: [
      { text: "This is the utility company calling about an overdue balance on your service.", score: 0.18, features: { authority_claim: 0.3, financial_terms: 0.35 } },
      { text: "Your electricity will be disconnected in forty five minutes unless payment is posted.", score: 0.69, features: { urgency: 0.9, time_pressure: 0.9, threat_language: 0.6 } },
      { text: "We cannot accept a card now, but a prepaid voucher will stop the shutoff.", score: 0.82, features: { unusual_payment: 0.95, financial_terms: 0.8, time_pressure: 0.8 } },
      { text: "Stay on the line and read the voucher numbers as soon as you buy them.", score: 0.9, features: { coercion: 0.8, credential_request: 0.7, unusual_payment: 0.9 } },
    ],
  },
  prize_notification: {
    description: 'Prize Notification Scam',
    lines: [
      { text: "Congratulations, your number was selected for the National Sweepstakes grand prize.", score: 0.22, features: { authority_claim: 0.2, financial_terms: 0.3 } },
      { text: "To release the winnings, you only need to pay the processing tax today.", score: 0.62, features: { financial_terms: 0.8, urgency: 0.6, time_pressure: 0.7 } },
      { text: "Buy two money orders and send me the confirmation numbers before noon.", score: 0.86, features: { unusual_payment: 0.9, time_pressure: 0.9, credential_request: 0.7 } },
      { text: "If you tell anyone, the prize committee may disqualify your claim.", score: 0.78, features: { coercion: 0.8, threat_language: 0.6, urgency: 0.7 } },
    ],
  },
}

const FALLBACK_AVAILABLE_CALLS: AvailableCall[] = Object.entries(SAMPLE_CALLS).map(([id, call]) => ({
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

type Tab = 'detection' | 'privacy' | 'federated' | 'privacy-dashboard'

interface AudioDevice {
  index: number
  name: string
  max_input_channels: number
  default_samplerate: number
}

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
  const [fraudSignalReceived, setFraudSignalReceived] = useState(false)
  const [privacySentences, setPrivacySentences] = useState<Array<{ text: string; score: number; features: Record<string, number> }>>([])
  const [gradientVectors, setGradientVectors] = useState<number[][]>([])
  const [isBackendDriven, setIsBackendDriven] = useState(false)
  const [audioDevices, setAudioDevices] = useState<AudioDevice[]>([])
  const [selectedInputDevice, setSelectedInputDevice] = useState<string>('')
  const [callHistory, setCallHistory] = useState<CallHistoryEntry[]>([])
  const [availableCalls, setAvailableCalls] = useState<AvailableCall[]>(FALLBACK_AVAILABLE_CALLS)
  const [latencyStats, setLatencyStats] = useState<LatencyStats | null>(null)
  const [liveInferenceMs, setLiveInferenceMs] = useState<number | null>(null)
  const [customText, setCustomText] = useState('')
  const [customResult, setCustomResult] = useState<CustomScoreResult | null>(null)
  const [customError, setCustomError] = useState<string | null>(null)
  const [isScoringCustom, setIsScoringCustom] = useState(false)

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

  // WebSocket hook for live backend connection.
  const wsUrl = currentCallId
    ? (() => {
        const params = new URLSearchParams()
        if (isMicActive) {
          params.set('interactive', '1')
        }
        if (selectedInputDevice) {
          params.set('input_device', selectedInputDevice)
        }
        const qs = params.toString()
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
        const host = import.meta.env.DEV ? 'localhost:8000' : window.location.host
        return `${protocol}//${host}/ws/call/${currentCallId}${qs ? `?${qs}` : ''}`
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
        backendDeliveredRef.current = true
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
          if (typeof data.inference_ms === 'number') setLiveInferenceMs(data.inference_ms)
          if (data.alert?.should_alert || displayScore >= 0.5) {
            setFraudSignalReceived(true)
            setAlertDismissed(false)
          }
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
    const knownCall = availableCalls.some(call => call.id === callId) || Boolean(SAMPLE_CALLS[callId])
    if (!knownCall) return

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
    setLiveInferenceMs(null)
    setIsBackendDriven(false)

    // Start duration timer
    if (durationRef.current) {
      clearInterval(durationRef.current)
    }
    durationRef.current = setInterval(() => {
      setCallDuration(prev => prev + 1)
    }, 1000)
  }, [availableCalls])

  useEffect(() => {
    const apiBase = import.meta.env.DEV ? 'http://localhost:8000' : ''

    const loadDemoMetadata = async () => {
      try {
        const resp = await fetch(`${apiBase}/api/calls`)
        const data = await resp.json()
        const calls: AvailableCall[] = Array.isArray(data.calls)
          ? data.calls.filter((call: AvailableCall) => call.id !== 'live_mic')
          : []
        if (calls.length > 0) {
          setAvailableCalls(calls)
        }
      } catch {
        setAvailableCalls(FALLBACK_AVAILABLE_CALLS)
      }
    }

    const loadLatencyStats = async () => {
      try {
        const resp = await fetch(`${apiBase}/api/latency`)
        const data = await resp.json()
        if (typeof data.p50_ms === 'number') {
          setLatencyStats(data)
        }
      } catch {
        setLatencyStats(null)
      }
    }

    const loadAudioDevices = async () => {
      try {
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

    void loadDemoMetadata()
    void loadLatencyStats()
    void loadAudioDevices()
  }, [])

  const localPlaybackRef = useRef<ReturnType<typeof setTimeout>[]>([])
  const backendDeliveredRef = useRef(false)

  useEffect(() => {
    if (!isCallActive || !currentCallId) return

    backendDeliveredRef.current = false

    // Try WebSocket first
    connect()

    // After 3 seconds, check if backend actually delivered any sentences.
    // If not, run local playback with built-in scripts.
    const fallbackTimer = setTimeout(() => {
      if (!backendDeliveredRef.current && currentCallId) {
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
    }, 3000)

    return () => {
      clearTimeout(fallbackTimer)
      localPlaybackRef.current.forEach(t => clearTimeout(t))
      localPlaybackRef.current = []
    }
  }, [isCallActive, currentCallId, connect])

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

  const scoreCustomText = useCallback(async () => {
    setIsScoringCustom(true)
    setCustomError(null)
    setCustomResult(null)
    try {
      const apiBase = import.meta.env.DEV ? 'http://localhost:8000' : ''
      const resp = await fetch(`${apiBase}/api/score-text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: customText }),
      })
      if (!resp.ok) {
        throw new Error(`Scoring failed with HTTP ${resp.status}`)
      }
      const data: CustomScoreResult = await resp.json()
      setCustomResult(data)
    } catch (err) {
      setCustomError(err instanceof Error ? err.message : 'Scoring failed')
    } finally {
      setIsScoringCustom(false)
    }
  }, [customText])

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
  const currentCallInfo = currentCallId
    ? availableCalls.find(call => call.id === currentCallId)
    : undefined
  const callerName = currentCallId === 'live_mic'
    ? 'Live Microphone'
    : currentCallInfo?.caller_name ?? (currentCallId ? SAMPLE_CALLS[currentCallId]?.description : 'No Active Call')
  const callerNumber = currentCallId === 'live_mic'
    ? 'On-device audio stream'
    : currentCallInfo?.caller ?? ''
  const modelStats = [
    { label: 'Streaming F1', value: '0.817', detail: 'LR edge model', icon: <Gauge className="h-4 w-4" /> },
    {
      label: 'Latency',
      value: liveInferenceMs != null
        ? `${liveInferenceMs.toFixed(2)} ms`
        : latencyStats ? `${latencyStats.p50_ms.toFixed(3)} ms` : 'Measuring',
      detail: liveInferenceMs != null
        ? 'live, per sentence'
        : latencyStats ? `${latencyStats.backend} p50, n=${latencyStats.n}` : 'live backend p50',
      icon: <Zap className="h-4 w-4" />,
    },
    { label: 'Model Size', value: '<1 KB', detail: 'hand-crafted head', icon: <Cpu className="h-4 w-4" /> },
    { label: 'Privacy', value: 'On-device', detail: 'no raw call upload', icon: <Shield className="h-4 w-4" /> },
  ]

  const tabs: { id: Tab; label: string; icon: React.ReactNode }[] = [
    { id: 'detection', label: 'Live Detection', icon: <Radio className="w-4 h-4" /> },
    { id: 'privacy', label: 'Privacy Demo', icon: <Lock className="w-4 h-4" /> },
    { id: 'privacy-dashboard', label: 'Privacy Dashboard', icon: <Shield className="w-4 h-4" /> },
  ]
  const isDetectionLoading = isCallActive && sentences.length === 0
  const completedCalls = callHistory.length
  const blockedCalls = callHistory.filter(call => call.outcome === 'Blocked').length
  const peakSessionScore = sentences.reduce((max, sentence) => Math.max(max, sentence.score), emaScore)

  return (
    <div className="min-h-screen bg-dark-bg text-gray-100 font-sans app-shell">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-white/10 bg-slate-950/80 backdrop-blur-xl">
        <div className="mx-auto flex max-w-[1440px] flex-col gap-4 px-4 py-4 sm:px-6 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-center gap-3">
            <div className="relative grid h-11 w-11 place-items-center rounded-xl border border-brand-teal/30 bg-brand-teal/10 shadow-lg shadow-brand-teal/10">
              <Shield className="w-6 h-6 text-brand-teal-light" />
              <div className="absolute -right-1 -top-1 h-3 w-3 rounded-full border-2 border-slate-950 bg-safe animate-pulse" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight sm:text-2xl">
                <span className="text-brand-teal">Sentinel</span>
                <span className="text-white">Edge</span>
              </h1>
              <p className="text-[10px] text-slate-400 font-medium tracking-widest uppercase">
                Real-time scam call defense
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
              {isConnected ? 'Live backend' : isBackendDriven ? 'Reconnecting...' : 'Local demo mode'}
            </span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="mx-auto max-w-[1440px] px-4 py-5 sm:px-6 sm:py-6">
        {activeTab === 'detection' && (
          <div className="space-y-6">
            <section className="hero-panel overflow-hidden rounded-2xl border border-white/10 bg-slate-950/60 p-5 shadow-2xl shadow-black/20 sm:p-6">
              <div className="grid gap-5 lg:grid-cols-[1.25fr_1fr] lg:items-center">
                <div className="space-y-4">
                  <div className="inline-flex items-center gap-2 rounded-full border border-brand-teal/25 bg-brand-teal/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-wider text-brand-teal-light">
                    <Server className="h-3.5 w-3.5" />
                    Production demo
                  </div>
                  <div>
                    <h2 className="max-w-3xl text-3xl font-semibold tracking-tight text-white sm:text-4xl">
                      Detect coercive fraud patterns while a call is still unfolding.
                    </h2>
                    <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-300">
                      SentinelEdge scores each spoken turn locally, maintains a streaming risk estimate, and surfaces the exact behaviors that triggered protection.
                    </p>
                  </div>
                  <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                    {modelStats.map(stat => (
                      <div key={stat.label} className="metric-tile">
                        <div className="mb-3 flex items-center justify-between text-slate-400">
                          {stat.icon}
                          <span className="text-[10px] uppercase tracking-wider">{stat.label}</span>
                        </div>
                        <div className="text-xl font-semibold tabular-nums text-white">{stat.value}</div>
                        <div className="mt-1 text-[11px] text-slate-500">{stat.detail}</div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-1">
                  <div className="ops-card">
                    <span className="text-slate-500">Current risk</span>
                    <strong className="text-white">{Math.round(emaScore * 100)}%</strong>
                  </div>
                  <div className="ops-card">
                    <span className="text-slate-500">Peak sentence</span>
                    <strong className="text-white">{Math.round(peakSessionScore * 100)}%</strong>
                  </div>
                  <div className="ops-card">
                    <span className="text-slate-500">Reviewed sessions</span>
                    <strong className="text-white">{completedCalls}</strong>
                  </div>
                  <div className="ops-card">
                    <span className="text-slate-500">Blocked calls</span>
                    <strong className="text-white">{blockedCalls}</strong>
                  </div>
                </div>
              </div>
            </section>

            {/* Demo Controls */}
            <DemoControls
              onSelectCall={startCall}
              onToggleMic={() => setIsMicActive(!isMicActive)}
              onSelectMicDevice={setSelectedInputDevice}
              isCallActive={isCallActive}
              isMicActive={isMicActive}
              selectedMicDevice={selectedInputDevice}
              micDevices={audioDevices.map(d => ({ value: String(d.index), label: `${d.index}: ${d.name}` }))}
              availableCalls={availableCalls}
            />

            <section className="glass-card overflow-hidden p-0">
              <div className="border-b border-white/10 bg-slate-950/40 px-4 py-3 sm:px-5">
                <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                  <div>
                    <h3 className="text-sm font-semibold text-white">Paste Your Own</h3>
                    <p className="text-xs text-slate-500">Score a transcript or suspicious call excerpt with the live backend model.</p>
                  </div>
                  <span className="text-[11px] tabular-nums text-slate-500">{customText.length}/5000</span>
                </div>
              </div>
              <div className="grid gap-4 p-4 sm:p-5 lg:grid-cols-[1.2fr_0.8fr]">
                <div className="space-y-3">
                  <textarea
                    value={customText}
                    onChange={(event) => setCustomText(event.target.value.slice(0, 5000))}
                    placeholder="Paste a call transcript or type a suspicious message..."
                    className="min-h-[140px] w-full resize-y rounded-xl border border-white/10 bg-slate-950/50 px-4 py-3 text-sm leading-6 text-slate-100 outline-none transition focus:border-brand-teal/40 focus:ring-2 focus:ring-brand-teal/10"
                    maxLength={5000}
                  />
                  <button
                    onClick={scoreCustomText}
                    disabled={isScoringCustom}
                    className="control-button px-4 py-2 text-sm font-semibold text-white"
                  >
                    {isScoringCustom ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Send className="h-4 w-4" />
                    )}
                    Score Text
                  </button>
                </div>

                <div className="rounded-xl border border-white/10 bg-slate-950/40 p-4">
                  {customError ? (
                    <div className="text-sm text-alert">{customError}</div>
                  ) : customResult ? (
                    <div className="space-y-4">
                      <div className="grid grid-cols-3 gap-2">
                        <div className="metric-tile">
                          <div className="text-[10px] uppercase tracking-wider text-slate-500">Final</div>
                          <div className="mt-1 text-lg font-semibold tabular-nums text-white">{Math.round(customResult.final_score * 100)}%</div>
                        </div>
                        <div className="metric-tile">
                          <div className="text-[10px] uppercase tracking-wider text-slate-500">Peak</div>
                          <div className="mt-1 text-lg font-semibold tabular-nums text-white">{Math.round(customResult.peak_score * 100)}%</div>
                        </div>
                        <div className="metric-tile">
                          <div className="text-[10px] uppercase tracking-wider text-slate-500">p50</div>
                          <div className="mt-1 text-lg font-semibold tabular-nums text-white">{customResult.latency.p50_ms.toFixed(3)} ms</div>
                        </div>
                      </div>

                      {customResult.sentences.length === 0 ? (
                        <div className="rounded-lg border border-white/10 bg-white/5 p-3 text-sm text-slate-400">
                          No text entered. The scorer is ready for a pasted transcript.
                        </div>
                      ) : (
                        <div className="max-h-[220px] space-y-2 overflow-y-auto pr-1">
                          {customResult.sentences.slice(0, 8).map(sentence => (
                            <div key={sentence.index} className="rounded-lg border border-white/10 bg-white/5 p-3">
                              <div className="mb-1 flex items-center justify-between gap-3 text-[11px] text-slate-500">
                                <span>Sentence {sentence.index + 1}</span>
                                <span className={sentence.should_alert ? 'text-alert' : 'text-brand-teal'}>
                                  EMA {Math.round(sentence.ema_score * 100)}%
                                </span>
                              </div>
                              <p className="text-xs leading-5 text-slate-300">{sentence.text}</p>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="flex h-full min-h-[160px] items-center justify-center rounded-lg border border-dashed border-white/10 text-center text-sm text-slate-500">
                      Results appear here without starting a call.
                    </div>
                  )}
                </div>
              </div>
            </section>

            {/* Main detection layout */}
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
              {/* Phone Simulator */}
              <div className="flex justify-center lg:col-span-4 lg:justify-start">
                <PhoneSimulator>
                  <CallScreen
                    callerName={currentCallId ? callerName : 'No Active Call'}
                    callerNumber={currentCallId ? callerNumber : ''}
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

        {activeTab === 'privacy-dashboard' && (
          <PrivacyDashboard />
        )}
      </main>
    </div>
  )
}
