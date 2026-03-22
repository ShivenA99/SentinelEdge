import { useState, useEffect, useRef, useCallback } from 'react'
import { Lock, Smartphone, Server, ArrowRight, Eye, EyeOff, ShieldCheck, RotateCcw, WifiOff, Wifi } from 'lucide-react'

// -------- Fallback example data for offline mode --------
const FALLBACK_EXAMPLES = [
  {
    text: "This is Officer James Wilson from the Internal Revenue Service.",
    score: 0.15,
    features: { authority_claim: 0.3, financial_terms: 0.2 },
    gradient: [-0.003421, 0.001872, -0.000543, 0.004219, -0.002103, 0.000891, -0.001567, 0.003245, 0.000123, -0.002876, 0.001432, -0.000765],
    sigma: 0.5,
    epsilon: 1.0,
  },
  {
    text: "There is an outstanding balance of $4,789 that must be resolved immediately.",
    score: 0.42,
    features: { urgency: 0.5, financial_terms: 0.7, specific_amount: 0.6, time_pressure: 0.4 },
    gradient: [0.002156, -0.004312, 0.001098, -0.003567, 0.000432, -0.002789, 0.004101, -0.001234, 0.003456, -0.000876, 0.002345, -0.001678],
    sigma: 0.5,
    epsilon: 1.0,
  },
  {
    text: "If this is not paid today, a warrant will be issued for your arrest.",
    score: 0.68,
    features: { urgency: 0.9, threat_language: 0.8, time_pressure: 0.9, authority_claim: 0.5 },
    gradient: [-0.001234, 0.003678, -0.002901, 0.000456, 0.004512, -0.003123, 0.001789, -0.000345, 0.002567, -0.004089, 0.000912, -0.003456],
    sigma: 0.5,
    epsilon: 1.0,
  },
  {
    text: "You need to purchase Google Play gift cards worth $4,789 and read me the codes.",
    score: 0.85,
    features: { urgency: 0.8, financial_terms: 0.9, gift_card_mention: 1.0, unusual_payment: 1.0 },
    gradient: [0.004567, -0.001234, 0.002890, -0.003456, 0.000789, -0.004123, 0.001567, 0.003012, -0.002345, 0.000678, -0.001890, 0.004234],
    sigma: 0.5,
    epsilon: 1.0,
  },
  {
    text: "This is the only way to avoid criminal prosecution. Do it now.",
    score: 0.92,
    features: { urgency: 1.0, threat_language: 0.95, time_pressure: 1.0, coercion: 0.9 },
    gradient: [-0.002345, 0.004567, -0.000891, 0.003210, -0.001678, 0.002456, -0.004012, 0.000345, 0.001789, -0.003567, 0.002123, -0.000456],
    sigma: 0.5,
    epsilon: 1.0,
  },
]

interface PrivacyDemoProps {
  sentences: Array<{
    text: string
    score: number
    features: Record<string, number>
  }>
  gradientVectors: number[][]
  isCallActive: boolean
}

interface DisplayEntry {
  text: string
  score: number
  features: Record<string, number>
  gradient: number[]
  sigma: number
  epsilon: number
}

export default function PrivacyDemo({ sentences, gradientVectors, isCallActive }: PrivacyDemoProps) {
  const [wsConnected, setWsConnected] = useState(false)
  const [wsEntries, setWsEntries] = useState<DisplayEntry[]>([])
  const [fallbackEntries, setFallbackEntries] = useState<DisplayEntry[]>([])
  const [isReplaying, setIsReplaying] = useState(false)

  const wsRef = useRef<WebSocket | null>(null)
  const replayTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const isMountedRef = useRef(true)

  // Try connecting to the privacy-demo WebSocket
  useEffect(() => {
    isMountedRef.current = true
    let ws: WebSocket | null = null

    try {
      ws = new WebSocket('ws://localhost:8000/ws/privacy-demo')
      wsRef.current = ws

      ws.onopen = () => {
        if (isMountedRef.current) {
          setWsConnected(true)
        }
      }

      ws.onmessage = (event) => {
        if (!isMountedRef.current) return
        try {
          const data = JSON.parse(event.data)
          const entry: DisplayEntry = {
            text: data.text ?? data.transcript ?? '',
            score: data.fraud_score ?? data.score ?? 0,
            features: data.features ?? {},
            gradient: data.gradient ?? data.noised_gradient ?? [],
            sigma: data.sigma ?? 0.5,
            epsilon: data.epsilon ?? 1.0,
          }
          setWsEntries(prev => [...prev, entry])
        } catch {
          // Ignore malformed messages
        }
      }

      ws.onerror = () => {
        if (isMountedRef.current) {
          setWsConnected(false)
        }
      }

      ws.onclose = () => {
        if (isMountedRef.current) {
          setWsConnected(false)
          wsRef.current = null
        }
      }
    } catch {
      setWsConnected(false)
    }

    return () => {
      isMountedRef.current = false
      if (ws) {
        ws.close(1000, 'Component unmount')
      }
      wsRef.current = null
    }
  }, [])

  // Determine which data to display
  // Priority: WebSocket data > live call data passed via props > fallback examples
  const useWebSocketData = wsConnected && wsEntries.length > 0
  const usePropData = !useWebSocketData && sentences.length > 0

  const displayEntries: DisplayEntry[] = useWebSocketData
    ? wsEntries
    : usePropData
      ? sentences.map((s, i) => ({
          text: s.text,
          score: s.score,
          features: s.features,
          gradient: gradientVectors[i] ?? Array.from({ length: 12 }, () => Number(((Math.random() - 0.5) * 0.01).toFixed(6))),
          sigma: 0.5,
          epsilon: 1.0,
        }))
      : fallbackEntries

  const hasNoData = displayEntries.length === 0 && !isCallActive

  // Replay: cycle through fallback examples one by one
  const startReplay = useCallback(() => {
    if (isReplaying) return
    setIsReplaying(true)
    setFallbackEntries([])
    setWsEntries([])

    let index = 0
    const playNext = () => {
      if (!isMountedRef.current || index >= FALLBACK_EXAMPLES.length) {
        if (isMountedRef.current) setIsReplaying(false)
        return
      }
      const example = FALLBACK_EXAMPLES[index]
      setFallbackEntries(prev => [...prev, example])
      index++
      replayTimerRef.current = setTimeout(playNext, 1500)
    }

    replayTimerRef.current = setTimeout(playNext, 500)
  }, [isReplaying])

  // Cleanup replay timer
  useEffect(() => {
    return () => {
      if (replayTimerRef.current) clearTimeout(replayTimerRef.current)
    }
  }, [])

  return (
    <div className="space-y-6">
      {/* Intro banner */}
      <div className="glass-card p-6 glow-teal">
        <div className="flex items-start gap-4">
          <div className="w-12 h-12 rounded-xl bg-brand-teal/10 flex items-center justify-center flex-shrink-0">
            <Lock className="w-6 h-6 text-brand-teal" />
          </div>
          <div className="flex-1">
            <h2 className="text-lg font-bold text-white mb-1">Privacy-First Architecture</h2>
            <p className="text-sm text-gray-400 leading-relaxed max-w-2xl">
              SentinelEdge processes all sensitive data on your device. The hub server only receives
              differentially-private gradient updates -- mathematical noise that cannot be reverse-engineered
              to reconstruct your conversations or personal information.
            </p>
          </div>
          <div className="flex items-center gap-3 flex-shrink-0">
            {/* Connection status */}
            <div className="flex items-center gap-1.5 text-xs">
              {wsConnected ? (
                <>
                  <Wifi className="w-3.5 h-3.5 text-safe" />
                  <span className="text-safe">Live</span>
                </>
              ) : (
                <>
                  <WifiOff className="w-3.5 h-3.5 text-gray-500" />
                  <span className="text-gray-500">Offline</span>
                </>
              )}
            </div>
            {/* Replay button */}
            <button
              onClick={startReplay}
              disabled={isReplaying || isCallActive}
              className={`
                flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all
                ${isReplaying || isCallActive
                  ? 'bg-gray-700/30 text-gray-600 cursor-not-allowed'
                  : 'bg-brand-teal/10 text-brand-teal border border-brand-teal/20 hover:bg-brand-teal/20 active:scale-[0.97]'
                }
              `}
            >
              <RotateCcw className={`w-3.5 h-3.5 ${isReplaying ? 'animate-spin' : ''}`} />
              {isReplaying ? 'Replaying...' : 'Replay Demo'}
            </button>
          </div>
        </div>
      </div>

      {/* Side by side panels with noise wall */}
      <div className="grid grid-cols-[1fr_auto_1fr] gap-0">
        {/* Device panel */}
        <div className="glass-card border-brand-teal/30 overflow-hidden">
          {/* Header */}
          <div className="px-6 py-4 bg-brand-teal/5 border-b border-brand-teal/20 flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-brand-teal/10 flex items-center justify-center">
              <Smartphone className="w-4 h-4 text-brand-teal" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-brand-teal">Your Device</h3>
              <p className="text-[10px] text-gray-500 uppercase tracking-wider">Everything stays here</p>
            </div>
            <Eye className="w-4 h-4 text-brand-teal ml-auto" />
          </div>

          {/* Content */}
          <div className="p-6 space-y-4 max-h-[500px] overflow-y-auto">
            {displayEntries.length === 0 && !isCallActive ? (
              <div className="text-center py-12">
                <Smartphone className="w-10 h-10 text-gray-700 mx-auto mb-3" />
                <p className="text-sm text-gray-600">Start a call to see device-side data</p>
                <p className="text-xs text-gray-700 mt-1">Full transcript and analysis visible here</p>
                <p className="text-xs text-gray-700 mt-2">
                  Or press <span className="text-brand-teal font-medium">Replay Demo</span> above
                </p>
              </div>
            ) : (
              displayEntries.map((entry, i) => (
                <div key={i} className="animate-fade-in space-y-2">
                  {/* Transcript text */}
                  <div className="bg-dark-bg/50 rounded-lg p-3 border border-brand-teal/10">
                    <p className="text-xs text-gray-500 mb-1 font-medium">Transcript #{i + 1}</p>
                    <p className="text-sm text-gray-200 font-mono leading-relaxed">
                      {entry.text}
                    </p>
                  </div>

                  {/* Features + score */}
                  <div className="flex gap-2">
                    <div className="flex-1 bg-dark-bg/30 rounded-lg p-2">
                      <p className="text-[10px] text-gray-600 mb-1">Fraud Score</p>
                      <p className={`text-lg font-bold tabular-nums ${
                        entry.score < 0.3 ? 'text-safe' :
                        entry.score < 0.5 ? 'text-warning' :
                        entry.score < 0.75 ? 'text-orange-400' :
                        'text-alert'
                      }`}>
                        {(entry.score * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="flex-1 bg-dark-bg/30 rounded-lg p-2">
                      <p className="text-[10px] text-gray-600 mb-1">Features</p>
                      <div className="flex flex-wrap gap-1">
                        {Object.entries(entry.features)
                          .filter(([, v]) => v > 0.3)
                          .slice(0, 3)
                          .map(([k]) => (
                            <span key={k} className="text-[9px] px-1.5 py-0.5 bg-brand-teal/10 text-brand-teal rounded-full">
                              {k.replace(/_/g, ' ')}
                            </span>
                          ))
                        }
                      </div>
                    </div>
                  </div>

                  {i < displayEntries.length - 1 && (
                    <div className="border-b border-dark-border/20" />
                  )}
                </div>
              ))
            )}
          </div>
        </div>

        {/* Visual noise wall - dashed vertical privacy boundary */}
        <div className="flex flex-col items-center justify-center px-4 relative">
          {/* Top label */}
          <div className="absolute top-4 bg-dark-bg/80 px-2 py-1 rounded-md z-10">
            <p className="text-[9px] text-gray-500 uppercase tracking-widest font-bold">Privacy</p>
          </div>

          {/* Dashed line */}
          <div className="h-full w-px relative">
            <div
              className="absolute inset-0 w-px"
              style={{
                backgroundImage: 'repeating-linear-gradient(to bottom, #0D9488 0px, #0D9488 6px, transparent 6px, transparent 14px)',
                opacity: 0.5,
              }}
            />
            {/* Noise glow effect */}
            <div
              className="absolute inset-0 w-6 -left-[11px]"
              style={{
                background: 'linear-gradient(to right, transparent, rgba(13, 148, 136, 0.08), transparent)',
              }}
            />
          </div>

          {/* Lock icon in the middle */}
          <div className="absolute top-1/2 -translate-y-1/2 w-8 h-8 rounded-full bg-dark-bg border border-brand-teal/30 flex items-center justify-center z-10">
            <Lock className="w-3.5 h-3.5 text-brand-teal" />
          </div>

          {/* Bottom label */}
          <div className="absolute bottom-4 bg-dark-bg/80 px-2 py-1 rounded-md z-10">
            <p className="text-[9px] text-gray-500 uppercase tracking-widest font-bold">Barrier</p>
          </div>
        </div>

        {/* Hub server panel */}
        <div className="glass-card border-gray-700/30 overflow-hidden">
          {/* Header */}
          <div className="px-6 py-4 bg-gray-800/30 border-b border-gray-700/20 flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gray-700/30 flex items-center justify-center">
              <Server className="w-4 h-4 text-gray-400" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-gray-400">Hub Server</h3>
              <p className="text-[10px] text-gray-600 uppercase tracking-wider">This is all we send</p>
            </div>
            <EyeOff className="w-4 h-4 text-gray-600 ml-auto" />
          </div>

          {/* Content */}
          <div className="p-6 space-y-4 max-h-[500px] overflow-y-auto">
            {displayEntries.length === 0 && !isCallActive ? (
              <div className="text-center py-12">
                <Server className="w-10 h-10 text-gray-700 mx-auto mb-3" />
                <p className="text-sm text-gray-600">No data transmitted yet</p>
                <p className="text-xs text-gray-700 mt-1">Only noisy gradients are sent</p>
              </div>
            ) : (
              displayEntries.map((entry, i) => (
                <div key={i} className="animate-fade-in">
                  <div className="bg-dark-bg/50 rounded-lg p-3 border border-gray-700/20">
                    <div className="flex items-center justify-between mb-2">
                      <p className="text-xs text-gray-600 font-medium">
                        Gradient Update #{i + 1}
                      </p>
                      <span className="text-[9px] px-1.5 py-0.5 bg-gray-700/30 text-gray-500 rounded-full">
                        DP-SGD
                      </span>
                    </div>
                    <div className="font-mono text-[10px] text-gray-600 leading-relaxed break-all">
                      [{entry.gradient.map((v, j) => (
                        <span key={j}>
                          <span className={
                            Math.abs(v) > 0.005 ? 'text-gray-400' : 'text-gray-700'
                          }>
                            {v.toFixed(6)}
                          </span>
                          {j < entry.gradient.length - 1 ? ', ' : ''}
                        </span>
                      ))}]
                    </div>
                    <div className="mt-2 flex items-center gap-3">
                      <div className="flex items-center gap-1.5">
                        <Lock className="w-2.5 h-2.5 text-gray-600" />
                        <p className="text-[9px] text-gray-600">
                          sigma={entry.sigma.toFixed(1)}
                        </p>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <ShieldCheck className="w-2.5 h-2.5 text-gray-600" />
                        <p className="text-[9px] text-gray-600">
                          epsilon={entry.epsilon.toFixed(1)}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Bottom explanation */}
      <div className="glass-card p-6">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-3 flex-shrink-0">
            <div className="w-10 h-10 rounded-full bg-brand-teal/10 flex items-center justify-center">
              <Smartphone className="w-5 h-5 text-brand-teal" />
            </div>
            <ArrowRight className="w-5 h-5 text-gray-600" />
            <div className="w-10 h-10 rounded-full bg-gray-700/30 flex items-center justify-center">
              <Lock className="w-5 h-5 text-gray-400" />
            </div>
            <ArrowRight className="w-5 h-5 text-gray-600" />
            <div className="w-10 h-10 rounded-full bg-gray-700/30 flex items-center justify-center">
              <Server className="w-5 h-5 text-gray-400" />
            </div>
          </div>
          <div className="flex-1 space-y-2">
            <div className="flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-safe" />
              <span className="text-sm font-semibold text-gray-200">Zero-Knowledge Fraud Detection</span>
            </div>
            <p className="text-xs text-gray-500 leading-relaxed">
              Raw audio and transcripts never leave your device. Only differentially-private model gradients
              are transmitted. Even if the hub server is compromised, your conversation data remains private.
              Mathematical guarantees ensure no individual call can be reconstructed from gradient updates.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
