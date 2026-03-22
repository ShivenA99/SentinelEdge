import { useState, useEffect, useRef, useCallback } from 'react'

// -------- Fallback example data --------
const FALLBACK_EXAMPLES = [
  {
    text: "This is Officer James Wilson from the Internal Revenue Service.",
    score: 0.15,
    features: { authority_claim: 0.3, financial_terms: 0.2 },
    gradient: Array(12).fill(0),
    sigma: 0.5,
    epsilon: 1.0,
  },
]

// -------- TYPES --------
interface PrivacyDemoProps {
  sentences: Array<{
    text: string
    score: number
    features: Record<string, number | undefined>
  }>
  gradientVectors: number[][]
  isCallActive: boolean
}

interface DisplayEntry {
  text: string
  score: number
  features: Record<string, number | undefined>
  gradient: number[]
  sigma: number
  epsilon: number
}

export default function PrivacyDemo({
  sentences,
  gradientVectors,
  isCallActive,
}: PrivacyDemoProps) {
  const [wsConnected, setWsConnected] = useState(false)
  const [wsEntries, setWsEntries] = useState<DisplayEntry[]>([])
  const [fallbackEntries, setFallbackEntries] = useState<DisplayEntry[]>([])
  const [isReplaying, setIsReplaying] = useState(false)

  const wsRef = useRef<WebSocket | null>(null)
  const replayTimerRef = useRef<any>(null)
  const isMountedRef = useRef(true)

  // -------- WebSocket --------
  useEffect(() => {
    isMountedRef.current = true
    let ws: WebSocket | null = null

    try {
      ws = new WebSocket('ws://localhost:8000/ws/privacy-demo')
      wsRef.current = ws

      ws.onopen = () => {
        if (isMountedRef.current) setWsConnected(true)
      }

      ws.onmessage = (event) => {
        if (!isMountedRef.current) return
        try {
          const data = JSON.parse(event.data)

          const entry: DisplayEntry = {
            text: data.text ?? data.transcript ?? '',
            score: data.fraud_score ?? data.score ?? 0,
            features: data.features ?? {}, // ✅ allow undefined values
            gradient: data.gradient ?? [],
            sigma: data.sigma ?? 0.5,
            epsilon: data.epsilon ?? 1.0,
          }

          setWsEntries((prev) => [...prev, entry])
        } catch {}
      }

      ws.onerror = () => setWsConnected(false)
      ws.onclose = () => setWsConnected(false)
    } catch {
      setWsConnected(false)
    }

    return () => {
      isMountedRef.current = false
      if (ws) ws.close()
    }
  }, [])

  // -------- Data selection --------
  const useWebSocketData = wsConnected && wsEntries.length > 0
  const usePropData = !useWebSocketData && sentences.length > 0

  const displayEntries: DisplayEntry[] = useWebSocketData
    ? wsEntries
    : usePropData
    ? sentences.map((s, i) => ({
        text: s.text,
        score: s.score,
        features: s.features, // ✅ allow undefined
        gradient: gradientVectors[i] ?? Array(12).fill(0),
        sigma: 0.5,
        epsilon: 1.0,
      }))
    : fallbackEntries

  // -------- Replay --------
  const startReplay = useCallback(() => {
    if (isReplaying) return

    setIsReplaying(true)
    setFallbackEntries([])

    let index = 0

    const playNext = () => {
      if (index >= FALLBACK_EXAMPLES.length) {
        setIsReplaying(false)
        return
      }

      setFallbackEntries((prev) => [...prev, FALLBACK_EXAMPLES[index]])
      index++

      replayTimerRef.current = setTimeout(playNext, 1500)
    }

    replayTimerRef.current = setTimeout(playNext, 500)
  }, [isReplaying])

  // -------- UI --------
  return (
    <div className="p-6 text-white">
      <div className="mb-4 flex justify-between">
        <h2 className="text-xl font-bold">Privacy Demo</h2>

        <button
          onClick={startReplay}
          className="bg-teal-500 px-3 py-1 rounded text-sm"
        >
          Replay
        </button>
      </div>

      {displayEntries.map((entry, i) => (
        <div key={i} className="mb-4 border p-3 rounded">
          <p className="text-sm">{entry.text}</p>

          <p className="text-xs text-gray-400 mt-2">
            Score: {(entry.score * 100).toFixed(1)}%
          </p>

          <div className="mt-2 flex gap-2 flex-wrap">
            {Object.entries(entry.features)
              .filter(([, v]) => (v ?? 0) > 0.3) // ✅ SAFE FIX
              .map(([k]) => (
                <span
                  key={k}
                  className="text-xs bg-teal-700 px-2 py-1 rounded"
                >
                  {k}
                </span>
              ))}
          </div>
        </div>
      ))}
    </div>
  )
}