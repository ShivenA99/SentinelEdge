import { useEffect, useMemo, useRef, useState } from 'react'
import { MessageSquare, Radio } from 'lucide-react'

interface TranscriptPanelProps {
  sentences: Array<{
    text: string
    score: number
    index: number
  }>
  isStreaming: boolean
}

const KEYWORD_STYLES: Array<{ pattern: RegExp; className: string }> = [
  { pattern: /\b(immediately|urgent|hurry|deadline|today|now)\b/gi, className: 'transcript-keyword transcript-keyword-urgent' },
  { pattern: /\b(gift cards?|google play|codes?)\b/gi, className: 'transcript-keyword transcript-keyword-payment' },
  { pattern: /\b(password|pin|social security|verification code|bank details?|ssn)\b/gi, className: 'transcript-keyword transcript-keyword-credential' },
  { pattern: /\b(arrest|warrant|prosecution|criminal)\b/gi, className: 'transcript-keyword transcript-keyword-threat' },
  { pattern: /\b(remote access|download|website|click|login)\b/gi, className: 'transcript-keyword transcript-keyword-action' },
]

function getScoreColor(score: number): string {
  if (score < 0.3) return 'border-safe'
  if (score < 0.5) return 'border-warning'
  if (score < 0.75) return 'border-orange-500'
  return 'border-alert'
}

function getScoreBg(score: number): string {
  if (score < 0.3) return 'bg-safe/5'
  if (score < 0.5) return 'bg-warning/5'
  if (score < 0.75) return 'bg-orange-500/5'
  return 'bg-alert/5'
}

function getScoreBadge(score: number): { bg: string; text: string; pulse: boolean } {
  if (score < 0.3) return { bg: 'bg-safe/10 text-safe', text: 'Safe', pulse: false }
  if (score < 0.5) return { bg: 'bg-warning/10 text-warning', text: 'Caution', pulse: false }
  if (score < 0.75) return { bg: 'bg-orange-500/10 text-orange-400', text: 'Suspicious', pulse: true }
  return { bg: 'bg-alert/10 text-alert', text: 'Danger', pulse: true }
}

function highlightKeywords(text: string) {
  if (!text) return null

  const matches: Array<{ start: number; end: number; className: string }> = []

  KEYWORD_STYLES.forEach(({ pattern, className }) => {
    const regex = new RegExp(pattern.source, pattern.flags)
    let match = regex.exec(text)
    while (match) {
      matches.push({
        start: match.index,
        end: match.index + match[0].length,
        className,
      })
      match = regex.exec(text)
    }
  })

  matches.sort((a, b) => a.start - b.start || b.end - a.end)

  const segments: Array<{ text: string; className?: string }> = []
  let cursor = 0

  matches.forEach(match => {
    if (match.start < cursor) return
    if (match.start > cursor) {
      segments.push({ text: text.slice(cursor, match.start) })
    }
    segments.push({
      text: text.slice(match.start, match.end),
      className: match.className,
    })
    cursor = match.end
  })

  if (cursor < text.length) {
    segments.push({ text: text.slice(cursor) })
  }

  return segments.map((segment, index) => (
    <span key={`${segment.text}-${index}`} className={segment.className}>
      {segment.text}
    </span>
  ))
}

interface TypewriterSentenceProps {
  text: string
  isLatest: boolean
}

function TypewriterSentence({ text, isLatest }: TypewriterSentenceProps) {
  const [visibleLength, setVisibleLength] = useState(isLatest ? 0 : text.length)

  useEffect(() => {
    if (!isLatest) {
      setVisibleLength(text.length)
      return
    }

    setVisibleLength(0)
    const stepMs = Math.max(16, Math.min(34, 520 / Math.max(text.length, 1)))
    const interval = window.setInterval(() => {
      setVisibleLength(current => {
        if (current >= text.length) {
          window.clearInterval(interval)
          return current
        }
        return current + 1
      })
    }, stepMs)

    return () => window.clearInterval(interval)
  }, [isLatest, text])

  const visibleText = useMemo(() => text.slice(0, visibleLength), [text, visibleLength])
  const isTyping = isLatest && visibleLength < text.length

  return (
    <span className={isTyping ? 'typewriter-cursor' : undefined}>
      {highlightKeywords(visibleText)}
    </span>
  )
}

export default function TranscriptPanel({ sentences, isStreaming }: TranscriptPanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [sentences])

  return (
    <div>
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <MessageSquare className="w-4 h-4 text-brand-teal" />
          <h3 className="text-sm font-semibold text-gray-200">Live Transcript</h3>
        </div>
        {isStreaming && (
          <div className="flex items-center gap-1.5 text-xs text-safe">
            <Radio className="w-3 h-3 animate-pulse" />
            <span className="font-medium">Streaming</span>
          </div>
        )}
      </div>

      <div
        ref={scrollRef}
        className="transcript-scroll space-y-2 max-h-[280px] overflow-y-auto pr-2 scroll-smooth"
      >
        {sentences.length === 0 ? (
          <div className="flex items-center justify-center py-12 text-gray-600">
            <div className="text-center">
              <MessageSquare className="w-8 h-8 mx-auto mb-2 opacity-30" />
              <p className="text-sm">Waiting for transcript...</p>
              <p className="text-xs mt-1 opacity-60">Start a sample call to see real-time analysis</p>
            </div>
          </div>
        ) : (
          sentences.map((sentence, i) => {
            const badge = getScoreBadge(sentence.score)
            const isLatest = i === sentences.length - 1

            return (
              <div
                key={sentence.index}
                className={`
                  flex items-start gap-3 rounded-lg border-l-[3px] p-3
                  ${getScoreColor(sentence.score)} ${getScoreBg(sentence.score)}
                  animate-fade-in
                `}
                style={{ animationDelay: `${i * 50}ms` }}
              >
                <span className="mt-0.5 w-4 flex-shrink-0 text-right font-mono text-[10px] text-gray-600">
                  {sentence.index + 1}
                </span>

                <p className="flex-1 font-mono text-sm leading-relaxed text-gray-300">
                  <TypewriterSentence text={sentence.text} isLatest={isLatest} />
                </p>

                <div className="flex flex-shrink-0 flex-col items-end gap-1">
                  <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded ${badge.bg} ${badge.pulse ? 'transcript-score-badge-pulse' : ''}`}>
                    {badge.text}
                  </span>
                  <span className="text-[10px] text-gray-500 font-mono tabular-nums">
                    {(sentence.score * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            )
          })
        )}

        {isStreaming && sentences.length > 0 && (
          <div className="flex items-center gap-2 px-3 py-2">
            <div className="flex gap-1">
              <div className="streaming-dot w-1.5 h-1.5 rounded-full bg-brand-teal" />
              <div className="streaming-dot w-1.5 h-1.5 rounded-full bg-brand-teal" />
              <div className="streaming-dot w-1.5 h-1.5 rounded-full bg-brand-teal" />
            </div>
            <span className="text-[10px] text-gray-500">Listening...</span>
          </div>
        )}
      </div>
    </div>
  )
}
