import { useEffect, useRef } from 'react'
import { MessageSquare, Radio } from 'lucide-react'

interface TranscriptPanelProps {
  sentences: Array<{
    text: string
    score: number
    index: number
  }>
  isStreaming: boolean
}

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

function getScoreBadge(score: number): { bg: string; text: string } {
  if (score < 0.3) return { bg: 'bg-safe/10 text-safe', text: 'Safe' }
  if (score < 0.5) return { bg: 'bg-warning/10 text-warning', text: 'Caution' }
  if (score < 0.75) return { bg: 'bg-orange-500/10 text-orange-400', text: 'Suspicious' }
  return { bg: 'bg-alert/10 text-alert', text: 'Danger' }
}

export default function TranscriptPanel({ sentences, isStreaming }: TranscriptPanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom when new sentences arrive
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [sentences])

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
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

      {/* Transcript lines */}
      <div
        ref={scrollRef}
        className="space-y-2 max-h-[280px] overflow-y-auto pr-2 scroll-smooth"
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
            return (
              <div
                key={i}
                className={`
                  flex items-start gap-3 p-3 rounded-lg border-l-[3px]
                  ${getScoreColor(sentence.score)} ${getScoreBg(sentence.score)}
                  animate-fade-in
                `}
                style={{ animationDelay: `${i * 50}ms` }}
              >
                {/* Line number */}
                <span className="text-[10px] text-gray-600 font-mono mt-0.5 flex-shrink-0 w-4 text-right">
                  {sentence.index + 1}
                </span>

                {/* Text */}
                <p className="flex-1 text-sm text-gray-300 font-mono leading-relaxed">
                  {sentence.text}
                </p>

                {/* Score badge */}
                <div className="flex-shrink-0 flex flex-col items-end gap-1">
                  <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded ${badge.bg}`}>
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

        {/* Streaming indicator */}
        {isStreaming && sentences.length > 0 && (
          <div className="flex items-center gap-2 px-3 py-2">
            <div className="flex gap-1">
              <div className="w-1.5 h-1.5 rounded-full bg-brand-teal animate-bounce" style={{ animationDelay: '0ms' }} />
              <div className="w-1.5 h-1.5 rounded-full bg-brand-teal animate-bounce" style={{ animationDelay: '150ms' }} />
              <div className="w-1.5 h-1.5 rounded-full bg-brand-teal animate-bounce" style={{ animationDelay: '300ms' }} />
            </div>
            <span className="text-[10px] text-gray-500">Listening...</span>
          </div>
        )}
      </div>
    </div>
  )
}
