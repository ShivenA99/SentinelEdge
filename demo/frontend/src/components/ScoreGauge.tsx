import { useEffect, useMemo, useRef, useState, type CSSProperties } from 'react'
import { ShieldCheck, ShieldAlert, ShieldX } from 'lucide-react'

interface ScoreGaugeProps {
  score: number
  label?: string
}

interface ParticleBurst {
  id: number
  color: string
}

const THRESHOLDS = [0.3, 0.5, 0.75]
const GRADIENT_STOPS = [
  { score: 0, color: '#10B981' },
  { score: 0.3, color: '#EAB308' },
  { score: 0.5, color: '#F59E0B' },
  { score: 0.75, color: '#F97316' },
  { score: 1, color: '#EF4444' },
]

function hexToRgb(hex: string) {
  const normalized = hex.replace('#', '')
  const value = Number.parseInt(normalized, 16)
  return {
    r: (value >> 16) & 255,
    g: (value >> 8) & 255,
    b: value & 255,
  }
}

function interpolateColor(score: number): string {
  const safeScore = Math.max(0, Math.min(1, score))

  for (let index = 1; index < GRADIENT_STOPS.length; index += 1) {
    const start = GRADIENT_STOPS[index - 1]
    const end = GRADIENT_STOPS[index]

    if (safeScore <= end.score) {
      const range = end.score - start.score || 1
      const progress = (safeScore - start.score) / range
      const startRgb = hexToRgb(start.color)
      const endRgb = hexToRgb(end.color)

      const r = Math.round(startRgb.r + (endRgb.r - startRgb.r) * progress)
      const g = Math.round(startRgb.g + (endRgb.g - startRgb.g) * progress)
      const b = Math.round(startRgb.b + (endRgb.b - startRgb.b) * progress)

      return `rgb(${r}, ${g}, ${b})`
    }
  }

  return GRADIENT_STOPS[GRADIENT_STOPS.length - 1].color
}

function getLevelText(score: number): string {
  if (score < 0.15) return 'Safe'
  if (score < 0.3) return 'Low Risk'
  if (score < 0.5) return 'Moderate'
  if (score < 0.75) return 'High Risk'
  return 'Critical'
}

function getLevelIcon(score: number) {
  if (score < 0.3) return <ShieldCheck className="w-5 h-5" />
  if (score < 0.75) return <ShieldAlert className="w-5 h-5" />
  return <ShieldX className="w-5 h-5" />
}

export default function ScoreGauge({ score, label = 'Fraud Score' }: ScoreGaugeProps) {
  const clampedScore = Math.max(0, Math.min(1, score))
  const previousScoreRef = useRef(clampedScore)
  const burstIdRef = useRef(0)
  const [bursts, setBursts] = useState<ParticleBurst[]>([])
  const [isThresholdPulseActive, setIsThresholdPulseActive] = useState(false)

  const percentage = Math.round(clampedScore * 100)
  const color = useMemo(() => interpolateColor(clampedScore), [clampedScore])

  const size = 200
  const strokeWidth = 12
  const radius = (size - strokeWidth) / 2
  const circumference = Math.PI * radius
  const offset = circumference - clampedScore * circumference
  const centerX = size / 2
  const centerY = size / 2

  useEffect(() => {
    const previousScore = previousScoreRef.current
    const crossedThreshold = THRESHOLDS.some(
      threshold =>
        (previousScore < threshold && clampedScore >= threshold) ||
        (previousScore > threshold && clampedScore <= threshold)
    )

    previousScoreRef.current = clampedScore

    if (!crossedThreshold) return

    const burst = {
      id: burstIdRef.current,
      color,
    }
    burstIdRef.current += 1

    setBursts(current => [...current, burst])
    setIsThresholdPulseActive(true)

    const pulseTimeout = window.setTimeout(() => {
      setIsThresholdPulseActive(false)
    }, 700)

    const cleanupTimeout = window.setTimeout(() => {
      setBursts(current => current.filter(item => item.id !== burst.id))
    }, 1300)

    return () => {
      window.clearTimeout(pulseTimeout)
      window.clearTimeout(cleanupTimeout)
    }
  }, [clampedScore, color])

  return (
    <div className="flex flex-col items-center">
      <div
        className={`score-gauge-shell relative ${isThresholdPulseActive ? 'score-gauge-threshold-pulse' : ''}`}
        style={
          {
            width: size,
            height: size,
            '--gauge-color': color,
          } as CSSProperties
        }
      >
        <div className="score-gauge-glow absolute inset-[24px] rounded-full" />

        {bursts.map(burst => (
          <div
            key={burst.id}
            className="score-gauge-burst pointer-events-none absolute inset-0"
            style={{ '--burst-color': burst.color } as CSSProperties}
            aria-hidden="true"
          >
            {Array.from({ length: 10 }).map((_, index) => {
              const angle = (360 / 10) * index
              const distance = 36 + (index % 3) * 10
              const delay = `${index * 35}ms`
              return (
                <span
                  key={index}
                  className="score-gauge-particle absolute left-1/2 top-1/2"
                  style={
                    {
                      '--particle-angle': `${angle}deg`,
                      '--particle-distance': `${distance}px`,
                      animationDelay: delay,
                    } as CSSProperties
                  }
                />
              )
            })}
          </div>
        ))}

        <svg
          width={size}
          height={size}
          viewBox={`0 0 ${size} ${size}`}
          className="score-gauge-svg transform -rotate-90"
        >
          <defs>
            <linearGradient id="scoreGaugeTrack" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#0f172a" />
              <stop offset="100%" stopColor="#1e293b" />
            </linearGradient>
          </defs>

          <circle
            cx={centerX}
            cy={centerY}
            r={radius}
            fill="none"
            stroke="url(#scoreGaugeTrack)"
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={`${circumference} ${circumference * 2}`}
            transform={`rotate(90, ${centerX}, ${centerY})`}
          />

          <circle
            cx={centerX}
            cy={centerY}
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={`${circumference} ${circumference * 2}`}
            strokeDashoffset={offset}
            transform={`rotate(90, ${centerX}, ${centerY})`}
            className="gauge-arc"
          />

          {[0, 0.25, 0.5, 0.75, 1].map(tick => {
            const angle = (tick * 180 - 90) * (Math.PI / 180)
            const outerR = radius + strokeWidth / 2 + 4
            const innerR = radius + strokeWidth / 2 + 10
            const x1 = centerX + outerR * Math.cos(angle)
            const y1 = centerY + outerR * Math.sin(angle)
            const x2 = centerX + innerR * Math.cos(angle)
            const y2 = centerY + innerR * Math.sin(angle)
            return (
              <line
                key={tick}
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                stroke="#475569"
                strokeWidth={1.5}
                strokeLinecap="round"
                transform={`rotate(90, ${centerX}, ${centerY})`}
              />
            )
          })}
        </svg>

        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <div className="mb-1 score-gauge-center" style={{ color }}>
            {getLevelIcon(clampedScore)}
          </div>
          <span className="score-gauge-center text-4xl font-bold tabular-nums tracking-tight" style={{ color }}>
            {percentage}
            <span className="text-lg font-normal text-gray-500">%</span>
          </span>
          <span className="score-gauge-center mt-0.5 text-xs font-semibold uppercase tracking-wide" style={{ color }}>
            {getLevelText(clampedScore)}
          </span>
        </div>
      </div>

      <p className="text-xs text-gray-500 font-medium mt-2">{label}</p>

      <div className="mt-3 flex items-center gap-3">
        {[
          { label: 'Safe', color: '#10B981' },
          { label: 'Warning', color: '#EAB308' },
          { label: 'High', color: '#F97316' },
          { label: 'Critical', color: '#EF4444' },
        ].map(item => (
          <div key={item.label} className="flex items-center gap-1">
            <div className="h-2 w-2 rounded-full" style={{ backgroundColor: item.color }} />
            <span className="text-[9px] text-gray-500">{item.label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
