import { useEffect, useState } from 'react'
import { ShieldCheck, ShieldAlert, ShieldX } from 'lucide-react'

interface ScoreGaugeProps {
  score: number  // 0.0 to 1.0
  label?: string
}

function getGaugeColor(score: number): string {
  if (score < 0.3) return '#10B981'   // safe green
  if (score < 0.5) return '#F59E0B'   // warning amber
  if (score < 0.75) return '#F97316'  // orange
  return '#EF4444'                     // alert red
}

function getGaugeGlow(score: number): string {
  if (score < 0.3) return 'drop-shadow(0 0 8px rgba(16, 185, 129, 0.4))'
  if (score < 0.5) return 'drop-shadow(0 0 8px rgba(245, 158, 11, 0.4))'
  if (score < 0.75) return 'drop-shadow(0 0 8px rgba(249, 115, 22, 0.4))'
  return 'drop-shadow(0 0 12px rgba(239, 68, 68, 0.5))'
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
  const [animatedScore, setAnimatedScore] = useState(0)

  // Smooth animation towards target score
  useEffect(() => {
    const duration = 600
    const startScore = animatedScore
    const startTime = performance.now()

    const animate = (currentTime: number) => {
      const elapsed = currentTime - startTime
      const progress = Math.min(elapsed / duration, 1)
      // Ease out cubic
      const eased = 1 - Math.pow(1 - progress, 3)
      const newScore = startScore + (score - startScore) * eased
      setAnimatedScore(newScore)

      if (progress < 1) {
        requestAnimationFrame(animate)
      }
    }

    requestAnimationFrame(animate)
  }, [score])

  const percentage = Math.round(animatedScore * 100)
  const color = getGaugeColor(animatedScore)
  const glow = getGaugeGlow(animatedScore)

  // SVG arc calculations
  const size = 200
  const strokeWidth = 12
  const radius = (size - strokeWidth) / 2
  const circumference = Math.PI * radius // Semicircle
  const offset = circumference - (animatedScore * circumference)

  // Arc path for semicircle (bottom half open)
  const centerX = size / 2
  const centerY = size / 2

  return (
    <div className="flex flex-col items-center">
      <div className="relative" style={{ width: size, height: size }}>
        <svg
          width={size}
          height={size}
          viewBox={`0 0 ${size} ${size}`}
          className="transform -rotate-90"
          style={{ filter: glow }}
        >
          {/* Background track */}
          <circle
            cx={centerX}
            cy={centerY}
            r={radius}
            fill="none"
            stroke="#1E293B"
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={`${circumference} ${circumference * 2}`}
            transform={`rotate(90, ${centerX}, ${centerY})`}
          />

          {/* Score arc */}
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
            className="transition-all duration-300"
          />

          {/* Tick marks */}
          {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
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
                x1={x1} y1={y1} x2={x2} y2={y2}
                stroke="#475569"
                strokeWidth={1.5}
                strokeLinecap="round"
                transform={`rotate(90, ${centerX}, ${centerY})`}
              />
            )
          })}
        </svg>

        {/* Center content */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <div style={{ color }} className="mb-1">
            {getLevelIcon(animatedScore)}
          </div>
          <span
            className="text-4xl font-bold tabular-nums tracking-tight"
            style={{ color }}
          >
            {percentage}
            <span className="text-lg font-normal text-gray-500">%</span>
          </span>
          <span
            className="text-xs font-semibold mt-0.5 tracking-wide uppercase"
            style={{ color }}
          >
            {getLevelText(animatedScore)}
          </span>
        </div>
      </div>

      {/* Label */}
      <p className="text-xs text-gray-500 font-medium mt-2">{label}</p>

      {/* Mini color legend */}
      <div className="flex items-center gap-3 mt-3">
        {[
          { label: 'Safe', color: '#10B981' },
          { label: 'Warning', color: '#F59E0B' },
          { label: 'High', color: '#F97316' },
          { label: 'Critical', color: '#EF4444' },
        ].map(item => (
          <div key={item.label} className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: item.color }} />
            <span className="text-[9px] text-gray-500">{item.label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
