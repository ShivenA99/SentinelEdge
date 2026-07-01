import { useState } from 'react'
import { Shield, Server, Cpu, Lock, Eye, EyeOff, Phone, FileText, Hash, Radio, ChevronDown, ChevronUp } from 'lucide-react'

interface PrivacyDashboardProps {
  callsAnalyzed?: number
  scamsBlocked?: number
  modelVersion?: string
  localTrainingSamples?: number
  totalDataTransmitted?: number
  privacyBudget?: number
  lastGradientVector?: number[]
  lastTranscriptSnippet?: string
}

const DEFAULT_GRADIENT: number[] = [
  0.000312, -0.000087, 0.000521, -0.000234, 0.000098,
  -0.000445, 0.000167, -0.000089, 0.000378, -0.000201,
  0.000456, -0.000312, 0.000123, -0.000567, 0.000289,
  -0.000134, 0.000445, -0.000223, 0.000089, -0.000378,
  0.000201, -0.000456, 0.000312, -0.000123,
]

const NEVER_ITEMS = [
  { icon: <Phone className="w-5 h-5" />, label: 'Your call recordings' },
  { icon: <FileText className="w-5 h-5" />, label: 'Your transcripts' },
  { icon: <Hash className="w-5 h-5" />, label: 'Who you called' },
  { icon: <Radio className="w-5 h-5" />, label: 'What was said' },
  { icon: <Phone className="w-5 h-5" />, label: 'Your phone number' },
]

function StatRow({
  label,
  value,
  unit = '',
  note,
  highlight = false,
  isText = false,
}: {
  label: string
  value: number | string
  unit?: string
  note?: string
  highlight?: boolean
  isText?: boolean
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-base text-gray-400 font-mono uppercase tracking-wider">{label}</span>
      <div className="flex items-baseline gap-2">
        <span className={`font-mono font-bold ${isText ? 'text-xl text-gray-200' : 'text-4xl'} ${highlight ? 'text-brand-teal' : 'text-white'}`}>
          {value}
        </span>
        {unit && <span className="text-lg text-gray-400 font-mono">{unit}</span>}
      </div>
      {note && <span className="text-base text-gray-500 font-mono">{note}</span>}
    </div>
  )
}

export default function PrivacyDashboard({
  callsAnalyzed = 0,
  scamsBlocked = 0,
  modelVersion = 'v1.2.4',
  localTrainingSamples = 0,
  totalDataTransmitted = 2.4,
  privacyBudget = 0.12,
  lastGradientVector = DEFAULT_GRADIENT,
  lastTranscriptSnippet = '"...this is Officer Wilson from the IRS, you owe $4,789 in back taxes and must pay immediately or face arrest..."',
}: PrivacyDashboardProps) {
  const [inspectorOpen, setInspectorOpen] = useState(false)
  const [showTranscript, setShowTranscript] = useState(false)

  const maxPrivacyBudget = 1.0
  const budgetPercent = Math.min((privacyBudget / maxPrivacyBudget) * 100, 100)
  const budgetColor = budgetPercent < 30 ? '#0D9488' : budgetPercent < 70 ? '#EAB308' : '#EF4444'

  return (
    <div className="space-y-8">

      {/* Header */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <p className="text-base text-brand-teal font-mono tracking-[4px] uppercase mb-2">
            Privacy Report
          </p>
          <h2 className="text-5xl font-bold text-white font-mono tracking-tight">
            What SentinelEdge knows about you
          </h2>
          <p className="text-gray-400 text-xl mt-2">
            Spoiler: almost nothing. Here's the proof.
          </p>
        </div>
        <div className="flex items-center gap-2 bg-brand-teal/10 border border-brand-teal/30 rounded-full px-5 py-3 self-start">
          <Shield className="w-5 h-5 text-brand-teal" />
          <span className="text-brand-teal text-base font-mono tracking-wider">PRIVACY VERIFIED</span>
        </div>
      </div>

      {/* Three column grid */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">

        {/* ON DEVICE */}
        <div className="glass-card p-8 glow-teal">
          <div className="flex items-center gap-3 mb-8">
            <div className="w-10 h-10 rounded-lg bg-brand-teal/10 flex items-center justify-center">
              <Cpu className="w-5 h-5 text-brand-teal" />
            </div>
            <span className="text-lg font-mono tracking-[2px] text-brand-teal uppercase">On Your Device</span>
          </div>
          <div className="space-y-8">
            <StatRow label="Calls analyzed" value={callsAnalyzed} />
            <StatRow label="Scams blocked" value={scamsBlocked} highlight />
            <StatRow label="Model version" value={modelVersion} isText />
            <StatRow
              label="Local training samples"
              value={localTrainingSamples}
              note="feature vectors only — no text"
            />
          </div>
          <div className="mt-8 pt-4 border-t border-dark-border/30">
            <p className="text-base text-gray-500 font-mono leading-relaxed">
              All data stored in isolated on-device memory. Cleared when the app closes.
            </p>
          </div>
        </div>

        {/* SENT TO SERVER */}
        <div className="glass-card p-8">
          <div className="flex items-center gap-3 mb-8">
            <div className="w-10 h-10 rounded-lg bg-gray-700/30 flex items-center justify-center">
              <Server className="w-5 h-5 text-gray-400" />
            </div>
            <span className="text-lg font-mono tracking-[2px] text-gray-400 uppercase">Sent to Server</span>
          </div>
          <div className="space-y-6">
            <div>
              <div className="flex justify-between items-baseline mb-2">
                <span className="text-base text-gray-400 font-mono uppercase tracking-wider">Total transmitted</span>
                <span className="text-4xl font-bold text-white font-mono">
                  {totalDataTransmitted.toFixed(1)}
                  <span className="text-lg text-gray-400 ml-1">KB</span>
                </span>
              </div>
              <p className="text-base text-gray-500 font-mono mb-4">
                ≈ {Math.round(totalDataTransmitted * 1024)} bytes of encrypted gradient noise
              </p>
              <div>
                <div className="flex justify-between mb-2">
                  <span className="text-base text-gray-500 font-mono">Your data</span>
                  <span className="text-base text-gray-500 font-mono">avg webpage (2 MB)</span>
                </div>
                <div className="h-3 bg-dark-border rounded-full overflow-hidden">
                  <div
                    className="h-full bg-brand-teal/50 rounded-full transition-all duration-1000"
                    style={{ width: `${Math.max((totalDataTransmitted / 2000) * 100, 0.5)}%` }}
                  />
                </div>
                <p className="text-base text-brand-teal font-mono mt-2">
                  {totalDataTransmitted.toFixed(1)} KB — {((totalDataTransmitted / 2000) * 100).toFixed(3)}% of a normal webpage
                </p>
              </div>
            </div>
            <div className="pt-4 border-t border-dark-border/30">
              <div className="flex justify-between items-baseline mb-3">
                <span className="text-base text-gray-400 font-mono uppercase tracking-wider">Privacy budget (ε)</span>
                <span className="text-4xl font-bold text-white font-mono">
                  {privacyBudget.toFixed(2)}
                  <span className="text-lg text-gray-400 ml-1">/ {maxPrivacyBudget.toFixed(1)}</span>
                </span>
              </div>
              <div className="h-3 bg-dark-border rounded-full overflow-hidden mb-3">
                <div
                  className="h-full rounded-full transition-all duration-1000"
                  style={{ width: `${budgetPercent}%`, background: budgetColor }}
                />
              </div>
              <p className="text-base text-gray-500 font-mono leading-relaxed">
                Lower ε = stronger privacy. Even if someone stole our server, they can't reconstruct your calls.
              </p>
            </div>
          </div>
        </div>

        {/* WE CAN'T SEE */}
        <div className="glass-card p-8 border border-red-500/10">
          <div className="flex items-center gap-3 mb-8">
            <div className="w-10 h-10 rounded-lg bg-red-500/10 flex items-center justify-center">
              <EyeOff className="w-5 h-5 text-red-400" />
            </div>
            <span className="text-lg font-mono tracking-[2px] text-red-400 uppercase">We Can't See</span>
          </div>
          <div className="space-y-1">
            {NEVER_ITEMS.map((item, i) => (
              <div
                key={i}
                className="flex items-center justify-between py-4 border-b border-dark-border/20 last:border-0"
              >
                <div className="flex items-center gap-3">
                  <span className="text-gray-500">{item.icon}</span>
                  <span className="text-xl text-gray-200 font-mono">{item.label}</span>
                </div>
                <span className="text-lg font-mono font-bold text-red-400 tracking-widest">NEVER</span>
              </div>
            ))}
          </div>
          <div className="mt-5 bg-red-500/5 border border-red-500/10 rounded-lg p-4">
            <p className="text-base text-red-400/70 font-mono leading-relaxed">
              These guarantees are enforced mathematically, not by policy.
              No configuration can override them.
            </p>
          </div>
        </div>
      </div>

      {/* GRADIENT INSPECTOR */}
      <div className="glass-card overflow-hidden">
        <button
          onClick={() => setInspectorOpen(v => !v)}
          className="w-full px-8 py-5 flex items-center justify-between hover:bg-white/5 transition-colors"
        >
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 rounded-lg bg-brand-teal/10 flex items-center justify-center">
              <Lock className="w-5 h-5 text-brand-teal" />
            </div>
            <div className="text-left">
              <p className="text-xl font-semibold text-white">Inspect the Last Gradient Update</p>
              <p className="text-base text-gray-400">See exactly what left your device — and compare it to what was said</p>
            </div>
          </div>
          {inspectorOpen
            ? <ChevronUp className="w-5 h-5 text-gray-500" />
            : <ChevronDown className="w-5 h-5 text-gray-500" />
          }
        </button>

        {inspectorOpen && (
          <div className="border-t border-dark-border/30 p-8 space-y-6">
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">

              {/* Server side */}
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <Server className="w-5 h-5 text-gray-500" />
                  <p className="text-base font-mono text-gray-400 uppercase tracking-wider">
                    What the server received
                  </p>
                </div>
                <div className="bg-dark-bg/60 rounded-lg p-5 border border-gray-700/20">
                  <p className="text-base text-gray-500 font-mono mb-3">
                    Noised gradient vector (DP-SGD, σ=0.5)
                  </p>
                  <div className="font-mono text-sm text-gray-500 leading-relaxed break-all">
                    [
                    {lastGradientVector.map((v, i) => (
                      <span key={i}>
                        <span className={Math.abs(v) > 0.0003 ? 'text-gray-400' : 'text-gray-700'}>
                          {v.toFixed(6)}
                        </span>
                        {i < lastGradientVector.length - 1 ? ', ' : ''}
                      </span>
                    ))}
                    ]
                  </div>
                  <p className="text-base text-gray-600 font-mono mt-4 pt-4 border-t border-gray-700/20">
                    Mathematically meaningless without the original data.
                    The transcript cannot be recovered even with unlimited compute.
                  </p>
                </div>
              </div>

              {/* Device side */}
              <div>
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Eye className="w-5 h-5 text-brand-teal" />
                    <p className="text-base font-mono text-brand-teal uppercase tracking-wider">
                      What actually happened (on device only)
                    </p>
                  </div>
                  <button
                    onClick={() => setShowTranscript(v => !v)}
                    className="text-base font-mono text-gray-500 hover:text-gray-300 transition-colors flex items-center gap-1"
                  >
                    {showTranscript ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    {showTranscript ? 'Hide' : 'Reveal'}
                  </button>
                </div>
                <div className="bg-dark-bg/60 rounded-lg p-5 border border-brand-teal/10 relative overflow-hidden">
                  <p className="text-base text-gray-500 font-mono mb-3">
                    Actual transcript (never transmitted)
                  </p>
                  <p
                    className={`text-base text-gray-300 font-mono leading-relaxed italic transition-all duration-300 select-none ${
                      showTranscript ? 'blur-none' : 'blur-sm'
                    }`}
                  >
                    {lastTranscriptSnippet}
                  </p>
                  {!showTranscript && (
                    <div className="absolute inset-0 flex items-center justify-center bg-dark-bg/20">
                      <button
                        onClick={() => setShowTranscript(true)}
                        className="flex items-center gap-2 px-4 py-2 bg-brand-teal/10 border border-brand-teal/30 rounded-full text-base text-brand-teal font-mono hover:bg-brand-teal/20 transition-colors"
                      >
                        <Eye className="w-4 h-4" />
                        Tap to reveal (stays on device)
                      </button>
                    </div>
                  )}
                  <div className="mt-4 pt-4 border-t border-brand-teal/10 flex items-center gap-2">
                    <Shield className="w-4 h-4 text-brand-teal" />
                    <p className="text-base text-brand-teal/70 font-mono">
                      This data never left your device.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Contrast summary */}
            <div className="bg-gradient-to-r from-brand-teal/5 via-transparent to-red-500/5 border border-dark-border/30 rounded-lg p-6">
              <div className="grid grid-cols-3 gap-4 text-center">
                <div>
                  <p className="text-base text-gray-500 font-mono uppercase tracking-wider mb-2">Full transcript</p>
                  <p className="text-3xl font-bold text-white font-mono">~200 chars</p>
                  <p className="text-base text-brand-teal font-mono">on device only</p>
                </div>
                <div className="flex items-center justify-center text-gray-500 font-mono text-3xl">→</div>
                <div>
                  <p className="text-base text-gray-500 font-mono uppercase tracking-wider mb-2">Gradient sent</p>
                  <p className="text-3xl font-bold text-white font-mono">{lastGradientVector.length} floats</p>
                  <p className="text-base text-gray-500 font-mono">meaningless noise</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}