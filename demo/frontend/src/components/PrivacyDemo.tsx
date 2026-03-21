import { Lock, Smartphone, Server, ArrowRight, Eye, EyeOff, ShieldCheck } from 'lucide-react'

interface PrivacyDemoProps {
  sentences: Array<{
    text: string
    score: number
    features: Record<string, number>
  }>
  gradientVectors: number[][]
  isCallActive: boolean
}

export default function PrivacyDemo({ sentences, gradientVectors, isCallActive }: PrivacyDemoProps) {
  return (
    <div className="space-y-6">
      {/* Intro banner */}
      <div className="glass-card p-6 glow-teal">
        <div className="flex items-start gap-4">
          <div className="w-12 h-12 rounded-xl bg-brand-teal/10 flex items-center justify-center flex-shrink-0">
            <Lock className="w-6 h-6 text-brand-teal" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-white mb-1">Privacy-First Architecture</h2>
            <p className="text-sm text-gray-400 leading-relaxed max-w-2xl">
              SentinelEdge processes all sensitive data on your device. The hub server only receives
              differentially-private gradient updates -- mathematical noise that cannot be reverse-engineered
              to reconstruct your conversations or personal information.
            </p>
          </div>
        </div>
      </div>

      {/* Side by side panels */}
      <div className="grid grid-cols-2 gap-6">
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
            {sentences.length === 0 ? (
              <div className="text-center py-12">
                <Smartphone className="w-10 h-10 text-gray-700 mx-auto mb-3" />
                <p className="text-sm text-gray-600">Start a call to see device-side data</p>
                <p className="text-xs text-gray-700 mt-1">Full transcript and analysis visible here</p>
              </div>
            ) : (
              sentences.map((sentence, i) => (
                <div key={i} className="animate-fade-in space-y-2">
                  {/* Transcript text */}
                  <div className="bg-dark-bg/50 rounded-lg p-3 border border-brand-teal/10">
                    <p className="text-xs text-gray-500 mb-1 font-medium">Transcript #{i + 1}</p>
                    <p className="text-sm text-gray-200 font-mono leading-relaxed">
                      {sentence.text}
                    </p>
                  </div>

                  {/* Features + score */}
                  <div className="flex gap-2">
                    <div className="flex-1 bg-dark-bg/30 rounded-lg p-2">
                      <p className="text-[10px] text-gray-600 mb-1">Fraud Score</p>
                      <p className={`text-lg font-bold tabular-nums ${
                        sentence.score < 0.3 ? 'text-safe' :
                        sentence.score < 0.5 ? 'text-warning' :
                        sentence.score < 0.75 ? 'text-orange-400' :
                        'text-alert'
                      }`}>
                        {(sentence.score * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="flex-1 bg-dark-bg/30 rounded-lg p-2">
                      <p className="text-[10px] text-gray-600 mb-1">Features</p>
                      <div className="flex flex-wrap gap-1">
                        {Object.entries(sentence.features)
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

                  {i < sentences.length - 1 && (
                    <div className="border-b border-dark-border/20" />
                  )}
                </div>
              ))
            )}
          </div>
        </div>

        {/* Arrow divider (shown between the panels) */}
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
            {gradientVectors.length === 0 ? (
              <div className="text-center py-12">
                <Server className="w-10 h-10 text-gray-700 mx-auto mb-3" />
                <p className="text-sm text-gray-600">No data transmitted yet</p>
                <p className="text-xs text-gray-700 mt-1">Only noisy gradients are sent</p>
              </div>
            ) : (
              gradientVectors.map((vector, i) => (
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
                      [{vector.map((v, j) => (
                        <span key={j}>
                          <span className={
                            Math.abs(v) > 0.005 ? 'text-gray-400' : 'text-gray-700'
                          }>
                            {v.toFixed(6)}
                          </span>
                          {j < vector.length - 1 ? ', ' : ''}
                        </span>
                      ))}]
                    </div>
                    <div className="mt-2 flex items-center gap-1.5">
                      <Lock className="w-2.5 h-2.5 text-gray-600" />
                      <p className="text-[9px] text-gray-600">
                        Noise scale: epsilon=1.0, delta=1e-5
                      </p>
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
