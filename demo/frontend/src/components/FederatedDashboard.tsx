import { useState, useCallback, useEffect, useRef } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  Area,
  AreaChart,
} from 'recharts'
import {
  GitBranch,
  Cpu,
  Shield,
  Lock,
  Play,
  RefreshCw,
  TrendingUp,
  Users,
  Database,
  Smartphone,
  WifiOff,
} from 'lucide-react'

// -------- Mock data generators (used as fallback when hub is offline) --------
function generateRoundData(numRounds: number) {
  const data = []
  let accuracy = 0.62
  let f1 = 0.55
  let loss = 0.82

  for (let i = 1; i <= numRounds; i++) {
    const accGain = (0.98 - accuracy) * 0.08 + (Math.random() - 0.5) * 0.015
    const f1Gain = (0.95 - f1) * 0.09 + (Math.random() - 0.5) * 0.02
    const lossDecay = loss * 0.06 + (Math.random() - 0.5) * 0.02

    accuracy = Math.min(0.98, accuracy + accGain)
    f1 = Math.min(0.95, f1 + f1Gain)
    loss = Math.max(0.05, loss - lossDecay)

    data.push({
      round: i,
      accuracy: Number(accuracy.toFixed(4)),
      f1: Number(f1.toFixed(4)),
      loss: Number(loss.toFixed(4)),
    })
  }
  return data
}

function generateDeviceData() {
  const devices = [
    { name: 'Device A', samples: 342, contribution: 0.23 },
    { name: 'Device B', samples: 287, contribution: 0.19 },
    { name: 'Device C', samples: 198, contribution: 0.14 },
    { name: 'Device D', samples: 456, contribution: 0.31 },
    { name: 'Device E', samples: 167, contribution: 0.13 },
  ]
  return devices
}

// Generate a random gradient delta for simulated FL submissions
function generateGradientDelta(size = 64): number[] {
  return Array.from({ length: size }, () =>
    Number(((Math.random() - 0.5) * 0.01).toFixed(6))
  )
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-dark-card border border-dark-border rounded-lg px-3 py-2 shadow-xl">
        <p className="text-xs text-gray-400 mb-1">Round {label}</p>
        {payload.map((entry: any, idx: number) => (
          <p key={idx} className="text-xs font-mono" style={{ color: entry.color }}>
            {entry.name}: {(entry.value * 100).toFixed(1)}%
          </p>
        ))}
      </div>
    )
  }
  return null
}

export default function FederatedDashboard() {
  const [roundData, setRoundData] = useState(() => generateRoundData(20))
  const [deviceData, setDeviceData] = useState(() => generateDeviceData())
  const [isSimulating, setIsSimulating] = useState(false)
  const [currentRound, setCurrentRound] = useState(20)
  const [hubConnected, setHubConnected] = useState(false)
  const [totalSamples, setTotalSamples] = useState(1450)
  const [privacyBudget, setPrivacyBudget] = useState('epsilon=1.0')

  const pollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const isMountedRef = useRef(true)

  // Fetch global metrics from the hub API
  const fetchGlobalMetrics = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:8080/v1/metrics/global', {
        signal: AbortSignal.timeout(3000),
      })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const data = await response.json()

      if (!isMountedRef.current) return

      // Map hub API response to our local state
      if (data.accuracy_history && Array.isArray(data.accuracy_history)) {
        const mapped = data.accuracy_history.map((entry: any, i: number) => ({
          round: entry.round ?? i + 1,
          accuracy: entry.accuracy ?? 0,
          f1: entry.f1 ?? 0,
          loss: entry.loss ?? 0,
        }))
        if (mapped.length > 0) {
          setRoundData(mapped)
          setCurrentRound(mapped.length)
        }
      }
      if (data.device_count != null) {
        // Update device count visually -- keep existing device data shape
        const count = Number(data.device_count)
        if (count > 0 && count !== deviceData.length) {
          const generated = Array.from({ length: count }, (_, i) => ({
            name: `Device ${String.fromCharCode(65 + i)}`,
            samples: Math.floor(Math.random() * 400) + 100,
            contribution: Number((1 / count).toFixed(2)),
          }))
          setDeviceData(generated)
        }
      }
      if (data.round_count != null) {
        setCurrentRound(Number(data.round_count))
      }
      if (data.total_samples != null) {
        setTotalSamples(Number(data.total_samples))
      }
      if (data.privacy_budget != null) {
        setPrivacyBudget(String(data.privacy_budget))
      }

      setHubConnected(true)
    } catch {
      if (isMountedRef.current) {
        setHubConnected(false)
      }
    }
  }, [deviceData.length])

  // Fetch round status (current round progress)
  const fetchRoundStatus = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:8080/v1/round/status', {
        signal: AbortSignal.timeout(3000),
      })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const data = await response.json()

      if (!isMountedRef.current) return

      if (data.current_round != null) {
        setCurrentRound(Number(data.current_round))
      }
      if (data.is_aggregating != null) {
        setIsSimulating(Boolean(data.is_aggregating))
      }
    } catch {
      // Silently fail -- global metrics fetch already handles connection status
    }
  }, [])

  // Poll hub API every 5 seconds while tab is active
  useEffect(() => {
    isMountedRef.current = true

    // Initial fetch
    fetchGlobalMetrics()
    fetchRoundStatus()

    const startPolling = () => {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current)
      pollIntervalRef.current = setInterval(() => {
        fetchGlobalMetrics()
        fetchRoundStatus()
      }, 5000)
    }

    const stopPolling = () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current)
        pollIntervalRef.current = null
      }
    }

    const handleVisibilityChange = () => {
      if (document.hidden) {
        stopPolling()
      } else {
        fetchGlobalMetrics()
        fetchRoundStatus()
        startPolling()
      }
    }

    startPolling()
    document.addEventListener('visibilitychange', handleVisibilityChange)

    return () => {
      isMountedRef.current = false
      stopPolling()
      document.removeEventListener('visibilitychange', handleVisibilityChange)
    }
  }, [fetchGlobalMetrics, fetchRoundStatus])

  const latestMetrics = roundData[roundData.length - 1]

  // Run FL Round: POST to hub if connected, else simulate locally
  const runSimulationRound = useCallback(async () => {
    setIsSimulating(true)

    if (hubConnected) {
      try {
        const payload = {
          device_id: `device_${Math.random().toString(36).slice(2, 8)}`,
          gradient_delta: generateGradientDelta(),
          num_samples: Math.floor(Math.random() * 200) + 50,
          round: currentRound + 1,
        }
        const response = await fetch('http://localhost:8080/v1/federated/submit', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
          signal: AbortSignal.timeout(5000),
        })

        if (response.ok) {
          // After successful submit, re-fetch to pick up the new round data
          await fetchGlobalMetrics()
          await fetchRoundStatus()
          setIsSimulating(false)
          return
        }
      } catch {
        // Hub became unreachable during the request -- fall through to local simulation
        setHubConnected(false)
      }
    }

    // Fallback: local simulation
    const timer = setTimeout(() => {
      setRoundData(prev => {
        const last = prev[prev.length - 1]
        const newRound = {
          round: last.round + 1,
          accuracy: Number(Math.min(0.98, last.accuracy + (0.98 - last.accuracy) * 0.08 + (Math.random() - 0.5) * 0.01).toFixed(4)),
          f1: Number(Math.min(0.95, last.f1 + (0.95 - last.f1) * 0.09 + (Math.random() - 0.5) * 0.015).toFixed(4)),
          loss: Number(Math.max(0.05, last.loss - last.loss * 0.06 + (Math.random() - 0.5) * 0.01).toFixed(4)),
        }
        return [...prev, newRound]
      })
      setCurrentRound(prev => prev + 1)
      setIsSimulating(false)
    }, 1500)

    return () => clearTimeout(timer)
  }, [hubConnected, currentRound, fetchGlobalMetrics, fetchRoundStatus])

  const stats = [
    {
      label: 'Model Version',
      value: `v${currentRound}`,
      icon: <Cpu className="w-4 h-4" />,
      color: 'text-brand-teal',
      bgColor: 'bg-brand-teal/10',
    },
    {
      label: 'Active Devices',
      value: String(deviceData.length),
      icon: <Users className="w-4 h-4" />,
      color: 'text-blue-400',
      bgColor: 'bg-blue-400/10',
    },
    {
      label: 'Total Samples',
      value: totalSamples.toLocaleString(),
      icon: <Database className="w-4 h-4" />,
      color: 'text-purple-400',
      bgColor: 'bg-purple-400/10',
    },
    {
      label: 'Privacy Budget',
      value: privacyBudget,
      icon: <Lock className="w-4 h-4" />,
      color: 'text-safe',
      bgColor: 'bg-safe/10',
    },
    {
      label: 'Accuracy',
      value: `${(latestMetrics.accuracy * 100).toFixed(1)}%`,
      icon: <TrendingUp className="w-4 h-4" />,
      color: 'text-brand-teal',
      bgColor: 'bg-brand-teal/10',
    },
    {
      label: 'F1 Score',
      value: `${(latestMetrics.f1 * 100).toFixed(1)}%`,
      icon: <Shield className="w-4 h-4" />,
      color: 'text-warning',
      bgColor: 'bg-warning/10',
    },
  ]

  return (
    <div className="space-y-6">
      {/* Offline banner */}
      {!hubConnected && (
        <div className="flex items-center gap-3 px-4 py-3 rounded-xl bg-warning/10 border border-warning/20">
          <WifiOff className="w-4 h-4 text-warning flex-shrink-0" />
          <p className="text-xs text-warning font-medium">
            Hub offline &mdash; showing simulated data
          </p>
        </div>
      )}

      {/* Header with action */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-brand-teal/10 flex items-center justify-center">
            <GitBranch className="w-5 h-5 text-brand-teal" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-lg font-bold text-white">Federated Learning Dashboard</h2>
              {/* Connection status indicator */}
              <span
                className={`inline-block w-2.5 h-2.5 rounded-full ${
                  hubConnected ? 'bg-safe animate-pulse' : 'bg-gray-500'
                }`}
                title={hubConnected ? 'Hub connected' : 'Hub offline'}
              />
            </div>
            <p className="text-xs text-gray-500">Model training across distributed edge devices</p>
          </div>
        </div>
        <button
          onClick={runSimulationRound}
          disabled={isSimulating}
          className={`
            flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all
            ${isSimulating
              ? 'bg-brand-teal/20 text-brand-teal/50 cursor-not-allowed'
              : 'bg-brand-teal text-white hover:bg-brand-teal-light active:scale-[0.98] shadow-lg shadow-brand-teal/20'
            }
          `}
        >
          {isSimulating ? (
            <RefreshCw className="w-4 h-4 animate-spin" />
          ) : (
            <Play className="w-4 h-4" />
          )}
          {isSimulating ? 'Aggregating...' : 'Run FL Round'}
        </button>
      </div>

      {/* Stats cards */}
      <div className="grid grid-cols-6 gap-4">
        {stats.map((stat) => (
          <div key={stat.label} className="glass-card p-4">
            <div className="flex items-center gap-2 mb-2">
              <div className={`w-7 h-7 rounded-lg ${stat.bgColor} flex items-center justify-center ${stat.color}`}>
                {stat.icon}
              </div>
            </div>
            <p className={`text-lg font-bold ${stat.color} tabular-nums`}>{stat.value}</p>
            <p className="text-[10px] text-gray-500 mt-0.5">{stat.label}</p>
          </div>
        ))}
      </div>

      {/* Charts */}
      <div className="grid grid-cols-2 gap-6">
        {/* Accuracy & F1 over rounds */}
        <div className="glass-card p-6">
          <h3 className="text-sm font-semibold text-gray-200 mb-4 flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-brand-teal" />
            Model Performance Over Rounds
          </h3>
          <ResponsiveContainer width="100%" height={280}>
            <AreaChart data={roundData}>
              <defs>
                <linearGradient id="gradAccuracy" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#0D9488" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#0D9488" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="gradF1" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#F59E0B" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#F59E0B" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
              <XAxis
                dataKey="round"
                tick={{ fontSize: 10, fill: '#64748B' }}
                axisLine={{ stroke: '#334155' }}
                tickLine={false}
              />
              <YAxis
                domain={[0.5, 1]}
                tick={{ fontSize: 10, fill: '#64748B' }}
                axisLine={{ stroke: '#334155' }}
                tickLine={false}
                tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend
                wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }}
              />
              <Area
                type="monotone"
                dataKey="accuracy"
                name="Accuracy"
                stroke="#0D9488"
                strokeWidth={2}
                fill="url(#gradAccuracy)"
                dot={false}
                activeDot={{ r: 4, strokeWidth: 0, fill: '#0D9488' }}
              />
              <Area
                type="monotone"
                dataKey="f1"
                name="F1 Score"
                stroke="#F59E0B"
                strokeWidth={2}
                fill="url(#gradF1)"
                dot={false}
                activeDot={{ r: 4, strokeWidth: 0, fill: '#F59E0B' }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Loss over rounds */}
        <div className="glass-card p-6">
          <h3 className="text-sm font-semibold text-gray-200 mb-4 flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-alert" />
            Training Loss
          </h3>
          <ResponsiveContainer width="100%" height={280}>
            <AreaChart data={roundData}>
              <defs>
                <linearGradient id="gradLoss" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#EF4444" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#EF4444" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
              <XAxis
                dataKey="round"
                tick={{ fontSize: 10, fill: '#64748B' }}
                axisLine={{ stroke: '#334155' }}
                tickLine={false}
              />
              <YAxis
                tick={{ fontSize: 10, fill: '#64748B' }}
                axisLine={{ stroke: '#334155' }}
                tickLine={false}
                tickFormatter={(v: number) => v.toFixed(2)}
              />
              <Tooltip content={<CustomTooltip />} />
              <Area
                type="monotone"
                dataKey="loss"
                name="Loss"
                stroke="#EF4444"
                strokeWidth={2}
                fill="url(#gradLoss)"
                dot={false}
                activeDot={{ r: 4, strokeWidth: 0, fill: '#EF4444' }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Device contributions */}
      <div className="glass-card p-6">
        <h3 className="text-sm font-semibold text-gray-200 mb-4 flex items-center gap-2">
          <Users className="w-4 h-4 text-brand-teal" />
          Per-Device Sample Contributions
        </h3>
        <div className="grid grid-cols-2 gap-6">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={deviceData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" horizontal={false} />
              <XAxis
                type="number"
                tick={{ fontSize: 10, fill: '#64748B' }}
                axisLine={{ stroke: '#334155' }}
                tickLine={false}
              />
              <YAxis
                dataKey="name"
                type="category"
                tick={{ fontSize: 11, fill: '#94A3B8' }}
                axisLine={{ stroke: '#334155' }}
                tickLine={false}
                width={70}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1E293B',
                  border: '1px solid #334155',
                  borderRadius: '8px',
                  fontSize: '12px',
                }}
              />
              <Bar
                dataKey="samples"
                name="Samples"
                fill="#0D9488"
                radius={[0, 4, 4, 0]}
                barSize={20}
              />
            </BarChart>
          </ResponsiveContainer>

          {/* Device details table */}
          <div>
            <table className="w-full">
              <thead>
                <tr className="border-b border-dark-border/30">
                  <th className="text-left text-[10px] text-gray-500 font-medium pb-2 uppercase tracking-wider">Device</th>
                  <th className="text-right text-[10px] text-gray-500 font-medium pb-2 uppercase tracking-wider">Samples</th>
                  <th className="text-right text-[10px] text-gray-500 font-medium pb-2 uppercase tracking-wider">Contribution</th>
                  <th className="text-right text-[10px] text-gray-500 font-medium pb-2 uppercase tracking-wider">Status</th>
                </tr>
              </thead>
              <tbody>
                {deviceData.map((device) => (
                  <tr key={device.name} className="border-b border-dark-border/10">
                    <td className="py-2.5 text-sm text-gray-300">{device.name}</td>
                    <td className="py-2.5 text-sm text-gray-400 text-right font-mono tabular-nums">{device.samples}</td>
                    <td className="py-2.5 text-sm text-brand-teal text-right font-mono tabular-nums">
                      {(device.contribution * 100).toFixed(0)}%
                    </td>
                    <td className="py-2.5 text-right">
                      <span className="inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full bg-safe/10 text-safe">
                        <div className="w-1 h-1 rounded-full bg-safe animate-pulse" />
                        Active
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* FL Process explanation */}
      <div className="glass-card p-6">
        <h3 className="text-sm font-semibold text-gray-200 mb-4">How Federated Learning Works</h3>
        <div className="grid grid-cols-4 gap-4">
          {[
            { step: '1', title: 'Local Training', desc: 'Each device trains on its own call data', icon: <Smartphone className="w-5 h-5" />, color: 'text-brand-teal' },
            { step: '2', title: 'Add DP Noise', desc: 'Differential privacy noise added to gradients', icon: <Lock className="w-5 h-5" />, color: 'text-safe' },
            { step: '3', title: 'Aggregate', desc: 'Hub averages noisy gradients from all devices', icon: <GitBranch className="w-5 h-5" />, color: 'text-warning' },
            { step: '4', title: 'Update Model', desc: 'Improved model pushed back to all devices', icon: <RefreshCw className="w-5 h-5" />, color: 'text-purple-400' },
          ].map((item) => (
            <div key={item.step} className="text-center">
              <div className={`w-12 h-12 rounded-xl bg-dark-bg mx-auto mb-3 flex items-center justify-center ${item.color}`}>
                {item.icon}
              </div>
              <div className="text-[10px] text-gray-600 mb-1">Step {item.step}</div>
              <h4 className="text-xs font-semibold text-gray-300 mb-1">{item.title}</h4>
              <p className="text-[10px] text-gray-500 leading-relaxed">{item.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
