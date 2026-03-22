import { ReactNode } from 'react'

interface PhoneSimulatorProps {
  children: ReactNode
}

export default function PhoneSimulator({ children }: PhoneSimulatorProps) {
  return (
    <div className="phone-frame">
      <div className="relative w-[290px] h-[600px] rounded-[46px] bg-[#111215] p-[4px] shadow-2xl shadow-black/60">
        <div className="pointer-events-none absolute inset-[1px] rounded-[45px] border border-white/10" />
        <div className="pointer-events-none absolute inset-x-[18px] top-[10px] h-[22px] rounded-full bg-white/5 blur-md" />

        <div className="relative h-full w-full overflow-hidden rounded-[42px] bg-gradient-to-b from-[#2a2b31] via-[#111216] to-[#050608] p-[7px]">
          <div className="absolute -left-[3px] top-[116px] h-[34px] w-[3px] rounded-l-sm bg-zinc-500/70" />
          <div className="absolute -left-[3px] top-[172px] h-[58px] w-[3px] rounded-l-sm bg-zinc-500/70" />
          <div className="absolute -left-[3px] top-[240px] h-[58px] w-[3px] rounded-l-sm bg-zinc-500/70" />
          <div className="absolute -right-[3px] top-[182px] h-[78px] w-[3px] rounded-r-sm bg-zinc-500/70" />

          <div className="relative h-full w-full overflow-hidden rounded-[35px] border border-black/70 bg-dark-bg shadow-[inset_0_0_0_1px_rgba(255,255,255,0.03)]">
            <div className="pointer-events-none absolute inset-0 rounded-[35px] shadow-[inset_0_1px_0_rgba(255,255,255,0.06),inset_0_-18px_40px_rgba(0,0,0,0.32)]" />

            <div className="absolute left-1/2 top-[7px] z-30 h-[34px] w-[124px] -translate-x-1/2 rounded-[20px] bg-black shadow-[0_4px_14px_rgba(0,0,0,0.55)]">
              <div className="absolute left-[18px] top-1/2 h-[10px] w-[10px] -translate-y-1/2 rounded-full bg-zinc-900 ring-1 ring-white/5" />
              <div className="absolute left-[35px] top-1/2 h-[7px] w-[7px] -translate-y-1/2 rounded-full bg-zinc-800 ring-1 ring-white/5" />
              <div className="absolute right-[18px] top-1/2 h-[8px] w-[40px] -translate-y-1/2 rounded-full bg-zinc-950/90" />
            </div>

            <div className="relative h-full w-full overflow-hidden rounded-[35px]">
              {children}
            </div>

            <div className="absolute bottom-1.5 left-0 right-0 z-30 flex justify-center">
              <div className="h-[4px] w-[118px] rounded-full bg-white/38" />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
