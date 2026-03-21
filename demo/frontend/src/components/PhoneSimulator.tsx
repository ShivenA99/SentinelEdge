import { ReactNode } from 'react'

interface PhoneSimulatorProps {
  children: ReactNode
}

export default function PhoneSimulator({ children }: PhoneSimulatorProps) {
  return (
    <div className="phone-frame">
      {/* Outer bezel */}
      <div className="relative w-[290px] h-[600px] bg-gradient-to-b from-gray-800 to-gray-900 rounded-[44px] p-[3px] shadow-2xl shadow-black/50">
        {/* Inner bezel with subtle highlight */}
        <div className="relative w-full h-full bg-gradient-to-b from-gray-900 via-black to-gray-900 rounded-[42px] p-[8px] overflow-hidden">
          {/* Side buttons */}
          <div className="absolute -left-[3px] top-[100px] w-[3px] h-[28px] bg-gray-700 rounded-l-sm" />
          <div className="absolute -left-[3px] top-[145px] w-[3px] h-[50px] bg-gray-700 rounded-l-sm" />
          <div className="absolute -left-[3px] top-[205px] w-[3px] h-[50px] bg-gray-700 rounded-l-sm" />
          <div className="absolute -right-[3px] top-[140px] w-[3px] h-[65px] bg-gray-700 rounded-r-sm" />

          {/* Screen area */}
          <div className="relative w-full h-full bg-dark-bg rounded-[36px] overflow-hidden">
            {/* Dynamic Island / Notch */}
            <div className="absolute top-0 left-0 right-0 z-30 flex justify-center pt-2">
              <div className="w-[100px] h-[28px] bg-black rounded-full flex items-center justify-center gap-2">
                <div className="w-[8px] h-[8px] rounded-full bg-gray-800 ring-1 ring-gray-700" />
                <div className="w-[4px] h-[4px] rounded-full bg-gray-700" />
              </div>
            </div>

            {/* Screen content */}
            <div className="relative w-full h-full overflow-hidden">
              {children}
            </div>

            {/* Home indicator */}
            <div className="absolute bottom-1.5 left-0 right-0 z-30 flex justify-center">
              <div className="w-[120px] h-[4px] bg-gray-600 rounded-full opacity-60" />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
