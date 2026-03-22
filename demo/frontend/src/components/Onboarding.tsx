import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

const screens = [
  {
    title: "Your calls aren’t always what they seem.",
    subtitle:
      "Americans lost over $10B to phone scams in 2024. Most don’t realize until it’s too late.",
  },
  {
    title: "Real-time scam detection.",
    subtitle:
      "SentinelEdge analyzes calls instantly and warns you before it's too late.",
    custom: true,
  },
  {
    title: "Privacy-first by design.",
    subtitle:
      "Everything stays on your device. Only encrypted updates leave.",
  },
  {
    title: "Try it now.",
    subtitle: "Watch SentinelEdge detect a scam in real time.",
    cta: true,
  },
];

export default function Onboarding({ onFinish }: { onFinish: () => void }) {
  const [index, setIndex] = useState(0);
  const screen = screens[index];

  const next = () => {
    if (index < screens.length - 1) {
      setIndex(index + 1);
    }
  };

  const back = () => {
    if (index > 0) {
      setIndex(index - 1);
    }
  };

  return (
    <div
      onClick={next} // 👈 click anywhere to advance
      className="h-screen w-full relative overflow-hidden bg-black text-white flex flex-col justify-between px-10 py-8 cursor-pointer"
    >
      {/* 🔥 Background Glow */}
      <div className="absolute inset-0 z-0">
        <div className="absolute top-[-200px] left-[-200px] w-[500px] h-[500px] bg-green-500/20 blur-3xl rounded-full animate-pulse" />
        <div className="absolute bottom-[-200px] right-[-200px] w-[500px] h-[500px] bg-blue-500/20 blur-3xl rounded-full animate-pulse" />
      </div>

      {/* Skip */}
      <div className="flex justify-end z-10">
        <button
          onClick={(e) => {
            e.stopPropagation();
            onFinish();
          }}
          className="text-gray-400 hover:text-white transition"
        >
          Skip
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 flex items-center justify-center z-10">
        <AnimatePresence mode="wait">
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 60 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -60 }}
            transition={{ duration: 0.5 }}
            className="max-w-5xl text-center space-y-10"
          >
            {/* Title */}
            <h1 className="text-7xl md:text-8xl font-bold leading-tight tracking-tight bg-gradient-to-r from-green-400 to-blue-500 bg-clip-text text-transparent">
              {screen.title}
            </h1>

            {/* Subtitle */}
            <p className="text-3xl md:text-4xl text-gray-300 max-w-3xl mx-auto">
              {screen.subtitle}
            </p>

            {/* Pipeline */}
            {screen.custom && (
              <div className="mt-12 flex flex-col items-center gap-4 text-2xl md:text-3xl">
                <div>Incoming Call</div>
                <div className="opacity-50">↓</div>
                <div>AI Analysis</div>
                <div className="opacity-50">↓</div>
                <div className="text-red-400 font-semibold">
                  ⚠️ Scam Detected (3–7s)
                </div>
              </div>
            )}

            {/* CTA */}
            {screen.cta && (
              <button
                onClick={(e) => {
                  e.stopPropagation(); // 🚨 prevent click-through
                  onFinish();
                }}
                className="mt-12 bg-green-500 hover:bg-green-600 text-black font-semibold text-2xl px-14 py-6 rounded-2xl shadow-[0_0_50px_rgba(34,197,94,0.5)] transition"
              >
                 Start Demo
              </button>
            )}
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Footer */}
      <div className="flex justify-between items-center z-10">
        {/* Back button */}
        <button
          onClick={(e) => {
            e.stopPropagation();
            back();
          }}
          className="text-lg text-gray-400 hover:text-white"
        >
          ← Back
        </button>

        {/* Dots */}
        <div className="flex gap-3">
          {screens.map((_, i) => (
            <div
              key={i}
              className={`w-3 h-3 rounded-full ${
                i === index ? "bg-white scale-125" : "bg-gray-600"
              }`}
            />
          ))}
        </div>

        {/* Invisible spacer for symmetry */}
        <div className="w-[80px]" />
      </div>
    </div>
  );
}