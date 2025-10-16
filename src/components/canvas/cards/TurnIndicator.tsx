"use client";

import React from "react";
import type { TurnIndicatorData } from "@/lib/canvas/types";
import { cn } from "@/lib/utils";
import { motion } from "motion/react";

export default function TurnIndicator({ data }: { data: TurnIndicatorData }) {
  const accent = data.accentColor || "#2563eb";
  const name = data.playerName || `Player ${data.currentPlayerId}`;
  const label = data.label || "Turn";
  return (
    <motion.div
      key={`turn-${data.currentPlayerId}`}
      className={cn(
        "inline-flex items-center gap-2 rounded-full",
        "px-4 py-2 shadow-[0_8px_20px_rgba(0,0,0,0.18)]",
        "backdrop-blur-md bg-white/70 dark:bg-neutral-900/60",
        "ring-1 ring-black/5 dark:ring-white/10"
      )}
      initial={{ scale: 1, boxShadow: '0 8px 20px rgba(0,0,0,0.18)' }}
      animate={{ scale: [1, 1.06, 1], boxShadow: [`0 8px 20px rgba(0,0,0,0.18)`, `0 0 0 3px ${accent}55`, `0 8px 20px rgba(0,0,0,0.18)`] }}
      transition={{ duration: 0.45, ease: 'easeOut' }}
    >
      <span
        className="inline-flex w-2.5 h-2.5 rounded-full"
        style={{ backgroundColor: accent }}
      />
      <span className="text-sm font-medium text-foreground/85">{label}:</span>
      <span className="text-sm font-semibold" style={{ color: accent }}>{name}</span>
    </motion.div>
  );
}
