"use client";

import React, { useEffect, useRef } from "react";
import type { HealthDisplayData } from "@/lib/canvas/types";
import { cn } from "@/lib/utils";
import { motion } from "motion/react";

export default function HealthDisplay({ data }: { data: HealthDisplayData }) {
  const accent = data.accentColor || "#ef4444";
  const value = Math.max(0, Number(data.value || 0));
  const max = Math.max(value, Number(data.max || value));
  const style = (data.style || "hearts") as "hearts" | "bullets";
  const prevRef = useRef<number>(value);

  // re-trigger animation when value changes
  const pulseKey = `${style}-${value}-${max}`;
  useEffect(() => {
    prevRef.current = value;
  }, [value]);

  return (
    <motion.div
      key={pulseKey}
      className={cn(
        "inline-flex items-center gap-2 rounded-full",
        "px-4 py-2 shadow-[0_8px_20px_rgba(0,0,0,0.18)]",
        "backdrop-blur-md bg-white/70 dark:bg-neutral-900/60",
        "ring-1 ring-black/5 dark:ring-white/10"
      )}
      initial={{ scale: 1 }}
      animate={{ scale: [1, 1.06, 1] }}
      transition={{ duration: 0.35, ease: "easeOut" }}
    >
      <span className="text-sm font-semibold" style={{ color: accent }}>
        {style === "hearts" ? "♥" : "•"}
      </span>
      {style === "hearts" ? (
        <div className="flex items-center gap-1">
          {Array.from({ length: Math.min(max, 8) }).map((_, i) => (
            <span key={i} className="text-sm" style={{ color: i < value ? accent : "#d1d5db" }}>♥</span>
          ))}
        </div>
      ) : (
        <div className="text-sm font-semibold" style={{ color: accent }}>{value}{max ? `/${max}` : ""}</div>
      )}
    </motion.div>
  );
}
