"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import type { ScoreBoardData, ScoreBoardEntry } from "@/lib/canvas/types";
import { cn } from "@/lib/utils";
import { motion } from "motion/react";

export default function ScoreBoard({ data }: { data: ScoreBoardData }) {
  const accent = data.accentColor || "#2563eb";
  const sorted = useMemo(() => {
    const arr = Array.isArray(data.entries) ? [...data.entries] : [];
    if (data.sort === "asc") arr.sort((a, b) => (a.score ?? 0) - (b.score ?? 0));
    else arr.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
    return arr;
  }, [data.entries, data.sort]);
  const leaderId = sorted[0]?.id ? String(sorted[0].id) : "";
  const prevLeaderRef = useRef<string>(leaderId);
  const [burst, setBurst] = useState(false);
  useEffect(() => {
    if (leaderId && prevLeaderRef.current && leaderId !== prevLeaderRef.current) {
      setBurst(true);
      const t = setTimeout(() => setBurst(false), 1200);
      return () => clearTimeout(t);
    }
    prevLeaderRef.current = leaderId;
  }, [leaderId]);

  return (
    <div
      className={cn(
        "relative rounded-2xl bg-white/70 dark:bg-neutral-900/60",
        "shadow-[0_12px_28px_rgba(0,0,0,0.22)] backdrop-blur-md",
        "ring-1 ring-black/5 dark:ring-white/10",
        "min-w-[18rem] w-[clamp(18rem,26vw,26rem)]"
      )}
    >
      {burst && (
        <div className="pointer-events-none absolute inset-0 overflow-visible z-10">
          {[...Array(16)].map((_, i) => {
            const x = Math.random() * 100; // percent
            const delay = Math.random() * 0.2;
            const rot = Math.random() * 180 - 90;
            const size = 6 + Math.random() * 6;
            const colors = [accent, '#f59e0b', '#22c55e', '#ef4444', '#3b82f6', '#a78bfa'];
            const color = colors[i % colors.length];
            return (
              <motion.div
                key={i}
                initial={{ y: 0, opacity: 1, rotate: 0 }}
                animate={{ y: -80 - Math.random()*40, opacity: 0, rotate: rot }}
                transition={{ duration: 1.1, delay, ease: 'easeOut' }}
                style={{ left: `${x}%`, top: '60%', width: size, height: size, backgroundColor: color }}
                className="absolute rounded-[2px]"
              />
            );
          })}
        </div>
      )}
      {/* Header */}
      <div
        className="px-4 py-3 flex items-center justify-between"
      >
        <div className="font-semibold tracking-wide" style={{ color: accent }}>
          {data.title || "Scoreboard"}
        </div>
        <div className="text-[11px] text-muted-foreground">
          {sorted.length} players
        </div>
      </div>
      {/* List */}
      <div className="p-3">
        {sorted.length === 0 ? (
          <div className="text-sm text-muted-foreground px-1 py-4 text-center">No scores yet.</div>
        ) : (
          <ul className="space-y-1">
            {sorted.map((e: ScoreBoardEntry, idx) => (
              <li
                key={e.id}
                className={cn(
                  "flex items-center justify-between px-3 py-2 rounded-xl",
                  "bg-white/65 dark:bg-neutral-900/40",
                  idx === 0 ? "ring-1" : "",
                )}
                style={{
                  boxShadow: idx === 0 ? `inset 0 0 0 1px ${accent}22` : undefined,
                }}
              >
                <div className="flex items-center gap-2">
                  <div
                    className="inline-flex items-center justify-center w-6 h-6 text-[11px] font-semibold rounded-full"
                    style={{ backgroundColor: `${accent}18`, color: accent }}
                  >
                    {idx + 1}
                  </div>
                  <span className="text-sm font-medium text-foreground/90">{e.name}</span>
                </div>
                <div className="text-sm font-semibold" style={{ color: accent }}>{e.score ?? 0}</div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
