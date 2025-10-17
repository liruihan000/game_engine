"use client";

import React, { useEffect, useMemo, useState } from "react";
import type { ReactionTimerData } from "@/lib/canvas/types";
import { cn } from "@/lib/utils";

function computeProgress(duration: number, startedAt?: number, running?: boolean): number {
  if (!running || !startedAt || duration <= 0) return 0;
  const elapsed = Math.max(0, Date.now() - startedAt) / 1000;
  return Math.min(1, elapsed / duration);
}

export default function ReactionTimer({ data }: { data: ReactionTimerData }) {
  const accent = data.accentColor || "#22c55e";
  const [tick, setTick] = useState(0);
  const duration = Math.max(0.1, Number(data.duration || 0));

  useEffect(() => {
    if (!data.running) return;
    const id = setInterval(() => setTick((v) => v + 1), 200);
    return () => clearInterval(id);
  }, [data.running]);

  const progress = useMemo(() => computeProgress(duration, data.startedAt, data.running), [duration, data.startedAt, data.running, tick]);
  const remaining = Math.max(0, Math.ceil(duration - progress * duration));

  return (
    <div
      className={cn(
        "relative rounded-2xl bg-white/70 dark:bg-neutral-900/60",
        "shadow-[0_8px_20px_rgba(0,0,0,0.18)] backdrop-blur-md",
        "ring-1 ring-black/5 dark:ring-white/10 px-4 py-3",
        "min-w-[14rem] w-[clamp(14rem,24vw,24rem)]"
      )}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="text-sm font-semibold text-foreground/85">{data.label || "Reaction Window"}</div>
        <div className="text-xs font-mono" style={{ color: accent }}>{remaining}s</div>
      </div>
      <div className="h-2.5 w-full rounded-full bg-black/10 dark:bg-white/10 overflow-hidden">
        <div
          className="h-full rounded-full"
          style={{ width: `${Math.round(progress * 100)}%`, background: `linear-gradient(90deg, ${accent}, ${accent}AA)` }}
        />
      </div>
    </div>
  );
}

