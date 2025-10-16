"use client";

import React from "react";
import type { CoinDisplayData } from "@/lib/canvas/types";
import { cn } from "@/lib/utils";

function Coin({ color }: { color: string }) {
  const c = color || "#f59e0b";
  return (
    <div
      className="w-10 h-10 rounded-full relative shadow-[0_6px_16px_rgba(0,0,0,0.25)]"
      style={{
        background: `radial-gradient(60% 60% at 35% 30%, #fff7e6 0%, ${c} 60%, #8b5a00 100%)`,
      }}
    >
      <div
        className="absolute inset-[2px] rounded-full"
        style={{ boxShadow: `inset 0 0 0 2px ${c}66` }}
      />
    </div>
  );
}

export default function CoinDisplay({ data }: { data: CoinDisplayData }) {
  const color = data.accentColor || "#f59e0b";
  const count = Math.max(0, Number(data.count || 0));
  const isSingle = count <= 1;

  return (
    <div
      className={cn(
        "relative rounded-2xl bg-white/70 dark:bg-neutral-900/60",
        "shadow-[0_10px_26px_rgba(0,0,0,0.22)] backdrop-blur-md",
        "ring-1 ring-black/5 dark:ring-white/10 px-4 py-3",
        "min-w-[12rem] w-[clamp(12rem,18vw,18rem)]"
      )}
    >
      <div className="flex items-center gap-3">
        {isSingle ? (
          <Coin color={color} />
        ) : (
          <div className="relative w-14 h-10">
            <div className="absolute left-0 top-1"><Coin color={color} /></div>
            <div className="absolute left-4 top-0"><Coin color={color} /></div>
            <div className="absolute left-8 top-2"><Coin color={color} /></div>
          </div>
        )}
        <div className="flex-1">
          <div className="text-sm font-semibold text-foreground/80">
            {data.currency || "gold"}
          </div>
          <div className="text-2xl font-extrabold tracking-tight" style={{ color }}>
            {isSingle ? 1 : count}
          </div>
        </div>
      </div>
      {data.showLabel && (
        <div className="mt-1 text-xs text-muted-foreground">{String(data.currency || "gold").toUpperCase()}</div>
      )}
    </div>
  );
}

