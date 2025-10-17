"use client";

import React from "react";
import type { StatementBoardData } from "@/lib/canvas/types";
import { cn } from "@/lib/utils";

export default function StatementBoard({ data }: { data: StatementBoardData }) {
  const accent = data.accentColor || "#2563eb";
  const items = Array.isArray(data.statements) ? data.statements.slice(0, 3) : [];
  const highlight = typeof data.highlightIndex === 'number' ? data.highlightIndex : -1;

  return (
    <div
      className={cn(
        "relative rounded-2xl bg-white/70 dark:bg-neutral-900/60",
        "shadow-[0_10px_26px_rgba(0,0,0,0.22)] backdrop-blur-md",
        "ring-1 ring-black/5 dark:ring-white/10 px-5 py-4",
        "min-w-[18rem] w-[clamp(18rem,28vw,28rem)]"
      )}
    >
      <div className="space-y-2">
        {items.map((text, idx) => {
          const active = idx === highlight;
          return (
            <div
              key={idx}
              className={cn(
                "flex items-start gap-3 px-3 py-2 rounded-xl",
                "bg-white/65 dark:bg-neutral-900/45",
                active ? "ring-1" : ""
              )}
              style={{ boxShadow: active ? `inset 0 0 0 1px ${accent}22` : undefined }}
            >
              <div
                className="inline-flex items-center justify-center w-6 h-6 text-[11px] font-semibold rounded-full mt-0.5"
                style={{ backgroundColor: `${accent}18`, color: accent }}
              >
                {idx + 1}
              </div>
              <div className={cn("text-sm text-foreground/90", data.locked ? "opacity-80" : "")}>{text || <span className="opacity-60">(empty)</span>}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

