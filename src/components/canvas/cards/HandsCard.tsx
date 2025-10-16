"use client";

import React from "react";
import type { HandsCardData } from "@/lib/canvas/types";
import { cn } from "@/lib/utils";

export function HandsCard({ data, title, subtitle }: { data: HandsCardData; title?: string; subtitle?: string; }) {
  const accent = data.color || "#111827"; // default neutral-900

  return (
    <div
      className={cn(
        "relative rounded-2xl bg-white/70 dark:bg-neutral-900/60",
        "shadow-[0_10px_30px_rgba(0,0,0,0.22)] backdrop-blur-md",
        "ring-1 ring-black/5 dark:ring-white/10",
        "min-w-56 max-w-80 w-[clamp(14rem,22vw,20rem)]",
      )}
    >
      {/* top bar */}
      <div
        className="px-4 py-2"
        style={{
          background:
            "linear-gradient(135deg, rgba(255,255,255,0.35), rgba(255,255,255,0.08))",
        }}
      >
        <div className="flex items-center justify-between gap-2">
          <span
            className="inline-flex items-center px-2 py-0.5 text-[11px] font-semibold rounded-full"
            style={{ backgroundColor: `${accent}15`, color: accent }}
          >
            {data.cardType || "card"}
          </span>
          {subtitle && (
            <span className="text-[11px] text-muted-foreground truncate max-w-[40%]">{subtitle}</span>
          )}
        </div>
      </div>

      {/* content */}
      <div className="p-4">
        <div
          className="font-bold text-[1.05rem] tracking-wide mb-2"
          style={{ color: accent }}
        >
          {data.cardName || title || "Card"}
        </div>
        <div className="text-sm text-foreground/80 leading-relaxed whitespace-pre-wrap">
          {data.descriptions || ""}
        </div>
      </div>

      {/* subtle glow */}
      <div
        className="pointer-events-none absolute inset-0 rounded-2xl"
        style={{ boxShadow: `inset 0 0 0 1px ${accent}20` }}
      />
    </div>
  );
}

export default HandsCard;
