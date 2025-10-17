"use client";

import React from "react";
import type { NightOverlayData } from "@/lib/canvas/types";

export default function NightOverlay({ data }: { data: NightOverlayData }) {
  if (!data.visible) return null;
  const opacity = Math.max(0, Math.min(1, typeof data.opacity === 'number' ? data.opacity : 0.5));
  const blur = data.blur ? "backdrop-blur-sm" : "";

  return (
    <div className={`absolute inset-0 z-5 pointer-events-none ${blur}`}>
      <div className="absolute inset-0 pointer-events-none" style={{ backgroundColor: `rgba(0,0,0,${opacity})` }} />
      {(data.title || data.subtitle) && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="px-6 py-4 rounded-2xl bg-white/10 dark:bg-black/20 ring-1 ring-white/10 shadow-[0_8px_24px_rgba(0,0,0,0.25)] pointer-events-none">
            {data.title && (
              <div className="text-white text-xl font-bold text-center drop-shadow-sm">{data.title}</div>
            )}
            {data.subtitle && (
              <div className="text-white/80 text-sm text-center mt-1">{data.subtitle}</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
