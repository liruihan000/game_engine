"use client";

import React from "react";
import type { InfluenceSetData } from "@/lib/canvas/types";
import { cn } from "@/lib/utils";
import { motion } from "motion/react";

function CardFace({ name, revealed, accent }: { name: string; revealed: boolean; accent: string }) {
  const key = revealed ? `front-${name}` : "back";
  return (
    <motion.div
      key={key}
      initial={{ rotateY: 90, opacity: 0.0, scale: 0.98 }}
      animate={{ rotateY: 0, opacity: 1, scale: 1 }}
      transition={{ duration: 0.35, ease: "easeOut" }}
      className={cn(
        "w-20 h-28 rounded-xl",
        "shadow-[0_10px_24px_rgba(0,0,0,0.25)]",
        revealed ? "backdrop-blur-md bg-white/80 dark:bg-neutral-900/70" : "bg-[linear-gradient(135deg,rgba(255,255,255,0.6),rgba(255,255,255,0.2))]",
        "ring-1 ring-black/5 dark:ring-white/10 flex items-center justify-center px-2 text-center"
      )}
      style={{ boxShadow: `inset 0 0 0 1px ${accent}22, 0 10px 24px rgba(0,0,0,0.25)` }}
    >
      {revealed ? (
        <span className="text-xs font-semibold" style={{ color: accent }}>{name || "?"}</span>
      ) : null}
    </motion.div>
  );
}

export default function InfluenceSet({ data }: { data: InfluenceSetData }) {
  const accent = data.accentColor || "#a78bfa";
  const c = Array.isArray(data.cards) ? data.cards.slice(0, 2) : [];
  const [c1, c2] = [c[0] || { name: "", revealed: false }, c[1] || { name: "", revealed: false }];
  return (
    <div className="inline-flex items-center gap-3">
      <CardFace name={c1.name} revealed={!!c1.revealed} accent={accent} />
      <CardFace name={c2.name} revealed={!!c2.revealed} accent={accent} />
    </div>
  );
}
