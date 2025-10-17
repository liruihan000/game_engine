"use client";

import { useState, useEffect } from "react";
import type { TimerData } from "@/lib/canvas/types";

interface TimerProps {
  data: TimerData;
}

export default function Timer({ data }: TimerProps) {
  const [remain, setRemain] = useState<number>(() => 
    typeof window !== 'undefined' ? Math.max(0, Number(data.duration || 0)) : 0
  );
  const [mounted, setMounted] = useState(false);
  
  useEffect(() => {
    setMounted(true);
  }, []);
  
  useEffect(() => {
    if (!mounted) return;
    setRemain(Math.max(0, Number(data.duration || 0)));
    if (!data.duration || data.duration <= 0) return;
    const started = Date.now();
    const id = setInterval(() => {
      const elapsed = Math.floor((Date.now() - started) / 1000);
      const next = Math.max(0, (data.duration as number) - elapsed);
      setRemain(next);
      if (next <= 0) {
        clearInterval(id);
      }
    }, 250);
    return () => clearInterval(id);
  }, [data.duration, mounted]);

  const m = Math.floor(remain / 60);
  const s = String(remain % 60).padStart(2, '0');

  return (
    <div className="bg-card border border-border rounded-lg p-3 shadow-sm min-w-[120px]">
      {data.label && (
        <div className="text-xs font-medium mb-1 text-center text-muted-foreground">
          {data.label}
        </div>
      )}
      <div className="text-lg font-mono font-bold text-center tracking-wide text-foreground">
        {m}:{s}
      </div>
      {remain === 0 && (
        <div className="text-xs text-red-500 text-center mt-1">
          Time&apos;s up!
        </div>
      )}
    </div>
  );
}