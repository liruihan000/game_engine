"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

export interface BroadcastInputProps {
  open: boolean;
  title?: string;
  placeholder?: string;
  confirmLabel?: string;
  initialValue?: string;
  onConfirm: (text: string) => void;
  onClose: () => void;
}

export default function BroadcastInput({
  open,
  title = "Broadcast",
  placeholder = "Type a broadcast message...",
  confirmLabel = "Send",
  initialValue = "",
  onConfirm,
  onClose,
}: BroadcastInputProps) {
  const [value, setValue] = useState<string>(initialValue);
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (open) {
      setValue(initialValue || "");
      // focus after mount
      const id = setTimeout(() => inputRef.current?.focus(), 50);
      return () => clearTimeout(id);
    }
  }, [open, initialValue]);

  const handleSubmit = useCallback(
    (e?: React.FormEvent) => {
      if (e) e.preventDefault();
      const text = (value || "").trim();
      if (!text) return;
      onConfirm(text);
      setValue("");
    },
    [value, onConfirm]
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        onClose();
      }
      if ((e.key === "Enter" || e.key === "NumpadEnter") && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit, onClose]
  );

  if (!open) return null;

  return (
    <div
      className={cn(
        "fixed left-1/2 -translate-x-1/2 bottom-6 z-50",
        "pointer-events-auto"
      )}
      aria-live="polite"
      aria-label="Broadcast input"
    >
      <form
        onSubmit={handleSubmit}
        className={cn(
          "flex items-center gap-2",
          // floating pill container
          "rounded-full shadow-2xl",
          "backdrop-blur supports-[backdrop-filter]:bg-white/60 supports-not-[backdrop-filter]:bg-white",
          "dark:supports-[backdrop-filter]:bg-black/40 dark:supports-not-[backdrop-filter]:bg-neutral-900",
          "border border-black/10 dark:border-white/10",
          "px-4 py-2"
        )}
        role="group"
      >
        <div className="flex flex-col">
          <span className="text-[11px] font-medium text-muted-foreground leading-none pl-1 pb-1">
            {title}
          </span>
          <input
            ref={inputRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            className={cn(
              "w-[56vw] max-w-[720px] min-w-[280px]",
              "bg-transparent text-sm outline-none",
              "px-2 py-1 rounded-full",
              "placeholder:text-gray-400 dark:placeholder:text-gray-500"
            )}
          />
        </div>
        <button
          type="submit"
          className={cn(
            "inline-flex items-center justify-center",
            "rounded-full px-4 h-8 text-sm font-semibold",
            "bg-blue-600 text-white hover:bg-blue-700",
            "shadow-md"
          )}
          aria-label={confirmLabel}
        >
          {confirmLabel}
        </button>
      </form>
    </div>
  );
}

