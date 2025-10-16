"use client";

import { Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";
import type { CardType } from "@/lib/canvas/types";

export function NewItemMenu({ onSelect, align = "end", className }: { onSelect: (t: CardType) => void; align?: "start" | "end" | "center", className?: string }) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="default" className={cn("gap-2 text-base font-semibold bg-card rounded-lg",
          className)}>
          <Plus className="size-5" />
          New
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align={align} className="w-[400px] bg-background p-2">
        <div className="grid grid-cols-2 gap-1">
          <DropdownMenuItem onClick={() => onSelect("character_card")} className="justify-start">Character Card</DropdownMenuItem>
          <DropdownMenuItem onClick={() => onSelect("action_button")} className="justify-start">Action Button</DropdownMenuItem>
          <DropdownMenuItem onClick={() => onSelect("phase_indicator")} className="justify-start">Phase Indicator</DropdownMenuItem>
          <DropdownMenuItem onClick={() => onSelect("text_display")} className="justify-start">Text Display</DropdownMenuItem>
          <DropdownMenuItem onClick={() => onSelect("voting_panel")} className="justify-start">Voting Panel</DropdownMenuItem>
          <DropdownMenuItem onClick={() => onSelect("avatar_set")} className="justify-start">Avatar Set</DropdownMenuItem>
          <DropdownMenuItem onClick={() => onSelect("background_control")} className="justify-start">Background Control</DropdownMenuItem>
          <DropdownMenuItem onClick={() => onSelect("result_display")} className="justify-start">Result Display</DropdownMenuItem>
          <DropdownMenuItem onClick={() => onSelect("timer")} className="justify-start">Timer</DropdownMenuItem>
          <DropdownMenuItem onClick={() => onSelect("hands_card")} className="justify-start">Hands Card</DropdownMenuItem>
          <DropdownMenuItem onClick={() => onSelect("score_board")} className="justify-start">Score Board</DropdownMenuItem>
          <DropdownMenuItem onClick={() => onSelect("coin_display")} className="justify-start">Coin Display</DropdownMenuItem>
          <DropdownMenuItem onClick={() => onSelect("statement_board")} className="justify-start">Statement Board</DropdownMenuItem>
          <DropdownMenuItem onClick={() => onSelect("reaction_timer")} className="justify-start">Reaction Timer</DropdownMenuItem>
          <DropdownMenuItem onClick={() => onSelect("turn_indicator")} className="justify-start">Turn Indicator</DropdownMenuItem>
          <DropdownMenuItem onClick={() => onSelect("health_display")} className="justify-start">Health Display</DropdownMenuItem>
          <DropdownMenuItem onClick={() => onSelect("influence_set")} className="justify-start">Influence Set</DropdownMenuItem>
          <DropdownMenuItem onClick={() => onSelect("broadcast_input")} className="justify-start">Broadcast Input</DropdownMenuItem>
        </div>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export default NewItemMenu;


