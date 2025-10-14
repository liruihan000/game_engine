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
      <DropdownMenuContent align={align} className="min-w-0 w-fit bg-background">
        <DropdownMenuItem onClick={() => onSelect("character_card")}>Character Card</DropdownMenuItem>
        <DropdownMenuItem onClick={() => onSelect("action_button")}>Action Button</DropdownMenuItem>
        <DropdownMenuItem onClick={() => onSelect("phase_indicator")}>Phase Indicator</DropdownMenuItem>
        <DropdownMenuItem onClick={() => onSelect("text_display")}>Text Display</DropdownMenuItem>
        <DropdownMenuItem onClick={() => onSelect("voting_panel")}>Voting Panel</DropdownMenuItem>
        <DropdownMenuItem onClick={() => onSelect("avatar_set")}>Avatar Set</DropdownMenuItem>
        <DropdownMenuItem onClick={() => onSelect("background_control")}>Background Control</DropdownMenuItem>
        <DropdownMenuItem onClick={() => onSelect("result_display")}>Result Display</DropdownMenuItem>
        <DropdownMenuItem onClick={() => onSelect("timer")}>Timer</DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export default NewItemMenu;


