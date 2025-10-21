"use client";

import type { Item, ItemData, CharacterCardData, ActionButtonData, PhaseIndicatorData, TextDisplayData, VotingPanelData, BackgroundControlData, ResultDisplayData, TimerData, AudiencePermissions, HandsCardData, ScoreBoardData, CoinDisplayData, StatementBoardData, ReactionTimerData, NightOverlayData, TurnIndicatorData, HealthDisplayData, InfluenceSetData, BroadcastInputData, PlayerStatesDisplayData, PlayerActionsDisplayData, DeathMarkerData } from "@/lib/canvas/types";
import HandsCard from "@/components/canvas/cards/HandsCard";
import ScoreBoard from "@/components/canvas/cards/ScoreBoard";
import CoinDisplay from "@/components/canvas/cards/CoinDisplay";
import StatementBoard from "@/components/canvas/cards/StatementBoard";
import ReactionTimer from "@/components/canvas/cards/ReactionTimer";
import NightOverlay from "@/components/canvas/cards/NightOverlay";
import TurnIndicator from "@/components/canvas/cards/TurnIndicator";
import HealthDisplay from "@/components/canvas/cards/HealthDisplay";
import InfluenceSet from "@/components/canvas/cards/InfluenceSet";
import Timer from "@/components/canvas/cards/Timer";
import { getPlayersFromStates, getCurrentPlayerId } from "@/lib/player-utils";
// import { chartAddField1Metric, chartRemoveField1Metric, chartSetField1Label, chartSetField1Value, projectAddField4Item, projectRemoveField4Item, projectSetField4ItemDone, projectSetField4ItemText } from "@/lib/canvas/updates";

// Simple Markdown renderer for text display
function renderMarkdown(text: string): string {
  if (!text) return '';
  
  return text
    // Headers
    .replace(/^### (.+)$/gm, '<h3 class="text-base font-bold mt-3 mb-2">$1</h3>')
    .replace(/^## (.+)$/gm, '<h2 class="text-lg font-bold mt-4 mb-2">$1</h2>')
    .replace(/^# (.+)$/gm, '<h1 class="text-xl font-bold mt-4 mb-3">$1</h1>')
    // Bold
    .replace(/\*\*(.+?)\*\*/g, '<strong class="font-bold">$1</strong>')
    // Italic
    .replace(/\*(.+?)\*/g, '<em class="italic">$1</em>')
    // Code inline
    .replace(/`(.+?)`/g, '<code class="bg-black/10 px-1 rounded text-xs font-mono">$1</code>')
    // Lists
    .replace(/^\* (.+)$/gm, '<li class="ml-4">• $1</li>')
    .replace(/^- (.+)$/gm, '<li class="ml-4">• $1</li>')
    // Line breaks
    .replace(/\n/g, '<br/>');
}

export function CardRenderer(props: {
  item: Item;
  onUpdateData: (updater: (prev: ItemData) => ItemData) => void;
  onToggleTag: (tag: string) => void;
  onButtonClick?: (item: Item) => void;
  onVote?: (votingId: string, playerId: string, option: string) => void;
  playerStates?: Record<string, Record<string, unknown>>;
  deadPlayers?: string[];
  playerActions?: Record<string, {
    name: string;
    actions: string;
    timestamp: number;
    phase: string;
  }>;
}) {
  const { item } = props;

  // Check audience permissions
  const checkAudiencePermissions = (data: Partial<AudiencePermissions>): boolean => {
    const currentPlayerId = getCurrentPlayerId();
    if (!currentPlayerId) return true; // Default to visible when no playerId
    
    // Check if data has audience permissions
    if (!data || typeof data.audience_type === 'undefined') return true;
    
    // Visible to everyone
    if (data.audience_type === true) {
      return true;
    }
    
    // Visible to specified players only
    return data.audience_ids?.includes(currentPlayerId) ?? false;
  };

  // Permission check - if current player lacks access, do not render
  if (!checkAudiencePermissions(item.data)) {
    return null;
  }

  // if (item.type === "note") {
  //   const d = item.data as NoteData;
  //   return (
  //     <div className="mt-4">
  //       <label className="mb-1 block text-xs font-medium text-gray-500">Field 1 (textarea)</label>
  //       <TextareaAutosize
  //         value={d.field1 ?? ""}
  //         onChange={(e) => onUpdateData(() => ({ field1: e.target.value }))}
  //         placeholder="Write note..."
  //         className="min-h-40 w-full resize-none rounded-md border bg-white/60 p-3 text-sm leading-6 outline-none placeholder:text-gray-400 transition-colors hover:ring-1 hover:ring-border focus:ring-2 focus:ring-accent/50 focus:shadow-sm focus:bg-accent/10 focus:text-accent focus:placeholder:text-accent/65"
  //         minRows={6}
  //       />
  //     </div>
  //   );
  // }

  // if (item.type === "chart") {
  //   const d = item.data as ChartData;
  //   return (
  //     <div className="mt-4">
  //       <div className="mb-2 flex items-center justify-between">
  //         <span className="text-sm font-medium">Field 1 (metrics)</span>
  //         <button
  //           type="button"
  //           className="inline-flex items-center gap-1 text-xs font-medium text-accent hover:underline"
  //           onClick={() => onUpdateData((prev) => chartAddField1Metric(prev as ChartData, "", "").next)}
  //         >
  //           <Plus className="size-3.5" />
  //           Add new
  //         </button>
  //       </div>
  //       <div className="space-y-3">
  //         {(!d.field1 || d.field1.length === 0) && (
  //           <div className="grid place-items-center py-1.75 text-xs text-primary/50 font-medium text-pretty">
  //             Nothing here yet. Add a metric to get started.
  //           </div>
  //         )}
  //         {d.field1.map((m, i) => {
  //           const number = String(m.id ?? String(i + 1)).padStart(3, "0");
  //           return (
  //           <div key={m.id ?? `metric-${i}`} className="flex items-center gap-3">
  //             <span className="text-xs font-mono text-muted-foreground/80">{number}</span>
  //             <input
  //               value={m.label}
  //               placeholder="Metric label"
  //               onChange={(e) => onUpdateData((prev) => chartSetField1Label(prev as ChartData, i, e.target.value))}
  //               className="w-25 rounded-md border px-2 py-1 text-sm outline-none transition-colors placeholder:text-gray-400 hover:ring-1 hover:ring-border focus:ring-2 focus:ring-accent/50 focus:shadow-sm focus:bg-accent/10 focus:text-accent focus:placeholder:text-accent/65"
  //             />
  //             <div className="flex items-center gap-3 flex-1">
  //               <Progress value={m.value || 0} />
  //             </div>
  //             <input
  //               className={cn(
  //                 "w-10 rounded-md border px-2 py-1 text-xs outline-none appearance-none [-moz-appearance:textfield]",
  //                 "[&::-webkit-outer-spin-button]:[-webkit-appearance:none] [&::-webkit-outer-spin-button]:m-0",
  //                 "[&::-webkit-inner-spin-button]:[-webkit-appearance:none] [&::-webkit-inner-spin-button]:m-0",
  //                 "transition-colors hover:ring-1 hover:ring-border focus:ring-2 focus:ring-accent/50 focus:shadow-sm",
  //                 "focus:bg-accent/10 focus:text-accent font-mono",
  //               )}
  //               type="number"
  //               min={0}
  //               max={100}
  //               value={m.value}
  //               onChange={(e) => onUpdateData((prev) => chartSetField1Value(prev as ChartData, i, e.target.value === "" ? "" : Number(e.target.value)))}
  //               placeholder="0"
  //             />
  //             <button
  //               type="button"
  //               aria-label="Delete metric"
  //               className="text-gray-400 hover:text-accent"
  //               onClick={() => onUpdateData((prev) => chartRemoveField1Metric(prev as ChartData, i))}
  //             >
  //               <X className="h-5 w-5 md:h-6 md:w-6" />
  //             </button>
  //           </div>
  //         );})}
  //       </div>
  //     </div>
  //   );
  // }

  // if (item.type === "project") {
  //   const d = item.data as ProjectData;
  //   const set = (partial: Partial<ProjectData>) => onUpdateData((prev) => ({ ...(prev as ProjectData), ...partial }));
  //   return (
  //     <div className="mt-4 @container">
  //       <div className="mb-3">
  //         <label className="mb-1 block text-xs font-medium text-gray-500">Field 1 (Text)</label>
  //         <input
  //           value={d.field1}
  //           onChange={(e) => set({ field1: e.target.value })}
  //           className="w-full rounded-md border px-2 py-1.5 text-sm outline-none transition-colors placeholder:text-gray-400 hover:ring-1 hover:ring-border focus:ring-2 focus:ring-accent/50 focus:shadow-sm focus:bg-accent/10 focus:text-accent focus:placeholder:text-accent/65"
  //           placeholder="Field 1 value"
  //         />
  //       </div>
  //       <div className="contents @xs:grid gap-3 md:grid-cols-2">
  //         <div className="@max-xs:mb-3">
  //           <label className="mb-1 block text-xs font-medium text-gray-500">Field 2 (Select)</label>
  //           <select
  //             value={d.field2}
  //             onChange={(e) => set({ field2: e.target.value })}
  //             required
  //             className="w-full rounded-md border px-2 py-1.5 text-sm outline-none transition-colors hover:ring-1 hover:ring-border focus:ring-2 focus:ring-accent/50 focus:shadow-sm focus:bg-accent/10 focus:text-accent invalid:text-gray-400"
  //           >
  //             <option value="">Select...</option>
  //             {["Option A", "Option B", "Option C"].map((opt) => (
  //               <option key={opt} value={opt}>{opt}</option>
  //             ))}
  //           </select>
  //         </div>
  //         <div>
  //           <label className="mb-1 block text-xs font-medium text-gray-500">Field 3 (Date)</label>
  //           <input
  //             type="date"
  //             value={d.field3}
  //             onChange={(e) => set({ field3: e.target.value })}
  //             required
  //             className="w-full rounded-md border px-2 py-1.5 text-sm outline-none transition-colors hover:ring-1 hover:ring-border focus:ring-2 focus:ring-accent/50 focus:shadow-sm focus:bg-accent/10 focus:text-accent invalid:text-gray-400"
  //           />
  //         </div>
  //       </div>
  //       <div className="mt-4">
  //         <div className="mb-2 flex items-center justify-between">
  //           <label className="block text-xs font-medium text-gray-500">Field 4 (checklist)</label>
  //           <button
  //             type="button"
  //             className="inline-flex items-center gap-1 text-xs font-medium text-accent hover:underline"
  //             onClick={() => onUpdateData((prev) => projectAddField4Item(prev as ProjectData, "").next)}
  //           >
  //             <Plus className="size-3.5" />
  //             Add new
  //           </button>
  //         </div>
  //         <div className="space-y-2">
  //           {(!d.field4 || d.field4.length === 0) && (
  //             <div className="grid place-items-center py-1.75 text-xs text-primary/50 font-medium text-pretty">
  //               Nothing here yet. Add a checklist item to get started.
  //             </div>
  //           )}
  //           {(d.field4 ?? []).map((c, i) => (
  //             <div key={c.id} className="flex items-center gap-3">
  //               <span className="text-xs font-mono text-muted-foreground/80">{String(c.id ?? String(i + 1)).padStart(3, "0")}</span>
  //               <input
  //                 type="checkbox"
  //                 checked={!!c.done}
  //                 onChange={(e) => onUpdateData((prev) => projectSetField4ItemDone(prev as ProjectData, c.id, e.target.checked))}
  //                 className="h-4 w-4"
  //               />
  //               <input
  //                 value={c.text}
  //                 placeholder="Checklist item label"
  //                 onChange={(e) => onUpdateData((prev) => projectSetField4ItemText(prev as ProjectData, c.id, e.target.value))}
  //                 className="flex-1 rounded-md border px-2 py-1 text-sm outline-none transition-colors placeholder:text-gray-400 hover:ring-1 hover:ring-border focus:ring-2 focus:ring-accent/50 focus:bg-accent/10 focus:text-accent focus:placeholder:text-accent/65"
  //               />
  //               <button
  //                 type="button"
  //                 aria-label="Delete checklist item"
  //                 className="text-gray-400 hover:text-accent"
  //                 onClick={() => onUpdateData((prev) => projectRemoveField4Item(prev as ProjectData, c.id))}
  //               >
  //                 <X className="h-5 w-5 md:h-6 md:w-6" />
  //               </button>
  //             </div>
  //           ))}
  //         </div>
  //       </div>
  //     </div>
  //   );
  // }

  // Game Component Renderers - following the same pattern as ChartData, EntityData, etc.
  if (item.type === "hands_card") {
    const d = item.data as HandsCardData;
    return (
      <HandsCard data={d} title={item.name} subtitle={item.subtitle} />
    );
  }

  if (item.type === "score_board") {
    const d = item.data as ScoreBoardData;
    return (
      <ScoreBoard data={d} />
    );
  }

  if (item.type === "coin_display") {
    const d = item.data as CoinDisplayData;
    return (
      <CoinDisplay data={d} />
    );
  }

  if (item.type === "statement_board") {
    const d = item.data as StatementBoardData;
    return (
      <StatementBoard data={d} />
    );
  }

  if (item.type === "reaction_timer") {
    const d = item.data as ReactionTimerData;
    return (
      <ReactionTimer data={d} />
    );
  }

  if (item.type === "night_overlay") {
    const d = item.data as NightOverlayData;
    return (
      <NightOverlay data={d} />
    );
  }

  if (item.type === "turn_indicator") {
    const d = item.data as TurnIndicatorData;
    return (
      <TurnIndicator data={d} />
    );
  }

  if (item.type === "health_display") {
    const d = item.data as HealthDisplayData;
    return (
      <HealthDisplay data={d} />
    );
  }

  if (item.type === "influence_set") {
    const d = item.data as InfluenceSetData;
    return (
      <InfluenceSet data={d} />
    );
  }
  if (item.type === "character_card") {
    const d = item.data as CharacterCardData;
    // Fixed size for character cards
    const cardSize = "w-55 h-75";
    // Minimalist RPG styling with subtle gold accents
    const role = String(d.role || "");
    
    return (
      <div className={[
        cardSize,
        "relative overflow-hidden rounded-2xl flex flex-col",
        "bg-black/80 backdrop-blur-xl border border-gray-600/40",
        "shadow-[0_20px_40px_rgba(0,0,0,0.9)]",
        "[background-image:linear-gradient(135deg,rgba(255,255,255,0.08)_0%,rgba(255,255,255,0.04)_30%,rgba(0,0,0,0.2)_70%,rgba(0,0,0,0.4)_100%)]",
        "[box-shadow:inset_0_1px_2px_rgba(255,255,255,0.1),inset_0_-1px_2px_rgba(0,0,0,0.3),0_20px_40px_rgba(0,0,0,0.9)]"
      ].join(" ")}>
        {/* top banner */}
        <div className="px-4 py-3 bg-black/30 backdrop-blur-sm border-b border-gray-600/30">
          <div className="flex items-center justify-center">
            <div className="font-extrabold tracking-wide text-[calc(0.95rem+0.3vw)] leading-none text-white/90 drop-shadow-sm">
              {role}
            </div>
          </div>
        </div>
        {/* body */}
        {d.description ? (
          <div className="p-4 text-sm text-slate-200/85 leading-relaxed flex-1">
            {d.description}
          </div>
        ) : (
          <div className="p-4 text-sm text-slate-300/80 italic flex-1">RPG Character</div>
        )}
        {/* subtle vignette */}
        <div className="pointer-events-none absolute inset-0 [background:radial-gradient(120%_80%_at_50%_0%,rgba(255,255,255,0.06)_0,rgba(0,0,0,0)_60%)]" />
      </div>
    );
  }

  if (item.type === "action_button") {
    const d = item.data as ActionButtonData;
    const getSizeClasses = (size: string = 'medium') => {
      const sizeMap: Record<string, string> = {
        small: "w-25 h-10",
        medium: "w-35 h-12",
        large: "w-50 h-14"
      };
      return sizeMap[size] || sizeMap.medium;
    };
    
    return (
      <button 
        className={`${getSizeClasses(d.size)} ${d.variant === 'primary' ? 'bg-primary text-primary-foreground' : d.variant === 'danger' ? 'bg-destructive text-destructive-foreground' : 'bg-secondary text-secondary-foreground'} rounded-md font-medium ${!d.enabled ? 'opacity-50 cursor-not-allowed' : 'hover:opacity-90'}`}
        disabled={!d.enabled}
        onClick={() => {
          if (d.enabled && props.onButtonClick) {
            props.onButtonClick(item);
          }
        }}
      >
        {d.label}
      </button>
    );
  }

  if (item.type === "phase_indicator") {
    const d = item.data as PhaseIndicatorData;
    // Fun, dark cartoon styling + auto sizing (only min constraints)
    const phase = (d.currentPhase || "").toLowerCase();
    const icon = phase.includes("night")
      ? "🌙"
      : phase.includes("day")
      ? "☀️"
      : phase.includes("vote")
      ? "🗳️"
      : "";

    return (
      <div
        className={
          [
            // Auto-size to parent; only minimum constraints
            "inline-flex max-w-none max-h-none w-auto h-auto",
            "min-w-48 min-h-16 px-6 py-4",
            // Glass morphism look - high contrast white
            "rounded-2xl bg-white/70",
            "border-2 border-white/80 shadow-[0_30px_60px_rgba(255,255,255,0.4)]",
            "[background-image:linear-gradient(135deg,rgba(255,255,255,0.8)_0%,rgba(255,255,255,0.6)_30%,rgba(255,255,255,0.4)_70%,rgba(0,0,0,0.1)_100%)]",
            "[box-shadow:inset_0_3px_6px_rgba(255,255,255,0.5),inset_0_-2px_4px_rgba(0,0,0,0.1),0_30px_60px_rgba(255,255,255,0.4)]",
            "text-gray-900 font-bold",
            // Layout
            "items-center justify-center gap-1.5",
          ].join(" ")
        }
      >
        <div className="flex items-center gap-2">
          {icon && (
            <span className="text-[calc(0.9rem+0.6vw)] leading-none">{icon}</span>
          )}
          <div className="flex flex-col items-start">
            <div className="font-extrabold tracking-wide text-[calc(0.8rem+0.5vw)] leading-tight">
              {d.currentPhase}
            </div>
            {d.description && (
              <div className="opacity-80 text-[calc(0.55rem+0.25vw)] leading-snug">
                {d.description}
              </div>
            )}
          </div>
        </div>

        {typeof d.timeRemaining === "number" && d.timeRemaining >= 0 && (
          <div className="mt-2 w-full">
            <div className="h-2.5 w-full rounded-full bg-slate-700/70 overflow-hidden border border-slate-600/60">
              <div className="h-full w-full bg-gradient-to-r from-indigo-400 via-fuchsia-400 to-pink-400 opacity-70" />
            </div>
            <div className="mt-1 text-[0.7rem] tabular-nums text-slate-300/90">
              {d.timeRemaining}s
            </div>
          </div>
        )}
      </div>
    );
  }

  if (item.type === "text_display") {
    const d = item.data as TextDisplayData;

    // Parchment-style theme with subtle variants
    // Intelligent sizing: fit content when small, fit grid when large with scrolling
    const contentLength = (d.content || '').length;
    const isLongContent = contentLength > 300; // Threshold for switching to grid-constrained mode
    
    const baseParchment = [
      // Smart adaptive layout with max height limit
      "relative overflow-hidden box-border",
      isLongContent 
        ? "w-full max-w-md max-h-96" // Large content: constrained width and height with scroll
        : "w-fit h-fit max-w-md", // Small content: natural size with max width
      // Glass morphism look - soft light purple
      "rounded-2xl bg-purple-100/80 backdrop-blur-2xl",
      "border border-purple-200/40 shadow-[0_25px_50px_rgba(196,181,253,0.3)]",
      "[background-image:linear-gradient(135deg,rgba(243,232,255,0.6)_0%,rgba(196,181,253,0.4)_30%,rgba(147,51,234,0.1)_70%,rgba(255,255,255,0.2)_100%)]",
      "[box-shadow:inset_0_2px_4px_rgba(243,232,255,0.8),inset_0_-2px_4px_rgba(196,181,253,0.3),0_25px_50px_rgba(196,181,253,0.3)]",
    ].join(" ");

    const getTypeStyles = (type?: string) => {
      switch (type) {
        case "error":
          return {
            container: `${baseParchment}`,
            title: "text-red-300/90",
            content: "text-red-200/80",
            icon: "☠️",
          };
        case "warning":
          return {
            container: `${baseParchment}`,
            title: "text-amber-300/90",
            content: "text-amber-200/80",
            icon: "⚠️",
          };
        case "success":
          return {
            container: `${baseParchment}`,
            title: "text-emerald-300/90",
            content: "text-emerald-200/80",
            icon: "✨",
          };
        default:
          return {
            container: `${baseParchment}`,
            title: "text-purple-800/90",
            content: "text-purple-700/80",
            icon: "📜",
          };
      }
    };

    const styles = getTypeStyles(d.type);

    return (
      <div className={`${styles.container} p-3 flex flex-col font-sans ${isLongContent ? '' : 'justify-center items-center'}`}>

        {d.title && (
          <div className={`relative z-[1] flex items-center justify-center gap-1 font-semibold text-xs mb-1 flex-shrink-0 ${styles.title}`}>
            <span className="text-sm leading-none flex-shrink-0">{styles.icon}</span>
            <span className="text-center text-xs" style={{ fontSize: 'clamp(0.7rem, 1.2vw, 0.75rem)' }}>{d.title}</span>
          </div>
        )}

        <div 
          className={`relative z-[1] ${isLongContent ? 'flex-1 min-h-0' : 'flex-shrink-0'} text-center leading-tight ${isLongContent ? 'overflow-y-auto' : 'overflow-visible'} ${styles.content} prose prose-xs max-w-none p-2`}
          style={{ 
            fontSize: 'clamp(1rem, 2vw, 1.125rem)',
            ...(isLongContent && {
              scrollbarWidth: 'thin',
              scrollbarColor: '#9ca3af #e5e7eb'
            })
          }}
          dangerouslySetInnerHTML={{ 
            __html: renderMarkdown(d.content || '') 
          }}
        />
      </div>
    );
  }

  if (item.type === "voting_panel") {
    const d = item.data as VotingPanelData;
    
    const handleVote = (option: string) => {
      console.log('Vote button clicked for option:', option);
      const playerId = sessionStorage.getItem('playerId');
      console.log('Retrieved playerId from sessionStorage:', playerId);
      
      if (!playerId) {
        console.warn('No playerId found in sessionStorage');
        // Use a default playerId for testing
        const defaultPlayerId = "1";
        console.log('Using default playerId:', defaultPlayerId);
        if (props.onVote) {
          props.onVote(d.votingId, defaultPlayerId, option);
        }
        return;
      }
      
      // Call the vote handler if provided
      if (props.onVote) {
        console.log('Calling onVote with:', d.votingId, playerId, option);
        props.onVote(d.votingId, playerId, option);
      } else {
        console.warn('No onVote handler provided');
      }
    };
    
    return (
      <div className="w-auto h-auto min-w-64 max-w-sm rounded-2xl bg-black/50 backdrop-blur-2xl border border-white/15 shadow-[0_25px_50px_rgba(0,0,0,0.8)] [background-image:linear-gradient(135deg,rgba(255,255,255,0.15)_0%,rgba(255,255,255,0.08)_30%,rgba(255,255,255,0.03)_70%,rgba(0,0,0,0.1)_100%)] [box-shadow:inset_0_2px_4px_rgba(255,255,255,0.15),inset_0_-2px_4px_rgba(0,0,0,0.1),0_25px_50px_rgba(0,0,0,0.8)] p-4 flex flex-col">
        {d.title && (
          <div className="text-lg font-bold text-center text-white/90 mb-4">
            {d.title}
          </div>
        )}
        <div className="flex gap-3 flex-wrap flex-1">
          {(d.options || []).map((option, index) => (
            <button
              key={index}
              onClick={() => handleVote(option)}
              className="flex-1 min-w-24 p-3 rounded-lg font-semibold transition-all duration-200 bg-white/10 backdrop-blur-sm border border-white/20 text-white/90 hover:bg-white/20 hover:border-white/30 hover:shadow-lg"
            >
              <div className="flex justify-center items-center">
                <span>{option}</span>
              </div>
            </button>
          ))}
        </div>
      </div>
    );
  }

  if (item.type === "avatar_set") {
    // Get players using the universal player utils
    const playerStates = props.playerStates || {};
    const players = getPlayersFromStates(playerStates);
    
    
    // If no players, use test data
    const playersToShow = players.length > 0 ? players : [
      { id: "1", name: "Alice" },
      { id: "2", name: "Bob" },
      { id: "3", name: "Charlie" },
      { id: "4", name: "Diana" }
    ];
    
    // Position players on left and right sides
    const leftPlayers: typeof playersToShow = [];
    const rightPlayers: typeof playersToShow = [];
    
    playersToShow.forEach((player, index) => {
      if (index % 2 === 0) {
        leftPlayers.push(player);
      } else {
        rightPlayers.push(player);
      }
    });
    
    return (
      <>
        {/* Left side avatars - positioned outside the canvas area */}
        <div className="absolute left-12 top-1/2 transform -translate-y-1/2 pointer-events-none">
          <div className="space-y-4">
            {leftPlayers.map((player) => {
              const isDead = props.deadPlayers?.includes(player.id) || false;
              
              return (
                <div key={player.id} className="flex flex-col items-center">
                  {/* Pure Cartoon Avatar */}
                  <div className={`w-20 h-20 flex items-center justify-center transition-all duration-200 hover:scale-110 ${isDead ? 'grayscale opacity-60' : ''}`}>
                    {(() => {
                      const avatarId = parseInt(player.id) % 6;
                      switch (avatarId) {
                        case 0: // Cute Cat
                          return (
                            <div className="text-center">
                              <div className="text-5xl drop-shadow-lg">😺</div>
                            </div>
                          );
                        case 1: // Cool Dog
                          return (
                            <div className="text-center">
                              <div className="text-5xl drop-shadow-lg">🐶</div>
                            </div>
                          );
                        case 2: // Happy Panda
                          return (
                            <div className="text-center">
                              <div className="text-5xl drop-shadow-lg">🐼</div>
                            </div>
                          );
                        case 3: // Cute Fox
                          return (
                            <div className="text-center">
                              <div className="text-5xl drop-shadow-lg">🦊</div>
                            </div>
                          );
                        case 4: // Cool Bear
                          return (
                            <div className="text-center">
                              <div className="text-5xl drop-shadow-lg">🐻</div>
                            </div>
                          );
                        default: // Cute Rabbit
                          return (
                            <div className="text-center">
                              <div className="text-5xl drop-shadow-lg">🐰</div>
                            </div>
                          );
                      }
                    })()}
                  </div>
                  {/* Modern nameplate */}
                  <div className={`text-[10px] mt-2 text-center font-medium px-3 py-1 rounded-full shadow-sm backdrop-blur-sm border ${isDead ? 'opacity-60' : ''} bg-black/20 border-white/20 text-white`}>
                    {player.name}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
        
        {/* Right side avatars - positioned outside the canvas area */}
        <div className="absolute right-12 top-1/2 transform -translate-y-1/2 pointer-events-none">
          <div className="space-y-4">
            {rightPlayers.map((player) => {
              const isDead = props.deadPlayers?.includes(player.id) || false;
              
              return (
                <div key={player.id} className="flex flex-col items-center">
                  {/* Pure Cartoon Avatar */}
                  <div className={`w-20 h-20 flex items-center justify-center transition-all duration-200 hover:scale-110 ${isDead ? 'grayscale opacity-60' : ''}`}>
                    {(() => {
                      const avatarId = parseInt(player.id) % 6;
                      switch (avatarId) {
                        case 0: // Cute Cat
                          return (
                            <div className="text-center">
                              <div className="text-5xl drop-shadow-lg">😺</div>
                            </div>
                          );
                        case 1: // Cool Dog
                          return (
                            <div className="text-center">
                              <div className="text-5xl drop-shadow-lg">🐶</div>
                            </div>
                          );
                        case 2: // Happy Panda
                          return (
                            <div className="text-center">
                              <div className="text-5xl drop-shadow-lg">🐼</div>
                            </div>
                          );
                        case 3: // Cute Fox
                          return (
                            <div className="text-center">
                              <div className="text-5xl drop-shadow-lg">🦊</div>
                            </div>
                          );
                        case 4: // Cool Bear
                          return (
                            <div className="text-center">
                              <div className="text-5xl drop-shadow-lg">🐻</div>
                            </div>
                          );
                        default: // Cute Rabbit
                          return (
                            <div className="text-center">
                              <div className="text-5xl drop-shadow-lg">🐰</div>
                            </div>
                          );
                      }
                    })()}
                  </div>
                  {/* Modern nameplate */}
                  <div className={`text-[10px] mt-2 text-center font-medium px-3 py-1 rounded-full shadow-sm backdrop-blur-sm border ${isDead ? 'opacity-60' : ''} bg-black/20 border-white/20 text-white`}>
                    {player.name}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </>
    );
  }

  if (item.type === "result_display") {
    const d = item.data as ResultDisplayData;
    
    return (
      <div className="flex items-center justify-center min-h-24 w-full">
        <div className="text-center">
          <div className="text-lg md:text-xl lg:text-2xl font-bold bg-gradient-to-r from-purple-600 via-pink-600 to-red-600 bg-clip-text text-transparent drop-shadow-lg transform hover:scale-105 transition-transform duration-300">
            {d.content || "RESULT"}
          </div>
        </div>
      </div>
    );
  }

  if (item.type === "background_control") {
    const d = item.data as BackgroundControlData;
    
    const backgroundOptions = [
      // Solid backgrounds
      { value: "white", label: "White", colorClass: "bg-white" },
      { value: "gray-900", label: "Gray (900)", colorClass: "bg-gray-900" },
      { value: "blue-50", label: "Blue (50)", colorClass: "bg-blue-50" },
      { value: "green-50", label: "Green (50)", colorClass: "bg-green-50" },
      { value: "purple-50", label: "Purple (50)", colorClass: "bg-purple-50" },
      // Felt/table textures
      { value: "felt-green", label: "Felt · Green", colorClass: "bg-[radial-gradient(80%_80%_at_30%_20%,#1b5e2a_0%,#155c2b_55%,#0e4a22_100%)]" },
      { value: "felt-blue", label: "Felt · Blue", colorClass: "bg-[radial-gradient(80%_80%_at_30%_20%,#1e3a8a_0%,#142f6b_55%,#0d2355_100%)]" },
      { value: "felt-red", label: "Felt · Red", colorClass: "bg-[radial-gradient(80%_80%_at_30%_20%,#7a1b1b_0%,#5c1414_55%,#3f0e0e_100%)]" },
      { value: "felt-brown", label: "Felt · Brown", colorClass: "bg-[radial-gradient(80%_80%_at_30%_20%,#6b4e2e_0%,#5a4026_55%,#47331e_100%)]" },
    ];
    
    return (
      <div className="p-4 border rounded-lg bg-card">
        <h3 className="text-sm font-medium mb-3">Background Color Control</h3>
        <div className="grid grid-cols-2 gap-2">
          {backgroundOptions.map((option) => (
            <button
              key={option.value}
              onClick={() => {
                props.onUpdateData(() => ({ backgroundColor: option.value } as BackgroundControlData));
                // Apply background to canvas immediately
                const canvas = document.querySelector('[data-canvas-container]') as HTMLElement;
                if (canvas) {
                  // Remove existing Tailwind bg-* classes, including arbitrary bg-[...]
                  canvas.className = canvas.className.replace(/bg-\w+(-\d+)?|bg-\[[^\]]*\]/g, '');
                  canvas.classList.add(option.colorClass);
                }
              }}
              className={`flex items-center gap-2 p-2 rounded-lg border transition-all ${
                d.backgroundColor === option.value
                  ? 'border-blue-500 bg-blue-50 shadow-sm'
                  : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
              }`}
            >
              <div className={`w-4 h-4 rounded border ${option.colorClass} ${
                option.value === 'white' ? 'border-gray-300' : 'border-gray-200'
              }`} />
              <span className="text-xs font-medium">{option.label}</span>
            </button>
          ))}
        </div>
        <div className="mt-2 text-xs text-gray-500">
          Current: {backgroundOptions.find(opt => opt.value === d.backgroundColor)?.label || 'White'}
        </div>
      </div>
    );
  }

  if (item.type === "timer") {
    const d = item.data as TimerData;
    return <Timer data={d} />;
  }

  if (item.type === "broadcast_input") {
    const d = item.data as BroadcastInputData;
    return (
      <div className="bg-card border border-border rounded-lg p-3 shadow-sm min-w-[280px]">
        <div className="flex items-center gap-2">
          <div className="flex flex-col flex-1">
            <span className="text-xs font-medium text-muted-foreground leading-none pb-1">
              {d.title || "Broadcast"}
            </span>
            <input
              placeholder={d.placeholder || "Type a broadcast message..."}
              className="bg-transparent text-sm outline-none px-2 py-1 rounded border border-input"
              defaultValue=""
            />
          </div>
          <button
            type="button"
            className="px-4 py-2 bg-blue-600 text-white rounded text-sm font-semibold hover:bg-blue-700"
          >
            {d.confirmLabel || "Send"}
          </button>
        </div>
      </div>
    );
  }

  if (item.type === "player_states_display") {
    const d = item.data as PlayerStatesDisplayData;
    const playerStates = props.playerStates || {};
    return (
      <div className="bg-card border border-border rounded-lg p-3 shadow-sm min-w-[320px]">
        <div className="text-sm font-medium text-foreground mb-2">
          {d.title || "Player States"}
        </div>
        <div 
          className="space-y-2 overflow-y-auto"
          style={{ maxHeight: d.maxHeight || "400px" }}
        >
          {Object.keys(playerStates).length === 0 ? (
            <div className="text-xs text-muted-foreground text-center py-4">
              No player states available
            </div>
          ) : (
            Object.entries(playerStates).map(([playerId, state]) => (
              <div key={playerId} className="p-2 bg-muted/50 rounded border text-xs">
                <div className="font-medium mb-1">Player {playerId}</div>
                <pre className="whitespace-pre-wrap text-xs font-mono overflow-hidden">
                  {JSON.stringify(state, null, 2)}
                </pre>
              </div>
            ))
          )}
        </div>
      </div>
    );
  }

  if (item.type === "player_actions_display") {
    const d = item.data as PlayerActionsDisplayData;
    const playerActions = props.playerActions || {};
    
    const playerEntries = Object.entries(playerActions);
    
    return (
      <div className="bg-card border border-border rounded-lg p-3 shadow-sm min-w-[320px]">
        <div className="text-sm font-medium text-foreground mb-2">
          {d.title || "Player Actions"}
        </div>
        <div 
          className="space-y-3 overflow-y-auto"
          style={{ maxHeight: d.maxHeight || "400px" }}
        >
          {playerEntries.length === 0 ? (
            <div className="text-xs text-muted-foreground text-center py-4">
              No player actions recorded
            </div>
          ) : (
            playerEntries.map(([playerId, playerData]) => {
              if (!playerData || typeof playerData !== 'object' || !playerData.actions) {
                return null;
              }
              
              const actions = Object.entries(playerData.actions)
                .map(([actionId, actionData]: [string, any]) => ({
                  actionId,
                  phase: actionData?.phase || '',
                  timestamp: actionData?.timestamp || 0,
                  action: actionData?.action || ''
                }))
                .sort((a, b) => b.timestamp - a.timestamp)
                .slice(0, d.maxItems || 50);
              
              return (
                <div key={playerId} className="p-3 bg-muted/20 border border-border rounded-lg">
                  <div className="text-sm font-medium text-foreground mb-2">
                    {playerData.name}
                  </div>
                  <div className="space-y-1">
                    {actions.length === 0 ? (
                      <div className="text-xs text-muted-foreground">No actions</div>
                    ) : (
                      actions.map((actionData) => (
                        <div key={actionData.actionId} className="text-xs p-2 bg-background/50 rounded">
                          <div className="flex items-center justify-between mb-1">
                            <span className="font-medium">{actionData.phase}</span>
                            <span className="text-muted-foreground">
                              {new Date(actionData.timestamp).toLocaleTimeString()}
                            </span>
                          </div>
                          <div>{actionData.action}</div>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    );
  }

  if (item.type === "death_marker") {
    const d = item.data as DeathMarkerData;
    
    return (
      <div className="bg-card border border-destructive rounded-lg p-3 shadow-sm min-w-[160px]" style={{ borderColor: d.accentColor || "#ef4444" }}>
        <div className="flex items-center justify-center space-x-2">
          <div className="text-2xl">💀</div>
          <div className="text-center">
            <div className="text-sm font-bold text-destructive-foreground" style={{ color: d.accentColor || "#ef4444" }}>
              {d.playerName}
            </div>
            {d.cause && (
              <div className="text-xs text-muted-foreground mt-1">
                {d.cause}
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  // Unknown component type - return error message
  return (
    <div className="p-4 border border-dashed border-red-300 rounded-lg bg-red-50">
      <div className="text-sm text-red-600 font-medium">Unknown component type: {item.type}</div>
      <div className="text-xs text-red-500 mt-1">This component type is not supported by the game engine.</div>
    </div>
  );
}

export default CardRenderer;
