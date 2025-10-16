"use client";

import type { Item, ItemData, CharacterCardData, ActionButtonData, PhaseIndicatorData, TextDisplayData, VotingPanelData, AvatarSetData, BackgroundControlData, ResultDisplayData, TimerData, AudiencePermissions, HandsCardData, ScoreBoardData, CoinDisplayData, StatementBoardData, ReactionTimerData, NightOverlayData, TurnIndicatorData, HealthDisplayData, InfluenceSetData, BroadcastInputData } from "@/lib/canvas/types";
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
export function CardRenderer(props: {
  item: Item;
  onUpdateData: (updater: (prev: ItemData) => ItemData) => void;
  onToggleTag: (tag: string) => void;
  onButtonClick?: (item: Item) => void;
  onVote?: (votingId: string, playerId: string, option: string) => void;
  playerStates?: Record<string, Record<string, unknown>>;
  deadPlayers?: string[];
}) {
  const { item } = props;

  // Check audience permissions
  const checkAudiencePermissions = (data: ItemData & AudiencePermissions): boolean => {
    const currentPlayerId = getCurrentPlayerId();
    if (!currentPlayerId) return true; // Default to visible when no playerId
    
    // Visible to everyone
    if (data.audience_type === true) {
      return true;
    }
    
    // Visible to specified players only
    return data.audience_ids.includes(currentPlayerId);
  };

  // Permission check - if current player lacks access, do not render
  if (!checkAudiencePermissions(item.data as ItemData & AudiencePermissions)) {
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
    const getSizeClasses = (size: string = 'medium') => {
      const sizeMap: Record<string, string> = {
        small: "w-45 h-60",
        medium: "w-55 h-75", 
        large: "w-70 h-95"
      };
      return sizeMap[size] || sizeMap.medium;
    };
    // Minimalist RPG styling with subtle gold accents
    const role = String(d.role || "");
    const r = role.toLowerCase();
    const roleIcon = r.includes("wolf") || r.includes("werewolf")
      ? "🐺"
      : r.includes("seer") || r.includes("oracle")
      ? "🔮"
      : r.includes("mage") || r.includes("wizard") || r.includes("witch")
      ? "🪄"
      : r.includes("warrior") || r.includes("knight") || r.includes("ronin")
      ? "⚔️"
      : r.includes("thief") || r.includes("rogue")
      ? "🗡️"
      : r.includes("villager") || r.includes("peasant")
      ? "🌾"
      : "";
    
    return (
      <div className={`${getSizeClasses(d.size)} relative overflow-hidden rounded-2xl border-2 border-amber-500/30 ring-4 ring-amber-400/10 bg-gradient-to-br from-slate-800 via-gray-900 to-black shadow-xl flex flex-col`}>
        {/* top banner */}
        <div className="px-4 py-3 bg-gradient-to-r from-amber-500/15 via-amber-400/10 to-amber-500/15 border-b border-amber-500/20">
          <div className="flex items-center gap-2">
            {roleIcon && (
              <span className="text-[calc(1rem+0.3vw)] leading-none">{roleIcon}</span>
            )}
            <div className="font-extrabold tracking-wide text-[calc(0.95rem+0.3vw)] leading-none bg-gradient-to-b from-amber-200 to-amber-400 text-transparent bg-clip-text drop-shadow">
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
            // Dark, cartoon-ish look
            "rounded-3xl bg-gradient-to-br from-slate-800 via-gray-900 to-black",
            "border-2 border-indigo-500/40 ring-4 ring-indigo-400/20",
            "text-slate-100 shadow-2xl drop-shadow-[0_0_10px_rgba(99,102,241,0.35)]",
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
    const baseParchment = [
      // Layout/container
      "relative overflow-hidden min-w-32 min-h-16 w-full h-full",
      "rounded-[18px] border",
      // Warm parchment background using layered gradients
      "bg-[radial-gradient(60%_80%_at_50%_20%,#f3e6c9_0%,#ead9b9_60%,#dfc7a5_100%)]",
      // Edge burn + inner glow
      "shadow-[0_10px_24px_rgba(0,0,0,0.35)]",
      "ring-1 ring-[#b79b79]/40",
      "[box-shadow:inset_0_0_30px_rgba(0,0,0,0.15),inset_0_1px_0_rgba(255,255,255,0.35)]",
    ].join(" ");

    const getTypeStyles = (type?: string) => {
      switch (type) {
        case "error":
          return {
            container: `${baseParchment} border-[#b68a7a] ring-red-900/15`,
            title: "text-[#6b2f24]",
            content: "text-[#4d251d]",
            icon: "☠️",
          };
        case "warning":
          return {
            container: `${baseParchment} border-[#c3a168] ring-amber-900/15`,
            title: "text-[#7a5a2f]",
            content: "text-[#4a3b2a]",
            icon: "⚠️",
          };
        case "success":
          return {
            container: `${baseParchment} border-[#9db38a] ring-emerald-900/15`,
            title: "text-[#355532]",
            content: "text-[#3e3a2f]",
            icon: "✨",
          };
        default:
          return {
            container: `${baseParchment} border-[#c9b59a]`,
            title: "text-[#5a4a3a]",
            content: "text-[#4a3b2a]",
            icon: "📜",
          };
      }
    };

    const styles = getTypeStyles(d.type);

    return (
      <div className={`${styles.container} p-5 flex flex-col gap-2 font-serif`}>
        {/* Texture overlays */}
        <div className="pointer-events-none absolute inset-0 opacity-25 mix-blend-multiply bg-[radial-gradient(40%_30%_at_20%_15%,rgba(0,0,0,0.06),transparent_70%),radial-gradient(30%_25%_at_80%_60%,rgba(0,0,0,0.05),transparent_60%)]" />
        <div className="pointer-events-none absolute inset-0 rounded-[18px] shadow-[inset_0_0_25px_rgba(0,0,0,0.18)]" />

        {d.title && (
          <div className={`relative z-[1] flex items-center gap-2 font-semibold text-sm mb-1 ${styles.title}`}>
            <span className="text-base leading-none">{styles.icon}</span>
            <span className="tracking-wide">{d.title}</span>
          </div>
        )}

        <div className={`relative z-[1] text-[1.25rem] leading-relaxed flex-1 ${styles.content}`}>
          {d.content}
        </div>
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
      <div className="w-full h-full bg-gradient-to-br from-purple-100 to-blue-100 border-2 border-purple-200 rounded-xl p-4 shadow-lg flex flex-col">
        {d.title && (
          <div className="text-lg font-bold text-center text-purple-800 mb-4">
            {d.title}
          </div>
        )}
        <div className="flex gap-3 flex-wrap flex-1">
          {d.options.map((option, index) => (
            <button
              key={index}
              onClick={() => handleVote(option)}
              className="flex-1 min-w-24 p-3 rounded-lg font-semibold transition-all duration-200 border-2 bg-white text-purple-700 border-purple-300 hover:bg-purple-50 hover:border-purple-400 hover:shadow-md"
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
    const d = item.data as AvatarSetData;
    
    // Get players using the universal player utils
    const playerStates = props.playerStates || {};
    const players = getPlayersFromStates(playerStates);
    
    // Avatar mapping based on type
    const avatarMap: Record<string, string> = {
      human: '👤',
      wolf: '🐺', 
      dog: '🐶',
      cat: '🐱'
    };
    
    const avatarEmoji = avatarMap[d.avatarType] || avatarMap.human;
    
    // Poker chip skins for a retro card-table vibe
    const chipSkins = [
      // Red chip
      'bg-[repeating-conic-gradient(#b91c1c_0deg_20deg,white_20deg_40deg)]',
      // Blue chip
      'bg-[repeating-conic-gradient(#1e3a8a_0deg_20deg,white_20deg_40deg)]',
      // Green chip
      'bg-[repeating-conic-gradient(#166534_0deg_20deg,white_20deg_40deg)]',
      // Black chip
      'bg-[repeating-conic-gradient(#111827_0deg_20deg,white_20deg_40deg)]',
      // Purple chip
      'bg-[repeating-conic-gradient(#6d28d9_0deg_20deg,white_20deg_40deg)]',
      // Teal chip
      'bg-[repeating-conic-gradient(#0f766e_0deg_20deg,white_20deg_40deg)]',
      // Orange chip
      'bg-[repeating-conic-gradient(#c2410c_0deg_20deg,white_20deg_40deg)]',
      // Brown chip
      'bg-[repeating-conic-gradient(#6b4423_0deg_20deg,white_20deg_40deg)]',
      // Pink chip
      'bg-[repeating-conic-gradient(#be185d_0deg_20deg,white_20deg_40deg)]',
    ];
    
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
        <div className="absolute left-0 top-1/2 transform -translate-y-1/2 pointer-events-none">
          <div className="space-y-4">
            {leftPlayers.map((player) => {
              const colorIndex = parseInt(player.id) % chipSkins.length;
              const isDead = props.deadPlayers?.includes(player.id) || false;
              
              return (
                <div key={player.id} className="flex flex-col items-center">
                  {/* Poker chip avatar */}
                  <div className={`relative w-16 h-16 rounded-full shadow-xl ring-2 ring-black/20 border-4 border-white ${isDead ? 'grayscale opacity-60 bg-gray-400' : chipSkins[colorIndex]}`}>
                    <div className="absolute inset-1 rounded-full bg-[radial-gradient(70%_70%_at_35%_30%,#fefefe_0%,#e6e6e6_60%,#cfcfcf_100%)] flex items-center justify-center text-2xl">
                      {avatarEmoji}
                    </div>
                  </div>
                  {/* Wooden nameplate */}
                  <div className={`text-[10px] mt-2 text-center font-semibold px-2 py-1 rounded-md shadow border ${isDead ? 'opacity-60' : ''} bg-[linear-gradient(135deg,#9a6b3f,#7b4a2e)] border-[#5c3a24] text-[#2b1a0e] drop-shadow-[0_1px_0_rgba(255,255,255,0.35)]`}>
                    {player.name}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
        
        {/* Right side avatars - positioned outside the canvas area */}
        <div className="absolute right-0 top-1/2 transform -translate-y-1/2 pointer-events-none">
          <div className="space-y-4">
            {rightPlayers.map((player) => {
              const colorIndex = parseInt(player.id) % chipSkins.length;
              const isDead = props.deadPlayers?.includes(player.id) || false;
              
              return (
                <div key={player.id} className="flex flex-col items-center">
                  {/* Poker chip avatar */}
                  <div className={`relative w-16 h-16 rounded-full shadow-xl ring-2 ring-black/20 border-4 border-white ${isDead ? 'grayscale opacity-60 bg-gray-400' : chipSkins[colorIndex]}`}>
                    <div className="absolute inset-1 rounded-full bg-[radial-gradient(70%_70%_at_35%_30%,#fefefe_0%,#e6e6e6_60%,#cfcfcf_100%)] flex items-center justify-center text-2xl">
                      {avatarEmoji}
                    </div>
                  </div>
                  {/* Wooden nameplate */}
                  <div className={`text-[10px] mt-2 text-center font-semibold px-2 py-1 rounded-md shadow border ${isDead ? 'opacity-60' : ''} bg-[linear-gradient(135deg,#9a6b3f,#7b4a2e)] border-[#5c3a24] text-[#2b1a0e] drop-shadow-[0_1px_0_rgba(255,255,255,0.35)]`}>
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
          <div className="text-4xl md:text-6xl lg:text-8xl font-black bg-gradient-to-r from-purple-600 via-pink-600 to-red-600 bg-clip-text text-transparent drop-shadow-2xl transform hover:scale-105 transition-transform duration-300">
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

  // Unknown component type - return error message
  return (
    <div className="p-4 border border-dashed border-red-300 rounded-lg bg-red-50">
      <div className="text-sm text-red-600 font-medium">Unknown component type: {item.type}</div>
      <div className="text-xs text-red-500 mt-1">This component type is not supported by the game engine.</div>
    </div>
  );
}

export default CardRenderer;
