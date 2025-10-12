"use client";

import type { Item, ItemData, CharacterCardData, ActionButtonData, PhaseIndicatorData, TextDisplayData } from "@/lib/canvas/types";
// import { chartAddField1Metric, chartRemoveField1Metric, chartSetField1Label, chartSetField1Value, projectAddField4Item, projectRemoveField4Item, projectSetField4ItemDone, projectSetField4ItemText } from "@/lib/canvas/updates";
export function CardRenderer(props: {
  item: Item;
  onUpdateData: (updater: (prev: ItemData) => ItemData) => void;
  onToggleTag: (tag: string) => void;
}) {
  const { item } = props;

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
    
    return (
      <div className={`${getSizeClasses(d.size)} bg-card border rounded-lg p-4 flex flex-col`}>
        <div className="text-sm text-muted-foreground">{d.role}</div>
        {d.description && <div className="text-xs mt-2 flex-1">{d.description}</div>}
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
      >
        {d.label}
      </button>
    );
  }

  if (item.type === "phase_indicator") {
    const d = item.data as PhaseIndicatorData;
    const getSizeClasses = (size: string = 'medium') => {
      const sizeMap: Record<string, string> = {
        small: "w-30 h-8",
        medium: "w-45 h-12",
        large: "w-60 h-16"
      };
      return sizeMap[size] || sizeMap.medium;
    };
    
    return (
      <div className={`${getSizeClasses(d.size)} bg-accent text-accent-foreground rounded-lg flex flex-col items-center justify-center`}>
        <div className="font-bold text-sm">{d.currentPhase}</div>
        {d.description && <div className="text-xs">{d.description}</div>}
        {d.timeRemaining && <div className="text-xs">{d.timeRemaining}s</div>}
      </div>
    );
  }

  if (item.type === "text_display") {
    const d = item.data as TextDisplayData;
    const getSizeClasses = (size: string = 'medium') => {
      const sizeMap: Record<string, string> = {
        small: "w-50 h-20",
        medium: "w-75 h-30",
        large: "w-100 h-45"
      };
      return sizeMap[size] || sizeMap.medium;
    };
    
    return (
      <div className={`${getSizeClasses(d.size)} ${d.type === 'error' ? 'bg-destructive/10 border-destructive' : d.type === 'warning' ? 'bg-yellow-50 border-yellow-200' : d.type === 'success' ? 'bg-green-50 border-green-200' : 'bg-card'} border rounded-lg p-3 flex flex-col`}>
        {d.title && <div className="font-semibold text-sm mb-2">{d.title}</div>}
        <div className="text-sm flex-1">{d.content}</div>
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


