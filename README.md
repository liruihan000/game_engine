# 🎮 Full-Stack AI Game Canvas

*Built in under 5 days — with almost no sleep and endless curiosity.*

**Documentation:** [Google Doc](https://docs.google.com/document/d/1CugOHIvGYZ7J339M6bQpwU7fyY-Dg1BHJ__HTwXXYDA/edit?usp=drive_link)  

**Future development plan:** [Google Doc](https://docs.google.com/document/d/10kWj0d3kgHijSTeN-svgy8nV4BalymIZrE-zh7uruv0/edit?usp=drive_link)  

**Demo Video:** [Watch Demo](https://youtu.be/DxSRnGJXdRA)

*The video shows two cases: Werewolf (Mafia) and Two Truths and a Lie.
I haven't made any customized operations on any DSLs. They are all automatically generated based on a description.*

---

## 💡 Overview

This project was born from obsession.  
Over five sleepless nights — maybe ten hours of rest in total — I poured everything I had into building a prototype that could **turn pure text into playable, dynamic AI-driven worlds**.

What started as a technical test quickly became a personal mission:  
to prove that **AI shouldn’t just describe worlds — it should *build* them, *run* them, and *evolve* them.**

It's a system where games are not programmed but described —  
where an agent reads a rule set, generates logic, composes UI, and plays *with* or *against* you.  
The result is a full-stack AI game engine that can orchestrate both logic and interface in real time.

Note: The multi-player system is not fully implemented yet. The missing piece is agent broadcasting, but once that's resolved, all other interfaces are already in place.

---

## ⚡ Installation

### Requirements
- Node.js ≥ 18  
- Python ≥ 3.12  
- pnpm (recommended)  
- OpenAI API key  

### Setup
```bash
git clone https://github.com/liruihan000/game_engine.git
cd game_engine
pnpm install
echo 'OPENAI_API_KEY=your-key' > agent/.env
pnpm dev  # launches frontend (:3000) + backend (:8123) 
```

### Dsl Generator

1. Visit **`:3000/dsl-generator`** in your browser.  
2. Enter the **game name** and **description**.  
3. Click **Generate** — the generation process currently takes about **10 minutes**.  
   > *Note:* Due to time constraints, optimization for generation speed hasn’t been a focus yet,  
   > but the current speed can be improved by at least **3×** in future iterations.  
4. The generated **YAML file** will be saved automatically in the **game directory**.  
5. Open the **Game Library** at **`:3000`**, and you’ll see the newly created game.  
   - If a game with the same name already exists, it will be **overwritten**.
  
### Gameplay Guide

1. Currently, only "Create Room" is available. The "Join Room" feature is a reserved interface for future multiplayer mode.

2. Due to the bot design, after completing a phase, please click “Continue” or wait for the timer to end to proceed to the next round.

---

## ⚙️ Architecture Overview

| Layer | Technology | Responsibility |
|-------|-------------|----------------|
| **Frontend** | Next.js + CopilotKit | Render canvas, expose UI tools, sync state |
| **API Layer** | Next.js | Backend Logic |
| **Agent** | Python + LangGraph | Interpret YAML DSL, orchestrate logic, control UI |
| **Storage (Future)** | Redis + PostgreSQL | Persistent sessions & structured data |

Each session runs in an isolated `threadId`, maintaining full separation of state and reasoning context.  
The AI agent communicates via **CopilotKit tool calls**, dynamically creating and modifying UI components — cards, votes, timers, text panels — with no manual code.

---

## 🧠 System Design

### Frontend (Next.js + React)
- Data-driven **canvas** rendering from agent state.
- Exposes frontend functions via `useCopilotAction`, enabling backend agents to modify UI in real time.
- Synchronization through `useCoAgent` ensures frontend and backend stay perfectly aligned.

### Middleware (CopilotKit)
- Acts as a **WebSocket bridge** between logic and presentation.
- Handles bi-directional state updates and tool calls.
- Ensures every UI action and state mutation can be triggered or observed by the agent.

### Agent (LangGraph)
- LangGraph orchestrates game flow from YAML DSL files.
- Multi-node architecture:
  - **ActionExecutor** — renders and updates UI components.  
  - **RefereeNode** — enforces rules and scoring.  
  - **BotBehaviorNode** — controls NPC logic.  
  - **PhaseNode** — manages transitions and timing.

---

## 🔬 Exploration and Experiments

I didn’t take the easy path.  
In five days, I tested and discarded multiple architectures before convergence:

1. **Recursive Research Agent** — deep reasoning, unstable under real-time load.  
2. **ReWOO-style Planner** — structured decomposition, slower but interpretable.  
3. **Single-Node Real-Time Agent** — reduce latency, but DSL execution is inaccurate..  

I also:
- Hand-coded the **backend-to-frontend Copilot bridge**.  
- Experimented with **multi-agent broadcast and delegation protocols**.  
- Tested nearly every available agent orchestration pattern (ReWOO, Deep Research, AutoPlan, custom graph loops).  
- Began exploring a **intermediate agent** — a design that could unify all logic and coordination under a single reasoning substrate.

---

## 🧩 What Works

- **Component-based AI Canvas** — The agent controls the atomic UI components.  
- **YAML DSL Engine** — Add new games by writing a sentence in dsl generator page.  
- **Multi-Agent Runtime** — Referee, bots coexist in the same session.  
- **Bi-Directional Sync** — Zero refresh, consistent across client and agent.  
- **Evaluation Tools** — Completion metrics, validation, and logs.  

---

