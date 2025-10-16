# Full-Stack AI Game Canvas

Built in less than five days — almost no sleep, complete focus.

**Documentation:** [https://docs.google.com/document/d/1CugOHIvGYZ7J339M6bQpwU7fyY-Dg1BHJ__HTwXXYDA/edit?usp=drive_link](https://docs.google.com/document/d/1CugOHIvGYZ7J339M6bQpwU7fyY-Dg1BHJ__HTwXXYDA/edit?usp=drive_link)  
**Demo Video:** [https://drive.google.com/file/d/18px28PHM45-oy7GmVgpQ5uknrBUTeqyM/view?usp=sharing](https://drive.google.com/file/d/18px28PHM45-oy7GmVgpQ5uknrBUTeqyM/view?usp=sharing)

---

## Overview

This project was born from obsession.
Over five sleepless nights — maybe ten hours of rest in total — I poured everything I had into building a prototype that could turn pure text into playable, dynamic AI-driven worlds.

What started as a “technical test” quickly became a personal mission:
to prove that AI shouldn’t just describe worlds — *it should build them, run them, and evolve them*.

This is a full-stack AI game engine that composes both backend logic and frontend UI directly from YAML DSL files — no hardcoded logic, no templates, no shortcuts.

---

## Architecture

| Layer | Technology | Responsibility |
|-------|-------------|----------------|
| Frontend | **Next.js + CopilotKit** | Render dynamic canvas, expose UI tools, sync state |
| API Layer | **Next.js Routes** | Bridge between client and backend |
| Backend | **Python + LangGraph** | Interpret DSL, orchestrate logic, control UI |
| Future | **Redis + PostgreSQL** | Persistent sessions and structured data |

Each game session runs in an isolated `threadId`, maintaining independent state and logic.  
The agent acts directly on the UI via CopilotKit tool calls — dynamically creating, modifying, or destroying interface components such as cards, timers, votes, and player lists.

---

## Technical Highlights

- **YAML DSL Engine** — Games are defined, not coded.  
- **Atomic Operations** — Shared primitives handle all gameplay mechanics.  
- **Custom Copilot Bridge** — Backend tool calls trigger frontend actions via `useCopilotAction`.  
- **Multi-Agent Runtime** — Referee, bots, and human players coexist under a single orchestration layer.  
- **Open Broadcast Protocol (Experimental)** — Prototype for decentralized multi-agent communication and delegation.  
- **Reliability Pipeline** — Deterministic replay, validation, and recovery ensure >90% completion rates.  

---

## Installation

### Requirements
- Node.js ≥ 18  
- Python ≥ 3.12  
- pnpm (recommended)  
- OpenAI API key  

### Steps
```bash
git clone https://github.com/liruihan000/game_engine.git
cd game_engine
pnpm install
echo 'OPENAI_API_KEY=your-api-key' > agent/.env
pnpm dev     # runs frontend (:3000) and backend (:8123)
