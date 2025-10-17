# Repository Guidelines

## Project Structure & Module Organization
- `src/`: Next.js UI (`app/`, `components/`, `lib/`, `hooks/`).
- `agent/`: LangGraph Python agent (`agent.py`, `requirements.txt`, `.venv/`).
- `games/`: DSL files defining game rules (only part that varies).
- `public/` assets; `scripts/` setup helpers. Config: `eslint.config.mjs`, `tsconfig.json`, `next.config.ts`, `.env.local` (UI), `agent/.env` (agent).

## Build, Test, and Development Commands
- `pnpm dev`: Run UI (:3000) and agent (:8123) concurrently.
- `pnpm dev:ui` / `pnpm dev:agent`: Start each service independently.
- `pnpm build` / `pnpm start`: Build and run production UI.
- `pnpm lint`: ESLint (Next + TypeScript config).
- `pnpm run install:agent`: Create `agent/.venv` and install Python deps.

## Coding Style & Naming Conventions
- TypeScript/React: 2 spaces; `PascalCase` components; hooks `use*`; prefer `const`; strict types.
- Python (agent): 4 spaces; `snake_case`; add type hints where useful.
- Filenames: components `PascalCase.tsx`; utilities `kebab-case.ts`.
- Formatting/Lint: ESLint only; follow defaults (no Prettier config in repo).


## Commit & Pull Request Guidelines
- Commits: Conventional Commits (`feat:`, `fix:`, `chore:`). Example: `feat: add voting panel tool`.
- PRs: Clear summary, rationale, linked issues, and screenshots/GIFs for UI changes. Verify `pnpm lint`, both services run, and docs updated when behavior/config changes.

## Agent‑Specific Instructions (from CLAUDE.md)
- DSL‑first: no game‑specific branches; rules live in `games/*.yaml`.
- Atomic tools: both backend and frontend actions are generic.
- State sync: use CopilotKit shared state; `updateItem` is the central mutation path on the UI.
- Expose UI capabilities via `useCopilotAction` (available: "remote"). Example: `name: "displayVotePanel"` with params, then update state via `setState`/`updateItem`.
- Setup: `echo 'OPENAI_API_KEY=...' > agent/.env`. Ports: UI 3000, Agent 8123 (`LANGGRAPH_DEPLOYMENT_URL` can override).
