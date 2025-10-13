# DSL Generator Agent

Automatically generate game DSL files from natural language descriptions.

## Features

- 🎮 Converts natural language game descriptions to structured YAML DSL
- 🔄 Iterative refinement: provide feedback to improve the generated DSL
- 💾 Automatic file saving to `games/` directory
- 📝 Follows the project's DSL specification exactly
- 🤖 Powered by GPT-4 for intelligent phase breakdown

## Usage

### Interactive Mode

```bash
# Start interactive session
./scripts/generate-dsl.sh

# Or directly with Python
cd agent
python agent_dsl_generator.py
```

**Example session:**
```
👤 You: Create a werewolf game for 6 players with day/night phases, 
       voting, and role reveals.

🤖 Agent: [Generates DSL with multiple phases]

📄 Generated DSL Preview:
─────────────────────────────────────────────
0:
  name: "Game Setup"
  description: "Initialize the werewolf game..."
...
─────────────────────────────────────────────

Type 'save' to save this DSL, or provide feedback to refine it.

👤 You: save as werewolf_game.yaml

🤖 Agent: ✅ DSL saved to: `games/werewolf_game.yaml`
```

### Direct Mode (CLI)

Generate DSL directly from command line:

```bash
./scripts/generate-dsl.sh "Create a trivia game with 5 rounds, 
multiple choice questions, scoring, and a winner announcement"
```

This will output the YAML directly to stdout (useful for piping).

## DSL Structure

The generator follows this structure:

```yaml
0:  # Phase ID
  name: "Phase Name"
  description: "What happens in this phase"
  actions:
    - description: "Clear previous UI"
      tools: ["clear_canvas"]
    - description: "Action description"
      tools: ["tool1", "tool2"]
  completion_criteria:
    type: "timer" | "player_action" | "final"
    # type-specific fields...
  next_phase:
    id: 1
    name: "Next Phase"
  logic:  # Optional branching
    - if: "condition"
      next_phase: {...}
```

## Available Tools

The generator knows about these UI tools:
- `phase_indicator` - Show current phase
- `text_display` - Display narrative text
- `character_card` - Character information
- `action_button` - Clickable buttons
- `vote_panel` - Voting interface
- `timer_display` - Countdown timer
- `status_indicator` - Game status
- `player_list` - Player roster
- `clear_canvas` - Clear all UI elements
- `deleteItem` - Remove specific element

## Design Principles

The generator follows these principles:

1. **Atomic Phases**: Each phase has clear, discrete actions
2. **UI Management**: First action in each phase clears previous UI
3. **Single Feedback Point**: Each phase requests player input at most once
4. **Clear Completion**: Every phase has explicit completion criteria
5. **Branching Logic**: Supports conditional phase transitions
6. **Final Phase**: Game end is always a separate phase

## Refinement Workflow

1. **Generate**: Describe your game in natural language
2. **Review**: Check the generated DSL preview
3. **Refine**: Provide feedback like:
   - "Make the introduction phase shorter"
   - "Add a voting phase before the reveal"
   - "Change the timer to 60 seconds"
4. **Save**: Type `save` or `save as <filename>`

## Examples

### Simple Game
```
Create a rock-paper-scissors game for 2 players with 
best of 3 rounds and a winner announcement.
```

### Complex Game
```
Create a mafia game for 8 players:
- Night phase: mafia votes to eliminate
- Day phase: discussion and voting
- Role reveals on elimination
- Win conditions: mafia equals town OR all mafia eliminated
```

### Narrative Game
```
Create a choose-your-own-adventure game where players 
make decisions at key story moments, with different 
endings based on their choices.
```

## Output

Generated DSL files are saved to:
```
games/<game_name>_<timestamp>.yaml
```

You can then use them with the main game agent:
```python
# In agent_no_loop.py or agent.py
DSL_FILE = "games/your_generated_game.yaml"
```

## Tips

- **Be specific**: Include number of players, phases, and win conditions
- **Describe flow**: Mention the order of phases and transitions
- **Include details**: Timer durations, voting rules, role reveals, etc.
- **Iterate**: Start simple, then refine with more details

## Troubleshooting

**Generated DSL has errors?**
- Provide more specific game description
- Use refinement mode to adjust specific phases
- Check logs in `logs/dsl_generator_*.log`

**Want to modify existing DSL?**
- Load the existing file
- Describe what you want to change
- Generator will update it

**Need help with DSL structure?**
- See `games/simple_choice_game.yaml` for examples
- Check `CLAUDE.md` for DSL specification
- Review `game_describe.md` for design patterns

