#!/bin/bash
# Quick script to run DSL Generator Agent

cd "$(dirname "$0")/.."

# Activate virtual environment if it exists
if [ -d "agent/.venv" ]; then
    source agent/.venv/bin/activate
fi

# Run the DSL generator
python agent/agent_dsl_generator.py "$@"

