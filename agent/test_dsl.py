#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to load and display DSL content for werewolf-(mafia) game
"""

import asyncio
import os
import sys
import yaml

# Add the agent directory to the path to import the DSL loading function
sys.path.append(os.path.dirname(__file__))

from dm_agent_with_bot import load_dsl_by_gamename

async def test_dsl_loading():
    """Test DSL loading for werewolf-(mafia) game"""
    
    gamename = "werewolf-(mafia)"
    print(f"Testing DSL loading for game: {gamename}")
    print("=" * 50)
    
    try:
        # Load DSL using the function from dm_agent_with_bot
        dsl_content = await load_dsl_by_gamename(gamename)
        
        if not dsl_content:
            print("FAILED: Failed to load DSL - empty content returned")
            return
        
        print("SUCCESS: DSL loaded successfully!")
        print(f"Top-level keys: {list(dsl_content.keys())}")
        print()
        
        # Display declaration section
        if 'declaration' in dsl_content:
            declaration = dsl_content['declaration']
            print("DECLARATION SECTION:")
            print(f"   Description: {declaration.get('description', 'N/A')[:100]}...")
            print(f"   Multiplayer: {declaration.get('is_multiplayer', 'N/A')}")
            print(f"   Min Players: {declaration.get('min_players', 'N/A')}")
            
            if 'roles' in declaration:
                print(f"   Roles: {[role['name'] for role in declaration['roles']]}")
            
            if 'player_states' in declaration:
                print(f"   Player State Fields: {list(declaration['player_states'].keys())}")
            print()
        
        # Display phases section
        if 'phases' in dsl_content:
            phases = dsl_content['phases']
            print("PHASES SECTION:")
            print(f"   Total phases: {len(phases)}")
            print(f"   Phase IDs: {list(phases.keys())}")
            
            # Show first few phases as examples
            for phase_id in sorted(list(phases.keys()))[:5]:
                phase = phases[phase_id]
                phase_name = phase.get('name', f'Phase {phase_id}')
                print(f"   Phase {phase_id}: {phase_name}")
                if 'description' in phase:
                    desc = phase['description'][:80] + "..." if len(phase['description']) > 80 else phase['description']
                    print(f"     Description: {desc}")
            
            if len(phases) > 5:
                print(f"     ... and {len(phases) - 5} more phases")
            print()
        
        # Test specific phase lookup
        print("TESTING PHASE LOOKUP:")
        test_phase_id = 0
        if 'phases' in dsl_content and test_phase_id in dsl_content['phases']:
            phase = dsl_content['phases'][test_phase_id]
            print(f"   Phase {test_phase_id} found:")
            print(f"     Name: {phase.get('name', 'N/A')}")
            print(f"     Description: {phase.get('description', 'N/A')[:100]}...")
            print(f"     Actions: {phase.get('actions', [])}")
        else:
            print(f"   Phase {test_phase_id} not found in DSL")
        
        print()
        print("SUCCESS: DSL test completed successfully!")
        
    except Exception as e:
        print(f"ERROR: Error during DSL testing: {e}")
        import traceback
        traceback.print_exc()

def test_file_access():
    """Test direct file access to verify the YAML file exists and is readable"""
    
    print("TESTING DIRECT FILE ACCESS:")
    print("=" * 30)
    
    # Construct file path
    script_dir = os.path.dirname(__file__)
    dsl_file_path = os.path.join(script_dir, '..', 'games', 'werewolf-(mafia).yaml')
    absolute_path = os.path.abspath(dsl_file_path)
    
    print(f"File path: {absolute_path}")
    print(f"File exists: {os.path.exists(absolute_path)}")
    
    if os.path.exists(absolute_path):
        try:
            file_size = os.path.getsize(absolute_path)
            print(f"File size: {file_size} bytes")
            
            # Read and parse YAML directly
            with open(absolute_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"Content length: {len(content)} characters")
                
                # Parse YAML
                parsed = yaml.safe_load(content)
                print(f"YAML parsed successfully")
                print(f"Parsed keys: {list(parsed.keys())}")
                
        except Exception as e:
            print(f"ERROR: Error reading file: {e}")
    else:
        # List games directory to see what files are available
        games_dir = os.path.join(script_dir, '..', 'games')
        if os.path.exists(games_dir):
            print(f"Games directory contents:")
            for file in os.listdir(games_dir):
                print(f"   - {file}")
        else:
            print(f"ERROR: Games directory not found: {games_dir}")
    
    print()

async def main():
    """Main test function"""
    print("DSL LOADING TEST")
    print("=" * 50)
    
    # Test direct file access first
    test_file_access()
    
    # Test DSL loading function
    await test_dsl_loading()

if __name__ == "__main__":
    asyncio.run(main())