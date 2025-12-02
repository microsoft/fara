#!/usr/bin/env python3
"""
Test script to verify Trajectory class can load text-based log files.
"""

import sys
from pathlib import Path

# Add the src directory to path
sys.path.insert(0, '/mnt/users/wangzhe/code/fara/webeval/src')

from webeval.trajectory import Trajectory

def main():
    traj_path = Path('/data/data/Fara/eval/runs/WebSurfer-fara-100-max_n_images-3/model_checkpoints/corbyrosset/WebVoyager_WebVoyager_data_08312025.jsonl/Nov252025/traj/Allrecipes--0/')

    print("="*80)
    print("Testing Trajectory Loading")
    print("="*80)
    print(f"\nTrajectory path: {traj_path}")
    print(f"Path exists: {traj_path.exists()}\n")

    try:
        # Load the trajectory
        print("Loading trajectory...")
        traj = Trajectory(traj_path)

        print("\n" + "="*80)
        print("✓ Trajectory loaded successfully!")
        print("="*80)

        # Display summary
        print(f"\nTrajectory Summary:")
        print(f"  - Representation: {repr(traj)}")
        print(f"  - Number of events: {len(traj.events)}")
        print(f"  - Number of actions: {len(traj.actions)}")
        print(f"  - Number of thoughts: {len(traj.thoughts)}")
        print(f"  - Number of screenshots: {len(traj.screenshots)}")
        print(f"  - Is aborted: {traj.is_aborted}")

        # Display actions and thoughts
        print("\n" + "="*80)
        print("Actions and Thoughts")
        print("="*80)

        for i, (thought, action) in enumerate(zip(traj.thoughts, traj.actions), 1):
            print(f"\n{'─'*80}")
            print(f"Action {i}:")
            print(f"{'─'*80}")
            print(f"Thought: {thought}")
            print(f"Action:  {action}")

        # Display answer information
        print("\n" + "="*80)
        print("Final Answer")
        print("="*80)
        print(f"Answer: {traj.answer.final_answer}")
        print(f"Is aborted: {traj.answer.is_aborted}")

        # Display token usage if available
        if traj.answer.token_usage:
            print("\n" + "="*80)
            print("Token Usage")
            print("="*80)
            for key, usage in traj.answer.token_usage.items():
                print(f"  {key}: {usage}")

        print("\n" + "="*80)
        print("✓ Test completed successfully!")
        print("="*80)

    except Exception as e:
        print("\n" + "="*80)
        print("✗ Failed to load trajectory!")
        print("="*80)
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
