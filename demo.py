"""
Demo script showing the complete workflow:
1. Load/generate CSV data
2. Train the agent
3. Evaluate performance
4. Explain how to use GUI
"""

import os
import sys


def main():
    print("=" * 70)
    print("RL-TicTacToe Demo - Complete Workflow")
    print("=" * 70)
    
    # Step 1: Check/Generate data
    print("\n[Step 1] Checking for training data...")
    if not os.path.exists("tic-tac-toe.data"):
        print("  Creating sample TicTacToe dataset...")
        from data_loader import create_sample_dataset
        create_sample_dataset()
    else:
        print("  ✓ Found existing dataset: tic-tac-toe.data")
    
    # Step 2: Train the agent
    print("\n[Step 2] Training the RL agent...")
    print("  This includes CSV pre-training + RL self-play training")
    
    from train import train
    print("  Starting quick training (1000 episodes)...")
    agent = train(episodes=1000, save_path="qtable.pkl")
    print("  ✓ Training complete!")
    
    # Step 3: Evaluate
    print("\n[Step 3] Evaluating agent performance...")
    from evaluate import evaluate
    results = evaluate(agent, episodes=1000)
    
    win_rate = results['win'] / sum(results.values()) * 100
    print(f"  ✓ Agent win rate: {win_rate:.1f}%")
    
    # Step 4: Usage instructions
    print("\n[Step 4] How to use:")
    print("=" * 70)
    print("\n  A) Play with GUI (recommended):")
    print("     $ python gui.py")
    print("     - Click cells to make your move")
    print("     - You play as O, Agent plays as X")
    print("     - Click 'New Game' to restart")
    
    print("\n  B) Play in console:")
    print("     $ python play.py")
    print("     - Enter cell numbers 0-8 to make moves")
    
    print("\n  C) Train with custom parameters:")
    print("     $ python train.py --episodes 50000")
    print("     $ python train.py --no-csv  # Skip CSV pre-training")
    
    print("\n  D) Evaluate agent:")
    print("     $ python evaluate.py")
    
    print("\n" + "=" * 70)
    print("Demo complete! Model saved to qtable.pkl")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
