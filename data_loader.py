"""
Data loader for TicTacToe CSV data.
Handles the Kaggle/UCI TicTacToe dataset format.

Dataset format:
- 9 columns for board positions (top-left to bottom-right)
- Each position can be 'x', 'o', or 'b' (blank)
- Last column is the class: 'positive' (x wins) or 'negative' (x doesn't win)
"""

import csv
import random
from typing import List, Tuple


def encode_symbol(symbol: str) -> int:
    """
    Encode board symbols to numeric values.
    x -> 1 (first player)
    o -> -1 (second player)
    b -> 0 (blank)
    ? -> 0 (treat missing as blank)
    """
    symbol = symbol.lower().strip()
    if symbol in ['x', 'X']:
        return 1
    elif symbol in ['o', 'O']:
        return -1
    elif symbol in ['b', 'B', '?', '']:
        return 0
    else:
        return 0


def encode_outcome(outcome: str) -> str:
    """
    Encode outcome labels.
    'positive' -> 'win' (X wins)
    'negative' -> 'draw' or 'loss' (X doesn't win)
    """
    outcome = outcome.lower().strip()
    if outcome == 'positive':
        return 'win'
    else:
        return 'negative'  # This could be draw or loss


def load_tictactoe_data(filepath: str) -> List[Tuple[Tuple[int, ...], str]]:
    """
    Load TicTacToe data from CSV file.
    
    Returns:
        List of tuples: (board_state, outcome)
        board_state is a tuple of 9 integers
        outcome is 'win' or 'negative'
    """
    data = []
    
    try:
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 10:
                    continue
                
                # First 9 columns are board positions
                board = tuple(encode_symbol(row[i]) for i in range(9))
                
                # Last column is outcome
                outcome = encode_outcome(row[9])
                
                data.append((board, outcome))
        
        print(f"Loaded {len(data)} game states from {filepath}")
        return data
    
    except FileNotFoundError:
        print(f"Warning: Data file {filepath} not found. Generating synthetic data...")
        return generate_synthetic_data(1000)


def generate_synthetic_data(num_samples: int = 1000) -> List[Tuple[Tuple[int, ...], str]]:
    """
    Generate synthetic TicTacToe game states for training.
    This creates random valid game states with their outcomes.
    """
    from game import TicTacToe
    
    data = []
    env = TicTacToe()
    
    for _ in range(num_samples):
        env.reset()
        
        # Play random moves until game ends
        while not env.done:
            legal = env.legal_actions()
            if not legal:
                break
            action = random.choice(legal)
            env.step(action)
        
        # Record the final state and outcome
        board = tuple(env.board)
        
        if env.winner == 1:
            outcome = 'win'
        else:
            outcome = 'negative'
        
        data.append((board, outcome))
    
    print(f"Generated {len(data)} synthetic game states")
    return data


def create_sample_dataset(filepath: str = 'tic-tac-toe.data'):
    """
    Create a sample TicTacToe dataset file in UCI format.
    This is useful when the original dataset cannot be downloaded.
    """
    from game import TicTacToe
    
    def symbol_str(val):
        if val == 1:
            return 'x'
        elif val == -1:
            return 'o'
        else:
            return 'b'
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        
        env = TicTacToe()
        
        # Generate 1000 complete games
        for _ in range(1000):
            env.reset()
            
            while not env.done:
                legal = env.legal_actions()
                if not legal:
                    break
                action = random.choice(legal)
                env.step(action)
            
            # Write final state
            row = [symbol_str(env.board[i]) for i in range(9)]
            
            # Add outcome
            if env.winner == 1:
                row.append('positive')
            else:
                row.append('negative')
            
            writer.writerow(row)
    
    print(f"Created sample dataset: {filepath}")


if __name__ == "__main__":
    # Test the data loader
    data = load_tictactoe_data("tic-tac-toe.data")
    if data:
        print(f"\nFirst few samples:")
        for i, (board, outcome) in enumerate(data[:5]):
            print(f"{i+1}. Board: {board}, Outcome: {outcome}")
