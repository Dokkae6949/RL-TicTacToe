"""
Utility functions shared across the TicTacToe RL project.
"""

def make_state_key(board_tuple, current_player):
    """Create a state key for Q-learning from board state and current player."""
    return (board_tuple, current_player)
