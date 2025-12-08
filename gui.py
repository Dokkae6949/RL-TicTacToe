"""
Basic TicTacToe GUI using tkinter.
Allows playing against the trained RL agent with a graphical interface.
"""

import tkinter as tk
from tkinter import messagebox
from game import TicTacToe
from agent import QLearningAgent
import os


def make_state_key(board_tuple, current_player):
    """Same state key function used in training."""
    return (board_tuple, current_player)


class TicTacToeGUI:
    def __init__(self, root, qtable_path="qtable.pkl"):
        self.root = root
        self.root.title("TicTacToe - RL Agent")
        self.root.resizable(False, False)
        
        # Initialize game and agent
        self.env = TicTacToe()
        self.agent = QLearningAgent()
        
        # Load trained model if available
        if os.path.exists(qtable_path):
            try:
                self.agent.load(qtable_path)
                self.model_loaded = True
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model_loaded = False
        else:
            self.model_loaded = False
            print(f"Warning: Model file {qtable_path} not found. Agent will play randomly.")
        
        # Game state
        self.board = self.env.reset()
        self.game_active = True
        
        # UI Components
        self.setup_ui()
        
        # Agent plays first (X)
        self.update_status("Agent's turn (X)...")
        self.root.after(500, self.agent_move)
    
    def setup_ui(self):
        """Create the GUI components."""
        # Status label
        self.status_label = tk.Label(
            self.root,
            text="Agent plays X, You play O",
            font=("Arial", 14),
            pady=10
        )
        self.status_label.pack()
        
        # Game board frame
        board_frame = tk.Frame(self.root)
        board_frame.pack(padx=20, pady=10)
        
        # Create 3x3 grid of buttons
        self.buttons = []
        for i in range(3):
            row_buttons = []
            for j in range(3):
                btn = tk.Button(
                    board_frame,
                    text=" ",
                    font=("Arial", 32, "bold"),
                    width=5,
                    height=2,
                    command=lambda idx=i*3+j: self.player_move(idx)
                )
                btn.grid(row=i, column=j, padx=2, pady=2)
                row_buttons.append(btn)
            self.buttons.append(row_buttons)
        
        # Control buttons
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)
        
        self.new_game_btn = tk.Button(
            control_frame,
            text="New Game",
            font=("Arial", 12),
            command=self.new_game,
            padx=20
        )
        self.new_game_btn.pack(side=tk.LEFT, padx=5)
        
        self.quit_btn = tk.Button(
            control_frame,
            text="Quit",
            font=("Arial", 12),
            command=self.root.quit,
            padx=20
        )
        self.quit_btn.pack(side=tk.LEFT, padx=5)
        
        # Info label
        info_text = "Loaded trained model" if self.model_loaded else "No trained model (random play)"
        self.info_label = tk.Label(
            self.root,
            text=info_text,
            font=("Arial", 10),
            fg="gray"
        )
        self.info_label.pack(pady=5)
    
    def update_board_display(self):
        """Update button texts to reflect current board state."""
        symbols = {1: "X", -1: "O", 0: " "}
        for i in range(3):
            for j in range(3):
                idx = i * 3 + j
                symbol = symbols[self.env.board[idx]]
                self.buttons[i][j].config(text=symbol)
                
                # Disable button if cell is occupied
                if self.env.board[idx] != 0 or not self.game_active:
                    self.buttons[i][j].config(state="disabled")
                else:
                    self.buttons[i][j].config(state="normal")
    
    def update_status(self, message):
        """Update the status label."""
        self.status_label.config(text=message)
        self.root.update()
    
    def player_move(self, action):
        """Handle player's move (O)."""
        if not self.game_active or self.env.current_player != -1:
            return
        
        if self.env.board[action] != 0:
            return
        
        # Player makes move
        self.board, _, done, _ = self.env.step(action)
        self.update_board_display()
        
        if done:
            self.end_game()
            return
        
        # Agent's turn
        self.update_status("Agent's turn (X)...")
        self.root.after(500, self.agent_move)
    
    def agent_move(self):
        """Agent makes its move (X)."""
        if not self.game_active or self.env.current_player != 1:
            return
        
        state_key = make_state_key(tuple(self.board), self.env.current_player)
        legal = self.env.legal_actions()
        
        if not legal:
            self.end_game()
            return
        
        # Agent selects action
        action = self.agent.get_action(state_key, legal, training=False)
        
        # Agent makes move
        self.board, _, done, _ = self.env.step(action)
        self.update_board_display()
        
        if done:
            self.end_game()
            return
        
        # Player's turn
        self.update_status("Your turn (O)")
    
    def end_game(self):
        """Handle game end."""
        self.game_active = False
        self.update_board_display()
        
        if self.env.winner == 1:
            message = "Agent wins!"
            self.update_status("Game Over - Agent wins!")
        elif self.env.winner == -1:
            message = "You win!"
            self.update_status("Game Over - You win!")
        else:
            message = "It's a draw!"
            self.update_status("Game Over - Draw!")
        
        # Show result in message box
        self.root.after(500, lambda: messagebox.showinfo("Game Over", message))
    
    def new_game(self):
        """Start a new game."""
        self.board = self.env.reset()
        self.game_active = True
        self.update_board_display()
        self.update_status("Agent's turn (X)...")
        self.root.after(500, self.agent_move)


def main():
    """Main entry point for the GUI."""
    import argparse
    
    parser = argparse.ArgumentParser(description='TicTacToe GUI with RL agent')
    parser.add_argument('--model', type=str, default='qtable.pkl', help='Path to trained model file')
    args = parser.parse_args()
    
    root = tk.Tk()
    app = TicTacToeGUI(root, qtable_path=args.model)
    root.mainloop()


if __name__ == "__main__":
    main()
