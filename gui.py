"""
Basic TicTacToe GUI using tkinter.
Allows playing against the trained RL agent with a graphical interface.
"""

import tkinter as tk
from tkinter import messagebox, ttk
from game import TicTacToe
from agent import QLearningAgent
from utils import make_state_key
import os


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
        
        # Configuration state
        self.dark_mode = False
        self.agent_starts = True  # Agent goes first by default
        self.player_symbol = "O"
        self.agent_symbol = "X"
        
        # Player assignments (1 is always first, -1 is always second)
        self.player_id = -1  # Player is second by default
        self.agent_id = 1    # Agent is first by default
        
        # Color schemes
        self.colors = {
            'light': {
                'bg': '#f0f0f0',
                'button': '#ffffff',
                'text': '#000000',
                'button_active': '#e0e0e0',
                'status_bg': '#ffffff',
                'frame_bg': '#f0f0f0'
            },
            'dark': {
                'bg': '#2b2b2b',
                'button': '#3c3c3c',
                'text': '#ffffff',
                'button_active': '#4a4a4a',
                'status_bg': '#3c3c3c',
                'frame_bg': '#2b2b2b'
            }
        }
        
        # Game state
        self.board = self.env.reset()
        self.game_active = True
        
        # UI Components
        self.setup_ui()
        
        # Apply initial theme
        self.apply_theme()
        
        # Start game based on configuration
        if self.agent_starts:
            self.update_status(f"Agent's turn ({self.agent_symbol})...")
            self.root.after(500, self.agent_move)
        else:
            self.update_status(f"Your turn ({self.player_symbol})")
    
    def setup_ui(self):
        """Create the GUI components."""
        # Configuration frame at the top
        config_frame = tk.Frame(self.root)
        config_frame.pack(padx=20, pady=10)
        
        # Dark mode toggle
        self.dark_mode_btn = tk.Button(
            config_frame,
            text="🌙 Dark Mode",
            font=("Arial", 10),
            command=self.toggle_dark_mode,
            padx=10
        )
        self.dark_mode_btn.grid(row=0, column=0, padx=5)
        
        # Who starts selection
        tk.Label(config_frame, text="First player:", font=("Arial", 10)).grid(row=0, column=1, padx=5)
        self.first_player_var = tk.StringVar(value="agent")
        ttk.Combobox(
            config_frame, 
            textvariable=self.first_player_var,
            values=["agent", "player"],
            state="readonly",
            width=10,
            font=("Arial", 10)
        ).grid(row=0, column=2, padx=5)
        
        # Symbol configuration
        tk.Label(config_frame, text="Your symbol:", font=("Arial", 10)).grid(row=0, column=3, padx=5)
        self.player_symbol_var = tk.StringVar(value="O")
        self.player_symbol_entry = tk.Entry(config_frame, textvariable=self.player_symbol_var, width=3, font=("Arial", 10))
        self.player_symbol_entry.grid(row=0, column=4, padx=5)
        
        tk.Label(config_frame, text="Agent symbol:", font=("Arial", 10)).grid(row=0, column=5, padx=5)
        self.agent_symbol_var = tk.StringVar(value="X")
        self.agent_symbol_entry = tk.Entry(config_frame, textvariable=self.agent_symbol_var, width=3, font=("Arial", 10))
        self.agent_symbol_entry.grid(row=0, column=6, padx=5)
        
        # Status label
        self.status_label = tk.Label(
            self.root,
            text=f"Agent plays {self.agent_symbol}, You play {self.player_symbol}",
            font=("Arial", 14),
            pady=10
        )
        self.status_label.pack()
        
        # Game board frame
        self.board_frame = tk.Frame(self.root)
        self.board_frame.pack(padx=20, pady=10)
        
        # Create 3x3 grid of buttons
        self.buttons = []
        for i in range(3):
            row_buttons = []
            for j in range(3):
                btn = tk.Button(
                    self.board_frame,
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
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(pady=10)
        
        self.new_game_btn = tk.Button(
            self.control_frame,
            text="New Game",
            font=("Arial", 12),
            command=self.new_game,
            padx=20
        )
        self.new_game_btn.pack(side=tk.LEFT, padx=5)
        
        self.quit_btn = tk.Button(
            self.control_frame,
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
        
        # Store config frame for theme application
        self.config_frame = config_frame
    
    def get_symbol_map(self):
        """Get symbol mapping based on configuration."""
        return {self.agent_id: self.agent_symbol, self.player_id: self.player_symbol, 0: " "}
    
    def update_board_display(self):
        """Update button texts to reflect current board state."""
        symbols = self.get_symbol_map()
        theme = self.colors['dark' if self.dark_mode else 'light']
        
        for i in range(3):
            for j in range(3):
                idx = i * 3 + j
                symbol = symbols[self.env.board[idx]]
                self.buttons[i][j].config(
                    text=symbol,
                    bg=theme['button'],
                    fg=theme['text'],
                    activebackground=theme['button_active']
                )
                
                # Disable button if cell is occupied
                if self.env.board[idx] != 0 or not self.game_active:
                    self.buttons[i][j].config(state="disabled")
                else:
                    self.buttons[i][j].config(state="normal")
    
    def update_status(self, message):
        """Update the status label."""
        self.status_label.config(text=message)
        self.root.update()
    
    def toggle_dark_mode(self):
        """Toggle between light and dark mode."""
        self.dark_mode = not self.dark_mode
        self.dark_mode_btn.config(text="☀️ Light Mode" if self.dark_mode else "🌙 Dark Mode")
        self.apply_theme()
    
    def apply_theme(self):
        """Apply the current theme to all UI components."""
        theme = self.colors['dark' if self.dark_mode else 'light']
        
        # Apply to root window
        self.root.config(bg=theme['bg'])
        
        # Apply to frames
        self.config_frame.config(bg=theme['frame_bg'])
        self.board_frame.config(bg=theme['frame_bg'])
        self.control_frame.config(bg=theme['frame_bg'])
        
        # Apply to labels
        self.status_label.config(bg=theme['status_bg'], fg=theme['text'])
        self.info_label.config(bg=theme['bg'], fg='gray')
        
        # Apply to configuration labels
        for widget in self.config_frame.winfo_children():
            if isinstance(widget, tk.Label):
                widget.config(bg=theme['frame_bg'], fg=theme['text'])
        
        # Apply to buttons
        self.dark_mode_btn.config(bg=theme['button'], fg=theme['text'], activebackground=theme['button_active'])
        self.new_game_btn.config(bg=theme['button'], fg=theme['text'], activebackground=theme['button_active'])
        self.quit_btn.config(bg=theme['button'], fg=theme['text'], activebackground=theme['button_active'])
        
        # Apply to game board buttons
        self.update_board_display()
    
    def player_move(self, action):
        """Handle player's move."""
        if not self.game_active or self.env.current_player != self.player_id:
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
        self.update_status(f"Agent's turn ({self.agent_symbol})...")
        self.root.after(500, self.agent_move)
    
    def agent_move(self):
        """Agent makes its move."""
        if not self.game_active or self.env.current_player != self.agent_id:
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
        self.update_status(f"Your turn ({self.player_symbol})")
    
    def end_game(self):
        """Handle game end."""
        self.game_active = False
        self.update_board_display()
        
        if self.env.winner == self.agent_id:
            message = "Agent wins!"
            self.update_status(f"Game Over - Agent ({self.agent_symbol}) wins!")
        elif self.env.winner == self.player_id:
            message = "You win!"
            self.update_status(f"Game Over - You ({self.player_symbol}) win!")
        else:
            message = "It's a draw!"
            self.update_status("Game Over - Draw!")
        
        # Show result in message box
        self.root.after(500, lambda: messagebox.showinfo("Game Over", message))
    
    def new_game(self):
        """Start a new game with current configuration."""
        # Update configuration from UI
        self.agent_starts = self.first_player_var.get() == "agent"
        
        # Validate and update symbols
        new_player_symbol = self.player_symbol_var.get().strip()
        new_agent_symbol = self.agent_symbol_var.get().strip()
        
        if new_player_symbol and new_agent_symbol and new_player_symbol != new_agent_symbol:
            self.player_symbol = new_player_symbol[:1]  # Use only first character
            self.agent_symbol = new_agent_symbol[:1]
        else:
            # Reset to defaults if invalid
            if not new_player_symbol or not new_agent_symbol:
                messagebox.showwarning("Invalid Symbols", "Symbols cannot be empty. Using defaults: O and X")
            elif new_player_symbol == new_agent_symbol:
                messagebox.showwarning("Invalid Symbols", "Player and agent must use different symbols. Using defaults: O and X")
            
            self.player_symbol = "O"
            self.agent_symbol = "X"
            self.player_symbol_var.set(self.player_symbol)
            self.agent_symbol_var.set(self.agent_symbol)
        
        # Update player IDs based on who starts
        if self.agent_starts:
            self.agent_id = 1   # Agent goes first
            self.player_id = -1  # Player goes second
        else:
            self.player_id = 1   # Player goes first
            self.agent_id = -1   # Agent goes second
        
        # Reset game
        self.board = self.env.reset()
        self.game_active = True
        self.update_board_display()
        
        # Start game based on configuration
        if self.agent_starts:
            self.update_status(f"Agent's turn ({self.agent_symbol})...")
            self.root.after(500, self.agent_move)
        else:
            self.update_status(f"Your turn ({self.player_symbol})")


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
