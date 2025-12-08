# RL-TicTacToe

A Reinforcement Learning (RL) agent that learns to play TicTacToe, enhanced with CSV data integration and a graphical user interface.

## Features

- **Q-Learning Agent**: Trains using reinforcement learning to master TicTacToe
- **CSV Data Integration**: Pre-trains using TicTacToe game data from Kaggle/UCI dataset
- **Graphical Interface**: Play against the trained agent using a tkinter-based GUI
- **Console Interface**: Traditional command-line play mode also available
- **Modular Architecture**: Separate components for game logic, RL agent, training, and GUI

## Project Structure

```
RL-TicTacToe/
├── game.py          # TicTacToe game environment
├── agent.py         # Q-Learning agent implementation
├── train.py         # Training script with CSV data integration
├── play.py          # Console-based play interface
├── gui.py           # Tkinter GUI for playing against agent
├── data_loader.py   # CSV data loading and preprocessing
├── evaluate.py      # Agent evaluation script
├── read_table.py    # Q-table inspection utility
├── tic-tac-toe.data # Training data (CSV format)
└── qtable.pkl       # Trained Q-table (generated after training)
```

## Requirements

- Python 3.7 or higher
- tkinter (usually included with Python)

No external dependencies required! This project uses only Python standard library.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Dokkae6949/RL-TicTacToe.git
cd RL-TicTacToe
```

2. That's it! No additional installation needed.

## Usage

### Training the Agent

The training process includes two phases:
1. **Pre-training** with CSV data (optional but recommended)
2. **Main RL training** through self-play

#### Basic Training

```bash
python train.py
```

This will:
- Load and pre-train using the TicTacToe CSV data (`tic-tac-toe.data`)
- Continue training for 50,000 episodes using Q-learning
- Save the trained model to `qtable.pkl`

#### Advanced Training Options

```bash
# Train with custom number of episodes
python train.py --episodes 100000

# Skip CSV pre-training
python train.py --no-csv

# Use custom CSV data file
python train.py --csv-file my_data.csv

# Save model to custom location
python train.py --output my_model.pkl
```

### CSV Data Format

The CSV data should be in UCI TicTacToe format:
- 9 columns for board positions (row-major order: top-left to bottom-right)
- Each position: `x`, `o`, or `b` (blank)
- 10th column: outcome (`positive` for X wins, `negative` for X doesn't win)
- Missing values (`?`) are automatically handled as blank spaces

Example:
```
x,x,x,x,o,o,x,o,o,positive
x,x,o,x,o,o,x,x,o,negative
```

If the data file doesn't exist, the system will automatically generate synthetic training data.

### Playing with GUI (Recommended)

Launch the graphical interface:

```bash
python gui.py
```

Features:
- Click on empty cells to make your move
- You play as O (second player)
- Agent plays as X (first player)
- "New Game" button to start over
- Visual feedback for game status

With custom model:
```bash
python gui.py --model my_model.pkl
```

### Playing in Console

For traditional command-line play:

```bash
python play.py
```

- Input cell numbers 0-8 (top-left to bottom-right)
- Board layout:
  ```
  0 | 1 | 2
  ---------
  3 | 4 | 5
  ---------
  6 | 7 | 8
  ```

### Evaluating the Agent

Test the agent's performance against a random opponent:

```bash
python evaluate.py
```

This runs 10,000 games and reports win/draw/loss rates.

## How It Works

### Q-Learning Algorithm

The agent uses Q-learning, a model-free reinforcement learning algorithm:

1. **State Representation**: Board configuration + current player
2. **Actions**: Placing a mark in an empty cell (0-8)
3. **Rewards**: 
   - +1 for winning
   - 0 for draw
   - -1 for losing
4. **Exploration vs Exploitation**: ε-greedy strategy with decay

### CSV Data Integration

The training process leverages historical game data:

1. **Data Preprocessing**: 
   - Converts symbols to numeric values (x→1, o→-1, b→0)
   - Handles missing values (?)
   - Encodes outcomes (win/loss/draw)

2. **Pre-training Phase**:
   - Samples game states from the CSV data
   - Lets the agent explore and learn from these states
   - Bootstraps the Q-table with game knowledge

3. **Main Training Phase**:
   - Agent plays against random opponent
   - Refines strategy through self-play
   - Continues learning from experience

### GUI Architecture

The tkinter GUI provides:
- **Grid Layout**: 3x3 button grid representing the board
- **Event Handling**: Click handlers for player moves
- **State Management**: Synchronizes game state with display
- **Status Updates**: Real-time game status messages
- **Turn Management**: Alternates between player and agent turns

## Performance

After training with CSV pre-training:
- Win rate: ~90%+ against random opponent
- Draw rate: ~8-10%
- Loss rate: <2%

Results may vary based on training parameters and random seed.

## Customization

### Modifying Training Parameters

Edit `train.py`:
- `EPISODES`: Number of training episodes (default: 50000)
- `CSV_PRETRAIN_EPISODES`: Number of CSV pre-training episodes (default: 5000)
- `EPSILON_DECAY`: Exploration decay rate (default: 0.99995)
- `MIN_EPSILON`: Minimum exploration rate (default: 0.05)

### Modifying Agent Parameters

Edit `agent.py`:
- `alpha`: Learning rate (default: 0.5)
- `gamma`: Discount factor (default: 0.99)
- `epsilon`: Initial exploration rate (default: 1.0)

## Troubleshooting

### GUI doesn't launch
- Make sure tkinter is installed: `python -m tkinter`
- On Linux, you may need: `sudo apt-get install python3-tk`

### Model file not found
- Run training first: `python train.py`
- Or specify custom model path: `python gui.py --model qtable.pkl`

### CSV data not found
- The system will auto-generate synthetic data
- Or download UCI TicTacToe dataset and save as `tic-tac-toe.data`

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- TicTacToe dataset from UCI Machine Learning Repository
- Q-Learning algorithm implementation based on Sutton & Barto's RL book
- GUI built with Python's tkinter module

## Future Enhancements

- [ ] Add different difficulty levels
- [ ] Implement deeper RL algorithms (DQN, Policy Gradient)
- [ ] Add multiplayer mode (human vs human)
- [ ] Enhanced GUI with animations
- [ ] Tournament mode against multiple AI strategies
- [ ] Save and replay game history
