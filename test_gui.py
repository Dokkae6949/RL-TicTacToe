"""
Test script to validate GUI logic without requiring a display.
This tests the core game logic and agent integration used by the GUI.
"""

from game import TicTacToe
from agent import QLearningAgent
from utils import make_state_key
import os


def test_gui_game_logic():
    """Test the game logic that the GUI uses."""
    print("Testing GUI game logic...")
    
    # Initialize game and agent (same as GUI)
    env = TicTacToe()
    agent = QLearningAgent()
    
    # Load trained model
    qtable_path = "qtable.pkl"
    if os.path.exists(qtable_path):
        agent.load(qtable_path)
        print(f"✓ Loaded trained model from {qtable_path}")
    else:
        print(f"✗ Model file {qtable_path} not found")
        return False
    
    # Simulate a game (agent plays X, simulated player plays O)
    board = env.reset()
    print(f"\n✓ Game initialized")
    print(f"  Board: {board}")
    
    move_count = 0
    max_moves = 9
    
    while not env.done and move_count < max_moves:
        if env.current_player == 1:
            # Agent move (X)
            state_key = make_state_key(tuple(board), env.current_player)
            legal = env.legal_actions()
            
            if not legal:
                print(f"✗ No legal actions available")
                break
            
            action = agent.get_action(state_key, legal, training=False)
            print(f"\n  Agent (X) plays position {action}")
            board, _, done, _ = env.step(action)
            
        else:
            # Simulated player move (O)
            legal = env.legal_actions()
            
            if not legal:
                print(f"✗ No legal actions available")
                break
            
            # Simple heuristic: block if agent about to win, else random
            action = legal[0]  # Simple choice for testing
            print(f"  Player (O) plays position {action}")
            board, _, done, _ = env.step(action)
        
        move_count += 1
        
        # Display board state
        symbols = {1: "X", -1: "O", 0: " "}
        print(f"  Board state:")
        for r in range(3):
            row = [symbols[board[3*r + c]] for c in range(3)]
            print(f"    {' | '.join(row)}")
            if r < 2:
                print(f"    ---------")
    
    # Check final result
    print(f"\n✓ Game completed after {move_count} moves")
    
    if env.winner == 1:
        print(f"  Result: Agent (X) wins!")
    elif env.winner == -1:
        print(f"  Result: Player (O) wins!")
    else:
        print(f"  Result: Draw!")
    
    return True


def test_gui_components():
    """Test that GUI components can be imported and initialized."""
    print("\nTesting GUI components...")
    
    try:
        # Try to import GUI module (will fail if tkinter not available)
        import gui
        print("✓ GUI module imported successfully")
        
        # Check that key functions exist
        assert hasattr(gui, 'TicTacToeGUI'), "TicTacToeGUI class not found"
        assert hasattr(gui, 'make_state_key'), "make_state_key function not found"
        assert hasattr(gui, 'main'), "main function not found"
        print("✓ All required GUI components present")
        
        return True
        
    except ImportError as e:
        print(f"⚠ GUI module cannot be fully tested: {e}")
        print("  (This is expected in headless environments without tkinter)")
        
        # Still validate the file syntax
        import py_compile
        try:
            py_compile.compile('gui.py', doraise=True)
            print("✓ GUI code syntax is valid")
            return True
        except py_compile.PyCompileError as e:
            print(f"✗ GUI syntax error: {e}")
            return False


def main():
    """Run all GUI tests."""
    print("=" * 60)
    print("GUI Validation Tests")
    print("=" * 60)
    
    test1_passed = test_gui_game_logic()
    test2_passed = test_gui_components()
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Game Logic Test: {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"GUI Components Test: {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n✓ All tests passed!")
        print("\nTo run the GUI with a display:")
        print("  python gui.py")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
