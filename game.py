from typing import List, Tuple, Optional

class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [0] * 9   # 0=leer, 1=X, -1=O
        self.current_player = 1
        self.done = False
        self.winner = None
        return tuple(self.board)

    def legal_actions(self) -> List[int]:
        return [i for i, v in enumerate(self.board) if v == 0]

    def step(self, action: int) -> Tuple[Tuple[int], int, bool, dict]:
        if self.done:
            raise RuntimeError("Game already finished")
        if self.board[action] != 0:
            self.done = True
            self.winner = -self.current_player
            return tuple(self.board), -1, True, {"illegal": True}

        self.board[action] = self.current_player
        self._check_done()

        if self.done:
            if self.winner == 0:
                reward = 0
            elif self.winner == self.current_player:
                reward = 1
            else:
                reward = -1
            return tuple(self.board), reward, True, {}
        else:
            self.current_player *= -1
            return tuple(self.board), 0, False, {}

    def _check_done(self):
        b = self.board
        lines = [
            (0,1,2),(3,4,5),(6,7,8),
            (0,3,6),(1,4,7),(2,5,8),
            (0,4,8),(2,4,6)
        ]
        for a,b1,c in lines:
            s = b[a] + b[b1] + b[c]
            if s == 3:
                self.done, self.winner = True, 1
                return
            if s == -3:
                self.done, self.winner = True, -1
                return
        if all(x != 0 for x in b):
            self.done, self.winner = True, 0

    def render(self):
        symbols = {1:"X",-1:"O",0:" "}
        for r in range(3):
            row = [symbols[self.board[3*r + c]] for c in range(3)]
            print("|".join(row))
            if r<2:
                print("-+-+-")