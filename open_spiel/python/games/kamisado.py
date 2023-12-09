# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as python3
"""Kamisado, implemented in Python.

This is a demonstration of implementing a deterministic perfect-information
game in Python.

Python games are significantly slower than C++, but it may still be suitable
for prototyping or for small games.

It is possible to run C++ algorithms on Python-implemented games. This is likely
to have good performance if the algorithm simply extracts a game tree and then
works with that (e.g. CFR algorithms). It is likely to be poor if the algorithm
relies on processing and updating states as it goes, e.g., MCTS.
"""

import copy
import numpy as np

from open_spiel.python.observation import IIGObserverForPublicInfoGame
import pyspiel

_NUM_PLAYERS = 2
_NUM_ROWS = 8
_NUM_COLS = 8
_NUM_CELLS = _NUM_ROWS * _NUM_COLS
_GAME_BOARD = np.array([
            ['ORG', 'BLU', 'PRP', 'PNK', 'YLW', 'RED', 'GRN', 'BWN'],
            ['RED', 'ORG', 'PNK', 'GRN', 'BLU', 'YLW', 'BWN', 'PRP'],
            ['GRN', 'PNK', 'ORG', 'RED', 'PRP', 'BWN', 'YLW', 'BLU'],
            ['PNK', 'PRP', 'BLU', 'ORG', 'BWN', 'GRN', 'RED', 'YLW'],
            ['YLW', 'RED', 'GRN', 'BWN', 'ORG', 'BLU', 'PRP', 'PNK'],
            ['BLU', 'YLW', 'BWN', 'PRP', 'RED', 'ORG', 'PNK', 'GRN'],
            ['PRP', 'BWN', 'YLW', 'BLU', 'GRN', 'PNK', 'ORG', 'RED'],
            ['BWN', 'GRN', 'RED', 'YLW', 'PNK', 'PRP', 'BLU', 'ORG']
        ], dtype='object')

class Action:
    """Represent player possible action."""

    def __init__(self, player, original_loc, new_loc):
        self.player = player
        self.original_loc = original_loc
        self.new_loc = new_loc

    def __str__(self):
        return f"p{self.player} original position:{self.original_loc} new position:{self.new_loc}"

    def __repr__(self):
        return self.__str__()

def create_possible_actions():
    # player0
    actions = []
    for i1 in range(_NUM_ROWS):
        for j1 in range(_NUM_COLS):
            actions.append(Action(0, (i1, j1), (i1, j1)))
            actions.append(Action(1, (i1, j1), (i1, j1)))
            for j2 in range(_NUM_COLS):
                # player 0
                for i2 in range(i1):
                    if j2 == j1 or abs(j1-j2) == i1-i2:
                        actions.append(Action(0, (i1, j1), (i2, j2)))
                # player 0
                for i3 in range(_NUM_ROWS-1, i1, -1):
                    if j2 == j1 or abs(j1-j2) == i3-i1:
                        actions.append(Action(1, (i1, j1), (i3, j2)))
    return actions

#print("TEST 1111")

_ACTIONS = create_possible_actions()
_ACTIONS_STR = [str(action) for action in _ACTIONS]

_GAME_TYPE = pyspiel.GameType(
    short_name="python_kamisado",
    long_name="Python Kamisado",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification={})
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=len(_ACTIONS),
    max_chance_outcomes=0,
    num_players=2,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=97)

class KamisadoGame(pyspiel.Game):
    """A Python version of the Kamisado game."""

    def __init__(self, params=None):
        super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

    def new_initial_state(self):
        """Returns a state corresponding to the start of a game."""
        return KamisadoState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        """Returns an object used for observing game state."""
        if ((iig_obs_type is None) or
                (iig_obs_type.public_info and not iig_obs_type.perfect_recall)):
            return BoardObserver(params)
        else:
            return IIGObserverForPublicInfoGame(iig_obs_type, params)


class KamisadoState(pyspiel.State):
    """A python version of the Kamisado state."""

    def __init__(self, game):
        """Constructor; should only be called by Game.new_initial_state."""
        super().__init__(game)
        self._cur_player = 0
        self._player0_score = 0.0
        self._is_terminal = False
        self.curr_color = None
        self.game_board = _GAME_BOARD
        self.player_board = np.array([
            ['W-ORG', 'W-BLU', 'W-PRP', 'W-PNK', 'W-YLW', 'W-RED', 'W-GRN', 'W-BWN'],
            ['', '', '', '', '', '', '', ''],
            ['', '', '', '', '', '', '', ''],
            ['', '', '', '', '', '', '', ''],
            ['', '', '', '', '', '', '', ''],
            ['', '', '', '', '', '', '', ''],
            ['', '', '', '', '', '', '', ''],
            ['B-BWN', 'B-GRN', 'B-RED', 'B-YLW', 'B-PNK', 'B-PRP', 'B-BLU', 'B-ORG']
        ], dtype='object')
        # self.board = np.full((_NUM_ROWS, _NUM_COLS), ".")

    # OpenSpiel (PySpiel) API functions are below. This is the standard set that
    # should be implemented by every perfect-information sequential-move game.

    def current_player(self):
        """Returns id of the next player to move, or TERMINAL if game is over."""
        return pyspiel.PlayerId.TERMINAL if self._is_terminal else self._cur_player

    def _legal_actions(self, player):
        """Returns a list of legal actions, sorted in ascending order."""
        # return [a for a in range(_NUM_CELLS) if self.board[_coord(a)] == "."]
        return self.get_legal_actions(player)

    def get_legal_actions(self, player):
        """Returns a list of legal actions."""
        assert player >= 0

        #print("TEST0")
        player_color = _player_color(player)
        piece_color = self.curr_color
        actions = []

        # first move of the game
        if piece_color is None:

            #print("TEST1")
            y = _NUM_ROWS - 1
            for x in range(_NUM_COLS):
                piece_color = self.game_board[y][x]
                # straight forward
                for i in range(1, y + 1):
                    if self.player_board[y - i][x] != '':
                        break
                    actions.append(Action(player, (y, x), (y - i, x)))
                # diagonal left
                for i in range(1, y + 1):
                    if x - i < 0 or self.player_board[y - i][x - i] != '':
                        break
                    actions.append(Action(player, (y, x), (y - i, x - i)))
                # diagonal right
                for i in range(1, y + 1):
                    if x + i >= _NUM_COLS or self.player_board[y - i][x + i] != '':
                        break
                    actions.append(Action(player, (y, x), (y - i, x + i)))

        # all other moves
        else:

            #print("TEST2")
            player_board = self.player_board.flatten()
            #print("TEST3")
            #arr_idx = np.where(player_board == player_color + '_' + piece_color)[0][0]
            #print(player_color + '_' + piece_color)
            arr_idx = np.argwhere(player_board == player_color + '-' + piece_color)
            #print(arr_idx)
            #print(type(arr_idx))
            arr_idx = arr_idx.flat[0]

            #print("TEST4")
            y, x = _coord(arr_idx)

            #print("TEST5")
            if self._cur_player == 0:
                # straight forward
                for i in range(1, y + 1):
                    if self.player_board[y - i][x] != '':
                        break
                    actions.append(Action(player, (y, x), (y - i, x)))
                # diagonal left
                for i in range(1, y + 1):
                    if x - i < 0 or self.player_board[y - i][x - i] != '':
                        break
                    actions.append(Action(player, (y, x), (y - i, x - i)))
                # diagonal right
                for i in range(1, y + 1):
                    if x + i >= _NUM_COLS or self.player_board[y - i][x + i] != '':
                        break
                    actions.append(Action(player, (y, x), (y - i, x + i)))

                #print("TEST6")
            else:
                # straight forward
                #print(_NUM_ROWS - y)
                #print(type(_NUM_ROWS - y))
                for i in range(1, _NUM_ROWS - y):
                    if self.player_board[y + i][x] != '':
                        break
                    actions.append(Action(player, (y, x), (y + i, x)))
                # diagonal left
                for i in range(1, _NUM_ROWS - y):
                    if x - i < 0 or self.player_board[y + i][x - i] != '':
                        break
                    actions.append(Action(player, (y, x), (y + i, x - i)))
                # diagonal right
                for i in range(1, _NUM_ROWS - y):
                    if x + i >= _NUM_COLS or self.player_board[y + i][x + i] != '':
                        break
                    actions.append(Action(player, (y, x), (y + i, x + i)))

            # if there are no valid moves, stay in place
            if len(actions) == 0:
                actions.append(Action(player, (y, x), (y, x)))

        #print(actions)
        #return actions
        actions_idx = [_ACTIONS_STR.index(str(action)) for action in actions]
        actions_idx.sort()
        #print(actions_idx)
        return actions_idx

    def _apply_action(self, action_id):
        """Applies the specified action to the state."""

        action = _ACTIONS[action_id]
        #print("aaaaaaaaaaaa")
        #print(action)
        #print(type(action))

        # update position of piece
        if action.new_loc != action.original_loc:
            self.player_board[action.new_loc[0]][action.new_loc[1]] = self.player_board[action.original_loc[0]][action.original_loc[1]]
            self.player_board[action.original_loc[0]][action.original_loc[1]] = ''

        # check for win condition
        if _player_color(0) + '-' in ''.join(self.player_board[0]):
            self._is_terminal = True
            self._player0_score = 1.0
        elif _player_color(1) + '-' in ''.join(self.player_board[-1]):
            self._is_terminal = True
            self._player0_score = -1.0

        # transfer play to other player
        else:
            self.curr_color = self.game_board[action.new_loc[0]][action.new_loc[1]]
            self._cur_player = 1 - self._cur_player

    # TODO: check for stalemate?

    def _action_to_string(self, player, action):
        """Action -> string."""
        return action.__str__()

    def is_terminal(self):
        """Returns True if the game is over."""
        return self._is_terminal

    def returns(self):
        """Total reward for each player over the course of the game so far."""
        return [self._player0_score, -self._player0_score]

    def __str__(self):
        """String for debug purposes. No particular semantics are required."""
        return _board_to_string(self.game_board, self.player_board)


class BoardObserver:
    """Observer, conforming to the PyObserver interface (see observation.py)."""

    def __init__(self, params):
        """Initializes an empty observation tensor."""
        if params:
            raise ValueError(f"Observation parameters not supported; passed {params}")
        # The observation should contain a 1-D tensor in `self.tensor` and a
        # dictionary of views onto the tensor, which may be of any shape.
        # Here the observation is indexed `(cell state, row, column)`.
        #shape = (1 + _NUM_PLAYERS, _NUM_ROWS, _NUM_COLS)
        shape = (17, _NUM_ROWS, _NUM_COLS)
        self.tensor = np.zeros(np.prod(shape), np.float32)
        self.dict = {"observation": np.reshape(self.tensor, shape)}

    def set_from(self, state, player):
        """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
        del player
        # We update the observation via the shaped tensor since indexing is more
        # convenient than with the 1-D tensor. Both are views onto the same memory.
        obs = self.dict["observation"]
        obs.fill(0)
        for row in range(_NUM_ROWS):
            for col in range(_NUM_COLS):
                pieces_str = ['', 'W-ORG', 'W-BLU', 'W-PRP', 'W-PNK', 'W-YLW', 'W-RED', 'W-GRN', 'W-BWN', 'B-BWN', 'B-GRN', 'B-RED', 'B-YLW', 'B-PNK', 'B-PRP', 'B-BLU', 'B-ORG']
                cell_state = pieces_str.index(state.player_board[row, col])
                #print(type(cell_state))
                obs[cell_state, row, col] = 1
                #cell_state = state.game_board[row][col] + ' ' + state.player_board[row][col]

    def string_from(self, state, player):
        """Observation of `state` from the PoV of `player`, as a string."""
        del player
        return _board_to_string(state.game_board, state.player_board)


# Helper functions for game details.

def _coord(move):
    """Returns (row, col) from an action id."""
    return (move // _NUM_COLS, move % _NUM_COLS)


def _player_color(player):
    return 'W' if player == 0 else 'B'


def _board_to_string(game_board, player_board):
    """Returns a string representation of the board."""
    out = copy.deepcopy(game_board)
    for i in range(_NUM_ROWS):
        for j in range(_NUM_COLS):
            if player_board[i][j] == '':
                out[i][j] = (out[i][j]).ljust(12)
            else:
                out[i][j] = (out[i][j] + ' [' + player_board[i][j] + ']').ljust(12)
                # out[i][j] = "{: >15}".format(out[i][j] + ' (' + player_board[i][j] + ')')
                #out[i][j] += ' (' + player_board[i][j] + ')'
    return "\n".join(" | ".join(row) for row in out)


# Register the game with the OpenSpiel library
pyspiel.register_game(_GAME_TYPE, KamisadoGame)


ks = KamisadoState(KamisadoGame())
print(str(ks))

print(ks.get_legal_actions(0))
ks._apply_action(ks.get_legal_actions(0)[0])

print(str(ks))
bo = BoardObserver(False)
bo.string_from(ks, 0)
#print(bo.dict)

