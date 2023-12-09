
import numpy as np

from open_spiel.python.observation import IIGObserverForPublicInfoGame
import pyspiel

_NUM_PLAYERS = 2
_NUM_ROWS = 3
_NUM_COLS = 3
_NUM_CELLS = _NUM_ROWS * _NUM_COLS
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
    num_distinct_actions=_NUM_CELLS,
    max_chance_outcomes=0,
    num_players=2,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=_NUM_CELLS)


class KamisadoGame(pyspiel.Game):
  """A Python version of the Tic-Tac-Toe game."""

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
  """A python version of the Tic-Tac-Toe state."""

  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self._cur_player = 0
    self._player0_score = 0.0
    self._is_terminal = False
    self.board = np.full((_NUM_ROWS, _NUM_COLS), ".")

  # OpenSpiel (PySpiel) API functions are below. This is the standard set that
  # should be implemented by every perfect-information sequential-move game.

  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""
    return pyspiel.PlayerId.TERMINAL if self._is_terminal else self._cur_player

  def _legal_actions(self, player):
    """Returns a list of legal actions, sorted in ascending order."""
    return [a for a in range(_NUM_CELLS) if self.board[_coord(a)] == "."]

  def _apply_action(self, action):
    """Applies the specified action to the state."""
    self.board[_coord(action)] = "x" if self._cur_player == 0 else "o"
    if _line_exists(self.board):
      self._is_terminal = True
      self._player0_score = 1.0 if self._cur_player == 0 else -1.0
    elif all(self.board.ravel() != "."):
      self._is_terminal = True
    else:
      self._cur_player = 1 - self._cur_player

  def _action_to_string(self, player, action):
    """Action -> string."""
    row, col = _coord(action)
    return "{}({},{})".format("x" if player == 0 else "o", row, col)

  def is_terminal(self):
    """Returns True if the game is over."""
    return self._is_terminal

  def returns(self):
    """Total reward for each player over the course of the game so far."""
    return [self._player0_score, -self._player0_score]

  def __str__(self):
    """String for debug purposes. No particular semantics are required."""
    return _board_to_string(self.board)


class BoardObserver:
  """Observer, conforming to the PyObserver interface (see observation.py)."""

  def __init__(self, params):
    """Initializes an empty observation tensor."""
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")
    # The observation should contain a 1-D tensor in `self.tensor` and a
    # dictionary of views onto the tensor, which may be of any shape.
    # Here the observation is indexed `(cell state, row, column)`.
    shape = (1 + _NUM_PLAYERS, _NUM_ROWS, _NUM_COLS)
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
        cell_state = ".ox".index(state.board[row, col])
        obs[cell_state, row, col] = 1

  def string_from(self, state, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    del player
    return _board_to_string(state.board)


# Helper functions for game details.


def _line_value(line):
  """Checks a possible line, returning the winning symbol if any."""
  if all(line == "x") or all(line == "o"):
    return line[0]


def _line_exists(board):
  """Checks if a line exists, returns "x" or "o" if so, and None otherwise."""
  return (_line_value(board[0]) or _line_value(board[1]) or
          _line_value(board[2]) or _line_value(board[:, 0]) or
          _line_value(board[:, 1]) or _line_value(board[:, 2]) or
          _line_value(board.diagonal()) or
          _line_value(np.fliplr(board).diagonal()))


def _coord(move):
  """Returns (row, col) from an action id."""
  return (move // _NUM_COLS, move % _NUM_COLS)


def _board_to_string(board):
  """Returns a string representation of the board."""
  return "\n".join("".join(row) for row in board)


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, KamisadoGame)


















'''
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

import numpy as np

from open_spiel.python.observation import IIGObserverForPublicInfoGame
import pyspiel

_NUM_PLAYERS = 2
_NUM_ROWS = 8
_NUM_COLS = 8
_NUM_CELLS = _NUM_ROWS * _NUM_COLS
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
    num_distinct_actions=56,  # TODO: 238
    max_chance_outcomes=0,
    num_players=2,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=97)


class Action:
    """Represent player possible action."""

    def __init__(self, player, color, original_loc, new_loc):
        self.player = player
        self.color = color
        self.original_loc = original_loc
        self.new_loc = new_loc

    def __str__(self):
        return f"p{self.player} color: {self.color} original position:{_coord(self.original_loc)} new position:{_coord(self.new_loc)}"

    def __repr__(self):
        return self.__str__()


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
        self.game_board = np.array([
            ['orange', 'blue', 'purple', 'pink', 'yellow', 'red', 'green', 'brown'],
            ['red', 'orange', 'pink', 'green', 'blue', 'yellow', 'brown', 'purple'],
            ['green', 'pink', 'orange', 'red', 'purple', 'brown', 'yellow', 'blue'],
            ['pink', 'purple', 'blue', 'orange', 'brown', 'green', 'red', 'yellow'],
            ['yellow', 'red', 'green', 'brown', 'orange', 'blue', 'purple', 'pink'],
            ['blue', 'yellow', 'brown', 'purple', 'red', 'orange', 'pink', 'green'],
            ['purple', 'brown', 'yellow', 'blue', 'green', 'pink', 'orange', 'red'],
            ['brown', 'green', 'red', 'yellow', 'pink', 'purple', 'blue', 'orange']
        ])
        self.player_board = np.array([
            ['white_orange', 'white_blue', 'white_purple', 'white_pink', 'white_yellow', 'white_red', 'white_green',
             'white_brown'],
            ['', '', '', '', '', '', '', ''],
            ['', '', '', '', '', '', '', ''],
            ['', '', '', '', '', '', '', ''],
            ['', '', '', '', '', '', '', ''],
            ['', '', '', '', '', '', '', ''],
            ['', '', '', '', '', '', '', ''],
            ['black_brown', 'black_green', 'black_red', 'black_yellow', 'black_pink', 'black_purple', 'black_blue',
             'black_orange']
        ])
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

        player_color = _player_color(player)
        piece_color = self.curr_color
        actions = []

        # first move of the game
        if piece_color is None:

            y = _NUM_ROWS - 1
            for x in range(_NUM_COLS):
                piece_color = self.game_board[y][x]
                # straight forward
                for i in range(1, y + 1):
                    if self.player_board[y - i][x] != '':
                        break
                    actions.append(Action(player, piece_color, (y, x), (y - i, x)))
                # diagonal left
                for i in range(1, y + 1):
                    if x - i < 0 or self.player_board[y - i][x - i] != '':
                        break
                    actions.append(Action(player, piece_color, (y, x), (y - i, x - i)))
                # diagonal right
                for i in range(1, y + 1):
                    if x + i >= _NUM_COLS or self.player_board[y - i][x + i] != '':
                        break
                    actions.append(Action(player, piece_color, (y, x), (y - i, x + i)))

        # all other moves
        else:

            player_board = self.player_board.flatten()
            arr_idx = np.where(player_board == player_color + '_' + piece_color)
            y, x = _coord(arr_idx)

            if self._cur_player == 0:
                # straight forward
                for i in range(1, y + 1):
                    if self.player_board[y - i][x] != '':
                        break
                    actions.append(Action(player, piece_color, (y, x), (y - i, x)))
                # diagonal left
                for i in range(1, y + 1):
                    if x - i < 0 or self.player_board[y - i][x - i] != '':
                        break
                    actions.append(Action(player, piece_color, (y, x), (y - i, x - i)))
                # diagonal right
                for i in range(1, y + 1):
                    if x + i >= _NUM_COLS or self.player_board[y - i][x + i] != '':
                        break
                    actions.append(Action(player, piece_color, (y, x), (y - i, x + i)))

            else:
                # straight forward
                for i in range(1, _NUM_ROWS - y):
                    if self.player_board[y + i][x] != '':
                        break
                    actions.append(Action(player, piece_color, (y, x), (y + i, x)))
                # diagonal left
                for i in range(1, _NUM_ROWS - y):
                    if x - i < 0 or self.player_board[y + i][x - i] != '':
                        break
                    actions.append(Action(player, piece_color, (y, x), (y + i, x - i)))
                # diagonal right
                for i in range(1, _NUM_ROWS - y):
                    if x + i >= _NUM_COLS or self.player_board[y + i][x + i] != '':
                        break
                    actions.append(Action(player, piece_color, (y, x), (y + i, x + i)))

            # if there are no valid moves, stay in place
            if len(actions) == 0:
                actions.append(Action(player, piece_color, (y, x), (y, x)))

        return actions

    def _apply_action(self, action):
        """Applies the specified action to the state."""

        # update position of piece
        self.player_board[action.new_loc[0]][action.new_loc[1]] = self.player_board[action.original_loc[0]][action.original_loc[1]]
        self.player_board[action.original_loc[0]][action.original_loc[1]] = ''

        # check for win condition
        if _player_color(0) in ''.join(self.player_board[0]):
            self._is_terminal = True
            self._player0_score = 1.0
        elif _player_color(1) in ''.join(self.player_board[-1]):
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
        return _board_to_string(self.board)


class BoardObserver:
    """Observer, conforming to the PyObserver interface (see observation.py)."""

    def __init__(self, params):
        """Initializes an empty observation tensor."""
        if params:
            raise ValueError(f"Observation parameters not supported; passed {params}")
        # The observation should contain a 1-D tensor in `self.tensor` and a
        # dictionary of views onto the tensor, which may be of any shape.
        # Here the observation is indexed `(cell state, row, column)`.
        shape = (1 + _NUM_PLAYERS, _NUM_ROWS, _NUM_COLS)
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
                cell_state = state.game_board[row][col] + ' ' + state.player_board[row][col]
                obs[cell_state, row, col] = 1

    def string_from(self, state, player):
        """Observation of `state` from the PoV of `player`, as a string."""
        del player
        return _board_to_string(state.game_board, state.player_board)


# Helper functions for game details.

def _coord(move):
    """Returns (row, col) from an action id."""
    return (move // _NUM_COLS, move % _NUM_COLS)


def _player_color(player):
    return 'black' if player == 0 else 'white'


def _board_to_string(game_board, player_board):
    """Returns a string representation of the board."""
    out = game_board
    for i in range(_NUM_ROWS):
        for j in range(_NUM_COLS):
            if player_board[i][j] != '':
                out[i][j] += ' (' + player_board[i][j] + ')'
    return "\n".join("\t\t".join(row) for row in out)


# Register the game with the OpenSpiel library
pyspiel.register_game(_GAME_TYPE, KamisadoGame)
'''
