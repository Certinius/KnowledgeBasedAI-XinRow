from __future__ import annotations
from abc import abstractmethod
from logging import root
import numpy as np
from typing import TYPE_CHECKING

from tree import Tree
if TYPE_CHECKING:
    from heuristics import Heuristic
    from board import Board


class PlayerController:
    """Abstract class defining a player
    """
    def __init__(self, player_id: int, game_n: int, heuristic: Heuristic) -> None:
        """
        Args:
            player_id (int): id of a player, can take values 1 or 2 (0 = empty)
            game_n (int): n in a row required to win
            heuristic (Heuristic): heuristic used by the player
        """
        self.player_id = player_id
        self.game_n = game_n
        self.heuristic = heuristic


    def get_eval_count(self) -> int:
        """
        Returns:
            int: The amount of times the heuristic was used to evaluate a board state
        """
        return self.heuristic.eval_count
    

    def __str__(self) -> str:
        """
        Returns:
            str: representation for representing the player on the board
        """
        if self.player_id == 1:
            return 'X'
        return 'O'
        

    @abstractmethod
    def make_move(self, board: Board) -> int:
        """Gets the column for the player to play in

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """
        pass


class MinMaxPlayer(PlayerController):
    """Class for the minmax player using the minmax algorithm
    Inherits from Playercontroller
    """
    def __init__(self, player_id: int, game_n: int, depth: int, heuristic: Heuristic) -> None:
        """
        Args:
            player_id (int): id of a player, can take values 1 or 2 (0 = empty)
            game_n (int): n in a row required to win
            depth (int): the max search depth
            heuristic (Heuristic): heuristic used by the player
        """
        super().__init__(player_id, game_n, heuristic)
        self.depth: int = depth


    def make_move(self, board: Board) -> int:
        """Gets the column for the player to play in

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """

        root = Node(board, self.player_id)
        Tree.expand_node(root, self.depth)

        def minimax_node(node, depth, maximizing_player):
            winner = self.heuristic.winning(node.board.get_board_state(), self.game_n)
            if depth == 0 or winner != 0:
                return self.heuristic.evaluate_board(self.player_id, node.board)

            if maximizing_player:
                max_eval = -np.inf
                best_move = None
                for child in node.children:
                    eval = minimax_node(child, depth - 1, False)
                    if eval > max_eval:
                        max_eval = eval
                        best_move = child
                if depth == self.depth and best_move is not None:
                    # Find which move (column) led to best_move
                    for col in range(board.width):
                        if board.is_valid(col):
                            new_board = board.get_new_board(col, self.player_id)
                            if np.array_equal(new_board.get_board_state(), best_move.board.get_board_state()):
                                return col
                    return 0  # fallback
                return max_eval
            else:
                min_eval = np.inf
                for child in node.children:
                    eval = minimax_node(child, depth - 1, True)
                    if eval < min_eval:
                        min_eval = eval
                return min_eval

        move = minimax_node(root, self.depth, True)
        return move


    

class AlphaBetaPlayer(PlayerController):
    """Class for the minmax player using the minmax algorithm with alpha-beta pruning
    Inherits from Playercontroller
    """
    def __init__(self, player_id: int, game_n: int, depth: int, heuristic: Heuristic) -> None:
        """
        Args:
            player_id (int): id of a player, can take values 1 or 2 (0 = empty)
            game_n (int): n in a row required to win
            depth (int): the max search depth
            heuristic (Heuristic): heuristic used by the player
        """
        super().__init__(player_id, game_n, heuristic)
        self.depth: int = depth


    def make_move(self, board: Board) -> int:
        """Gets the column for the player to play in

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """

        root = Node(board, self.player_id)
        Tree.expand_node(root, self.depth)

        def alphabeta_node(node, depth, alpha, beta, maximizing_player):
            winner = self.heuristic.winning(node.board.get_board_state(), self.game_n)
            if depth == 0 or winner != 0:
                return self.heuristic.evaluate_board(self.player_id, node.board)

            if maximizing_player:
                max_eval = -np.inf
                best_move = None
                for child in node.children:
                    eval = alphabeta_node(child, depth - 1, alpha, beta, False)
                    if eval > max_eval:
                        max_eval = eval
                        best_move = child
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
                if depth == self.depth and best_move is not None:
                    for col in range(board.width):
                        if board.is_valid(col):
                            new_board = board.get_new_board(col, self.player_id)
                            if np.array_equal(new_board.get_board_state(), best_move.board.get_board_state()):
                                return col
                    return 0  # fallback
                return max_eval
            else:
                min_eval = np.inf
                for child in node.children:
                    eval = alphabeta_node(child, depth - 1, alpha, beta, True)
                    if eval < min_eval:
                        min_eval = eval
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
                return min_eval

        move = alphabeta_node(root, self.depth, -np.inf, np.inf, True)
        return move


class HumanPlayer(PlayerController):
    """Class for the human player
    Inherits from Playercontroller
    """
    def __init__(self, player_id: int, game_n: int, heuristic: Heuristic) -> None:
        """
        Args:
            player_id (int): id of a player, can take values 1 or 2 (0 = empty)
            game_n (int): n in a row required to win
            heuristic (Heuristic): heuristic used by the player
        """
        super().__init__(player_id, game_n, heuristic)

    
    def make_move(self, board: Board) -> int:
        """Gets the column for the player to play in

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """
        print(board)

        if self.heuristic is not None:
            print(f'Heuristic {self.heuristic} calculated the best move is:', end=' ')
            print(self.heuristic.get_best_action(self.player_id, board) + 1, end='\n\n')

        col: int = self.ask_input(board)

        print(f'Selected column: {col}')
        return col - 1
    

    def ask_input(self, board: Board) -> int:
        """Gets the input from the user

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """
        try:
            col: int = int(input(f'Player {self}\nWhich column would you like to play in?\n'))
            assert 0 < col <= board.width
            assert board.is_valid(col - 1)
            return col
        except ValueError: # If the input can't be converted to an integer
            print('Please enter a number that corresponds to a column.', end='\n\n')
            return self.ask_input(board)
        except AssertionError: # If the input matches a full or non-existing column
            print('Please enter a valid column.\nThis column is either full or doesn\'t exist!', end='\n\n')
            return self.ask_input(board)
