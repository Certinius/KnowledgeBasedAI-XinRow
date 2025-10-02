from __future__ import annotations
from typing import List, Optional

from board import Board

from numpy import ndarray


class Node:
    def __init__(self, board: Board, player_id: int, children: Optional[ List[Node]] = None) -> None:
        """player_id here is the id of the player whose turn it (technically) is
        """
        self._board:        Board       = board
        self._player_id:    int         = player_id

        self._children:     List[Node]  = children or []

    @property
    def board(self) -> Board:
        return self._board

    @property
    def player_id(self) -> int:
        return self._player_id

    @property
    def children(self) -> List[Node]:
        return self._children

    @children.setter
    def children(self, children: List[Node]) -> None:
        self._children = children

    def __str__(self) -> str:
        output = ""

        output += f"player {self.player_id}'s turn from state:\n"
        output += self.board.__str__()
        output += f"{len(self.children)} possible moves:\n"

        for child in self.children:
            output += child.__str__()

        return output


class Tree:
    minimax_depth: int

    def __init__(self, root: Node) -> None:
        self.root = root

    def create_tree(self, depth: int) -> None:
        assert self.root is not None, "no Tree root"

        self.expand_node(self.root, depth)

        self.minimax_depth = depth

    @staticmethod
    def expand_node(node: Node, depth: int) -> None:
        if depth == 0:
            return

        children: List[Node] = []

        boards = Tree.create_boards(node.board, node.player_id)
        for b in boards:
            child = Node(b, 3 - node.player_id)
            Tree.expand_node(child, depth - 1)

            children.append(child)

        node.children = children

    @staticmethod
    def create_boards(starting_board: Board, player_id: int) -> List[Board]:
        """Returns all resulting boards after
        every possible turn has been made from
        a board (state) starting_board by
        player with id = player_id
        """

        res: List[Board] = []

        for i in range(starting_board.width):
            if starting_board.is_valid(i):
                temp = Board(starting_board)
                if temp.play(i, player_id):
                    res.append(temp)

        return res

    def __str__(self) -> str:
        """
        Returns:
            str: a human-readable representation of the tree
        """

        assert self.root is not None, "no Tree root"

        return self.root.__str__()
