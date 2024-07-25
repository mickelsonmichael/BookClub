from copy import deepcopy


class ChessSpace:
    x: int
    y: int

    x_to_letter_map = {
        1: "a",
        2: "b",
        3: "c",
        4: "d",
        5: "e",
        6: "f",
        7: "g",
        8: "h"
    }

    def get_name(self):
        return f'{self.x}{self.y}'

    def up(self):
        return ChessSpace(self.x, self.y + 1)

    def down(self):
        return ChessSpace(self.x, self.y - 1)

    def left(self):
        return ChessSpace(self.x - 1, self.y)

    def right(self):
        return ChessSpace(self.x + 1, self.y)


class ChessPiece:
    name: str
    get_moves: callable[ChessSpace, list[ChessSpace]]

    def __init__(self, name: str, get_moves: callable[ChessSpace, list[ChessSpace]]):
        self.name = name
        self.get_moves = get_moves

    def clone(self):
        return deepcopy(self)


class ChessGame:
    pieces: list[ChessPiece]

    def __init__(self, pieces: list[ChessPiece]):
        self.pieces = pieces


if __name__ == "__main__":
    pawn_prototype = ChessPiece(
        "pawn",
        get_moves=lambda space: space.x + 1
    )

    king_prototype = ChessPiece(
        "king",
        lambda space: [space.up(), space.down(), space.left(), space.right()]
    )

    # gotta make more pawns from the prototype
    white_pawns = [pawn_prototype.clone() for _ in range(8)]

    game = ChessGame([king_prototype] + white_pawns)

    print(game)
