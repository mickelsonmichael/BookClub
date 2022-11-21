using GameService.Game.Pieces;

namespace GameService.Game;

public record Piece(
    string Position,
    string Type,
    string Owner
)
{
    public static Piece From(IChessPiece piece) => new(
        piece.Position.Stringify(),
        piece.Id.PieceType.ToString(),
        piece.Id.Player.ToString()
    );
}
