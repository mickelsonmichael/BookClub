using GameService.Game.Positions;

namespace GameService.Game.Pieces;

public interface IChessPiece
{
    ChessPieceId Id { get; }

    Position Position { get; }

    IEnumerable<Position> FindValidMoves(IEnumerable<IChessPiece> otherPieces);

    MoveResult Move(Position targetPosition, IEnumerable<IChessPiece> otherPieces);

    bool IsAlly(IChessPiece otherPiece) => otherPiece.Id.Player == Id.Player;
}
