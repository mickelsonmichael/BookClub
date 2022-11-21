using GameService.Game.Players;
using GameService.Game.Positions;

namespace GameService.Game.Pieces;

public abstract class ChessPiece : IChessPiece
{
    protected ChessPiece(ChessPieceId id, Position position)
    {
        Id = id;
        Position = position;
    }

    public ChessPieceId Id { get; }

    public Position Position { get; private set; }

    protected Position InitialPosition => Id.StartingPosition;

    protected Player Owner => Id.Player;

    public abstract IEnumerable<Position> FindValidMoves(IEnumerable<IChessPiece> otherPieces);

    public virtual MoveResult Move(Position targetPosition, IEnumerable<IChessPiece> otherPieces)
    {
        IEnumerable<Position> validMoves = FindValidMoves(otherPieces);

        if (!validMoves.Contains(targetPosition))
        {
            return MoveResult.Invalid("Cannot perform the move. Position is not a valid move for the piece.");
        }

        Position = targetPosition;

        IEnumerable<IChessPiece> capturedPieces = otherPieces.Where(piece => piece.Position == targetPosition);

        return MoveResult.Valid(capturedPieces);
    }
}
