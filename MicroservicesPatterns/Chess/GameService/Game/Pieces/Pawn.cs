using GameService.Game.Players;
using GameService.Game.Positions;

namespace GameService.Game.Pieces;

public class Pawn : ChessPiece
{
    public Pawn(Player side, Position position)
        : base(new ChessPieceId(side, PieceType.Pawn, position), position)
    {
    }

    public Pawn(ChessPieceId id, Position position)
        : base(id, position)
    {
    }

    public override IEnumerable<Position> FindValidMoves(IEnumerable<IChessPiece> otherPieces)
    {
        return GetForwardMoves(otherPieces).Concat(
            GetDiagonalMoves(otherPieces)
        );
    }

    public override MoveResult Move(Position targetPosition, IEnumerable<IChessPiece> otherPieces)
    {
        // Pawn requires special treatment because of the "En Passant" rule
        // https://www.chess.com/terms/en-passant

        IEnumerable<Position> forwardMoves = GetForwardMoves(otherPieces);

        if (forwardMoves.Contains(targetPosition))
        {
            return MoveResult.Valid();
        }

        IEnumerable<Position> diagonalMoves = GetDiagonalMoves(otherPieces);

        if (!diagonalMoves.Contains(targetPosition))
        {
            return MoveResult.Invalid("The selected position is not a valid move");
        }

        IChessPiece? directCapture = otherPieces.SingleOrDefault(p => p.Position == targetPosition);

        if (directCapture != null)
        {
            return MoveResult.Valid(directCapture);
        }

        Position enPassantTargetPosition = targetPosition.Letter > Position.Letter
            ? Position.Right()
            : Position.Left();

        IChessPiece enPassantPiece = otherPieces.Single(p => p.Position == enPassantTargetPosition);

        return MoveResult.Valid(enPassantPiece);
    }

    private IEnumerable<Position> GetForwardMoves(IEnumerable<IChessPiece> otherPieces)
    {
        Position nextForward(Position position) => Owner == Player.White
            ? position.Up()
            : position.Down();

        Position oneForward = nextForward(Position);

        if (oneForward.IsValid && !otherPieces.Any(p => p.Position == oneForward))
        {
            yield return oneForward;
        }

        if (Position != InitialPosition)
        {
            yield break;
        }

        Position twoForward = nextForward(oneForward);

        if (twoForward.IsValid && !otherPieces.Any(p => p.Position == twoForward))
        {
            yield return twoForward;
        }
    }

    private IEnumerable<Position> GetDiagonalMoves(IEnumerable<IChessPiece> otherPieces)
    {
        Position forwardLeft = Owner == Player.White
            ? Position.Up().Left()
            : Position.Down().Left();

        Position left = Position.Left();

        if (forwardLeft.IsValid && otherPieces.Any(p => p.Position == forwardLeft || p.Position == left))
        {
            yield return forwardLeft;
        }

        Position forwardRight = Owner == Player.White
            ? Position.Up().Right()
            : Position.Down().Right();

        Position right = Position.Right();

        if (forwardLeft.IsValid && otherPieces.Any(p => p.Position == forwardRight || p.Position == forwardLeft))
        {
            yield return forwardRight;
        }
    }
}
