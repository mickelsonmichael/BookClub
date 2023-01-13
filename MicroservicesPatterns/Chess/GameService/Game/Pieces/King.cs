using GameService.Game.Players;
using GameService.Game.Positions;

namespace GameService.Game.Pieces;

public class King : ChessPiece
{
    public King(Player player, Position position)
        : base(new ChessPieceId(player, PieceType.King, position), position)
    {
    }

    public override IEnumerable<Position> FindValidMoves(IEnumerable<IChessPiece> otherPieces)
    {
        bool IsValid(Position position)
        {
            if (!position.IsValid) return false;

            IChessPiece? pieceAtPosition = otherPieces.SingleOrDefault(p => p.Position == position);

            return pieceAtPosition?.IsAlly(this) != false;
        }

        return new[]
        {
            Position.Up(),
            Position.Up().Right(),
            Position.Right(),
            Position.Down().Right(),
            Position.Down(),
            Position.Down().Left(),
            Position.Left(),
            Position.Up().Left()
        }.Where(IsValid)
            .ToArray();
    }
}
