using GameService.Game.Pieces;
using GameService.Game.Players;
using GameService.Game.Positions;

namespace GameService.Game.Rules;

public class CheckmateWinCondition : IWinCondition
{
    public Player? GetWinner(IEnumerable<IChessPiece> pieces)
    {
        IChessPiece[] kings = pieces.Where(x => x.Id.PieceType == PieceType.King).ToArray();

        ValidateScene(kings);

        Func<IChessPiece, bool> IsCheckmated = IsCheckmatedWithPieces(pieces);

        IChessPiece[] checkmatedKings = kings
            .Where(IsCheckmated)
            .ToArray();

        if (checkmatedKings.Length > 1)
        {
            throw new InvalidOperationException(
                $"Game state is invalid. {checkmatedKings.Length} kings were in checkmate instead of one or none."
            );
        }

        return checkmatedKings.Length == 1
            ? checkmatedKings[0].Id.Player
            : null;
    }

    private static void ValidateScene(IChessPiece[] kings)
    {
        if (kings.Length != 2)
        {
            throw new InvalidOperationException(
                $"Checkmate cannot be determined. Games must have two kings in play but received {kings.Length}."
            );
        }

        if (kings[0].IsAlly(kings[1]))
        {
            throw new InvalidOperationException(
                $"Checkmate cannot be determined. Games must have two kings of opposing sides, but received two kings on the {kings[0].Id.Player} side"
            );
        }
    }

    private static Func<IChessPiece, bool> IsCheckmatedWithPieces(IEnumerable<IChessPiece> pieces)
    {
        return (IChessPiece king) =>
        {
            IEnumerable<IChessPiece> enemies = pieces.Where(piece => !piece.IsAlly(king));

            IEnumerable<Position> positionsCoveredByEnemies = enemies.SelectMany(piece => piece.FindValidMoves(pieces)).Distinct();

            Position[] kingMoves = king.FindValidMoves(pieces).ToArray();

            if (kingMoves.Length == 0)
            {
                return true;
            }

            return !kingMoves.Except(positionsCoveredByEnemies).Any();
        };
    }
}
