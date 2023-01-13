using GameService.Game.Pieces;

namespace GameService.Game.Rules;

public interface IRuleSet
{
    ICollection<IWinCondition> WinConditions { get; }

    IEnumerable<IChessPiece> GetInitialPieces();
}
