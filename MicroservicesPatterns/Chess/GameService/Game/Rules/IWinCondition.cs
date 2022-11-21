using GameService.Game.Pieces;
using GameService.Game.Players;

namespace GameService.Game.Rules;

public interface IWinCondition
{
    Player? GetWinner(IEnumerable<IChessPiece> pieces);
}
