using GameService.Game.Pieces;
using GameService.Game.Players;
using GameService.Game.Rules;

namespace GameService.Game;

public static class GameFactory
{
    public static ChessGame CreateStandard(PlayerId playerId)
    {
        GameId gameId = GameId.Create();

        IRuleSet standardRules = new StandardRuleSet();

        IEnumerable<IChessPiece> initialPieces = standardRules.GetInitialPieces();

        Board board = new(initialPieces);

        return new ChessGame(
            gameId,
            hostId: playerId,
            guestId: null,
            turn: Player.White,
            board,
            standardRules.WinConditions
        );
    }
}
