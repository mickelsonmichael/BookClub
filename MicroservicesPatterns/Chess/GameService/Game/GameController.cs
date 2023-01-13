using GameService.Game.Players;
using GameService.Game.Positions;

namespace GameService.Game;

public class GameController
{
    public GameController(IGameRepository gameRepository)
    {
        _gameRepository = gameRepository;
    }

    public ChessGame? GetGame(GameId gameId) =>
        _gameRepository.GetGame(gameId);

    public ChessGame CreateGame(PlayerId playerId)
    {
        ChessGame game = GameFactory.CreateStandard(playerId);

        _gameRepository.AddGame(game);

        return game;
    }

    public ChessGame? JoinGame(GameId gameId, PlayerId playerId)
    {
        ChessGame? game = GetGame(gameId);

        if (game != null)
        {
            game.JoinGame(playerId);

            _gameRepository.Update(game);
        }

        return game;
    }

    public MoveResult? Move(Move move)
    {
        ChessGame? game = GetGame(move.GameId);

        if (game == null)
        {
            return null;
        }

        MoveResult result = game.Move(move);

        if (result.IsSuccessful)
        {
            _gameRepository.Update(game);
        }

        return result;
    }

    private readonly IGameRepository _gameRepository;
}
