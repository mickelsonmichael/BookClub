namespace GameService.Game;

public class InMemoryGameRepository : IGameRepository
{
    public ChessGame? GetGame(GameId gameId) =>
        _games.TryGetValue(gameId, out ChessGame? game) ? game : null;

    public ChessGame AddGame(ChessGame game)
    {
        _games.Add(game.Id, game);

        return game;
    }

    public void Update(ChessGame game)
    {
        _games[game.Id] = game;
    }

    private readonly Dictionary<GameId, ChessGame> _games = new();
}
