namespace GameService.Game;

public interface IGameRepository
{
    ChessGame? GetGame(GameId gameId);
    ChessGame AddGame(ChessGame game);
    void Update(ChessGame game);
}
