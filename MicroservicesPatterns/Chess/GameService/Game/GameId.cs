namespace GameService.Game;

public record GameId(string Value)
{
    public static GameId Create() => new(Guid.NewGuid().ToString());

    public static implicit operator GameId(string value) => new(value);
    public static implicit operator string(GameId gameId) => gameId.Value;
}
