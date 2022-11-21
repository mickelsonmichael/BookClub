namespace GameService.Game.Players;

public record PlayerId(string Value)
{
    public static implicit operator PlayerId(string value) => new(value);

    public static implicit operator string(PlayerId id) => id.Value;
}
