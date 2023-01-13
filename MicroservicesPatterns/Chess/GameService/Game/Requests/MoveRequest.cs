using GameService.Game.Positions;

namespace GameService.Game.Requests;

public record MoveRequest(
    string GameId,
    string PlayerId,
    string From,
    string To
)
{
    public Move ToMove() => new(
        GameId,
        PlayerId,
        Position.Parse(From),
        Position.Parse(To)
    );
}
