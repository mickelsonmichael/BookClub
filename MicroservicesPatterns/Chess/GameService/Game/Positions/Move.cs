using GameService.Game.Players;

namespace GameService.Game.Positions;

public record Move(
    GameId GameId,
    PlayerId PlayerId,
    Position From,
    Position To
)
{
    public bool IsValid() => From != To;
}
