namespace GameService.Game.Requests;

public record JoinRequest(
    string GameId,
    string PlayerId
);
