namespace GameService.Game.Responses;

public record MoveResponse(
    string Position,
    string[] CapturedPieces
);
