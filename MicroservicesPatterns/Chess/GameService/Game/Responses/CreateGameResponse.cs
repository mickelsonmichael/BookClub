namespace GameService.Game;

public record CreateGameResponse(
    string GameId,
    IEnumerable<Piece> Pieces
)
{
    public static CreateGameResponse From(ChessGame game) =>
        new(
            game.Id.Value,
            game.GetPieces().Select(Piece.From)
        );
}
