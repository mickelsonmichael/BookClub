namespace ConsoleService.Games;

public record ChessGame(
    string GameId,
    ICollection<ChessPiece> Pieces
);
