using GameService.Game.Pieces;

namespace GameService.Game.Positions;

public record MoveResult(
    ErrorList Errors,
    ICollection<IChessPiece> CapturedPieces
)
{
    public bool IsSuccessful => Errors.IsEmpty();

    public static MoveResult Invalid(string errorMessage) => new(new ErrorList().Add(errorMessage), Array.Empty<IChessPiece>());
    public static MoveResult Valid() => new(new ErrorList(), Array.Empty<IChessPiece>());
    public static MoveResult Valid(IEnumerable<IChessPiece> capturedPieces) => new(new ErrorList(), new List<IChessPiece>(capturedPieces));
    public static MoveResult Valid(IChessPiece capturedPiece) => new(new ErrorList(), new List<IChessPiece>() { capturedPiece });
}
