namespace ConsoleService.Games;

public record ChessPiece(
    string Position,
    string Type,
    string Owner
)
{
    public char Letter => Type switch
    {
        "King" => 'K',
        "Queen" => 'Q',
        "Bishop" => 'B',
        "Knight" => 'N',
        "Rook" => 'R',
        "Pawn" => 'P',
        _ => 'X'
    };
}
