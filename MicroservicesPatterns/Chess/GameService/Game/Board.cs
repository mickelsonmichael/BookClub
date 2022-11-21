using GameService.Game.Pieces;
using GameService.Game.Positions;

namespace GameService.Game;

public class Board
{
    public Board(IEnumerable<IChessPiece> pieces)
    {
        _pieces = new List<IChessPiece>(pieces);
    }

    public IReadOnlyCollection<IChessPiece> Pieces => _pieces;

    public IEnumerable<Position> FindMovesFor(Position target)
    {
        IChessPiece? piece = GetPiece(target);

        return piece == null
            ? Enumerable.Empty<Position>()
            : piece.FindValidMoves(Pieces);
    }

    public MoveResult Move(Position from, Position to)
    {
        IChessPiece? piece = GetPiece(from);

        if (piece == null)
        {
            return MoveResult.Invalid($"There is no piece at position [{from}].");
        }

        return piece.Move(to, _pieces);
    }

    public IChessPiece? GetPiece(Position at) => _pieces.SingleOrDefault(p => p.Position == at);

    private readonly List<IChessPiece> _pieces;
}
