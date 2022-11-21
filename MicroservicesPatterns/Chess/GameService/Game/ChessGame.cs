using GameService.Game.Pieces;
using GameService.Game.Players;
using GameService.Game.Rules;
using GameService.Game.Positions;

namespace GameService.Game;

public class ChessGame
{
    public GameId Id { get; }

    public PlayerId WhitePlayerId { get; }

    public PlayerId? BlackPlayerId { get; private set; }

    public Player Turn { get; private set; }

    public ChessGame(GameId gameId, PlayerId hostId, PlayerId? guestId, Player turn, Board board, IEnumerable<IWinCondition> winConditions)
    {
        Id = gameId;

        WhitePlayerId = hostId;
        BlackPlayerId = guestId;
        Turn = turn;
        _board = board;
        _winConditions = new List<IWinCondition>(winConditions);
    }

    public void JoinGame(PlayerId playerId)
    {
        if (BlackPlayerId != null)
        {
            throw new InvalidOperationException("Cannot join a game with two players");
        }

        BlackPlayerId = playerId;
    }

    public MoveResult Move(Move move)
    {
        if (BlackPlayerId == null)
        {
            return MoveResult.Invalid("Cannot perform the move because a player has not joined.");
        }

        Player? winner = GetWinner();

        if (winner != null)
        {
            return MoveResult.Invalid($"Cannot perform the move because the {winner} player has already won.");
        }

        bool isMovingPlayersTurn = Turn == Player.White
            ? WhitePlayerId == move.PlayerId
            : BlackPlayerId == move.PlayerId;

        if (isMovingPlayersTurn)
        {
            return MoveResult.Invalid("Cannot perform the move. It is not the player's turn.");
        }

        MoveResult result = _board.Move(move.From, move.To);

        if (result.IsSuccessful)
        {
            Turn = Turn == Player.White ? Player.Black : Player.White;
        }

        return result;
    }

    public Player? GetWinner()
    {
        Player[] winConditionsMet = _winConditions
            .Select(c => c.GetWinner(_board.Pieces))
            .Where(winner => winner != null)
            .Cast<Player>()
            .Distinct()
            .ToArray();

        if (winConditionsMet.Length == 0)
        {
            return null;
        }

        return winConditionsMet.Length == 1
            ? winConditionsMet[0]
            : Player.White; // TODO: handle a tie
    }

    public ICollection<IChessPiece> GetPieces() => new List<IChessPiece>(_board.Pieces);

    private readonly Board _board;
    private readonly List<IWinCondition> _winConditions;
}
