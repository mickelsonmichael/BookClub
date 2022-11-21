using System.Text;

namespace ConsoleService.Games;

public class GameController
{
    public GameController(ApiGameRepository gameRepository)
    {
        _gameRepository = gameRepository;
    }

    public async Task Play()
    {
        ChessGame? game = await CreateGame();

        if (game == null)
        {
            Console.WriteLine("Whoops couldn't make the game, try again");

            return;
        }

        string turn = _white;

        while (true)
        {
            game = await GetGame(game.GameId);

            PrintBoard(game.Pieces);

            Console.WriteLine("Select a piece to move: ");

            string from = ReadPosition();

            Console.WriteLine("Select where to move the piece: ");

            string to = ReadPosition();

            string[]? captured = await MakeMove(game.GameId, turn, from, to);

            if (captured == null)
            {
                Console.WriteLine("Hmmm that doesn't seem to be a valid move");

                continue;
            }

            if (captured.Length > 0)
            {
                Console.WriteLine($"Piece(s) at {string.Join(", ", captured)} have been captured");
            }

            turn = turn == _white ? _black : _white;
        }
    }

    public async Task<ChessGame?> CreateGame()
    {
        ChessGame? game = await _gameRepository.CreateGame(_white);

        if (game == null)
        {
            return null;
        }

        await _gameRepository.JoinGame(game.GameId, _black);

        return game;
    }

    private async Task<ChessGame> GetGame(string gameId)
    {
        ChessGame? game = await _gameRepository.GetGame(gameId);

        if (game == null)
        {
            Console.WriteLine("Uh oh, looks like the game went missing!");

            Environment.Exit(1);
        }

        return game;
    }

    public async Task<string[]?> MakeMove(string gameId, string playerId, string from, string to)
    {
        return await _gameRepository.MovePiece(gameId, playerId, from, to);
    }

    public static string ReadPosition()
    {
        while (true)
        {
            string? position = Console.ReadLine();

            if (position == null)
            {
                Console.WriteLine("Invalid position. You have to enter something");

                continue;
            }

            return position;
        }
    }

    private static void PrintBoard(IEnumerable<ChessPiece> pieces)
    {
        Console.WriteLine("  || A | B | C | D | E | F | G | H |");
        Console.WriteLine("- || - | - | - | - | - | - | - | - |");

        for (int i = 8; i > 0; i--)
        {
            StringBuilder sb = new($"{i} ||");

            for (char l = 'a'; l <= 'h'; l = (char)(l + 1))
            {
                string position = $"{l}{i}";

                ChessPiece? piece = pieces.SingleOrDefault(p => string.Equals(p.Position, position, StringComparison.CurrentCultureIgnoreCase));

                sb.Append(' ').Append(piece?.Letter ?? ' ').Append(" |");
            }

            Console.WriteLine(sb.ToString());
        }
    }

    private const string _white = "white-player";
    private const string _black = "black-player";
    private readonly ApiGameRepository _gameRepository;
}
