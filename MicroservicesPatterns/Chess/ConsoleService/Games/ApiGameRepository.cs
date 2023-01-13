using System.Net.Http.Json;

namespace ConsoleService.Games;

public class ApiGameRepository
{
    public ApiGameRepository()
    {
        HttpClientHandler handler = new()
        {
            ServerCertificateCustomValidationCallback = HttpClientHandler.DangerousAcceptAnyServerCertificateValidator
        };

        _client = new HttpClient(handler)
        {
            BaseAddress = new Uri("https://localhost:7286")
        };
    }

    public async Task<ChessGame?> CreateGame(string playerId)
    {
        HttpResponseMessage resp = await _client.PostAsJsonAsync("/game", playerId);

        if (!resp.IsSuccessStatusCode)
        {
            Console.WriteLine($"Unable to create the game! {resp.StatusCode}");

            return null;
        }

        return await resp.Content.ReadFromJsonAsync<ChessGame>();
    }

    public async Task JoinGame(string gameId, string playerId)
    {
        var body = new
        {
            gameId,
            playerId
        };

        HttpResponseMessage resp = await _client.PostAsJsonAsync("/game/join", body);

        if (!resp.IsSuccessStatusCode)
        {
            Console.WriteLine($"Unable to join the game! {resp.StatusCode}");
        }
    }

    public Task<ChessGame?> GetGame(string gameId)
    {
        return _client.GetFromJsonAsync<ChessGame>($"/game?gameId={gameId}");
    }

    public async Task<string[]?> MovePiece(string gameId, string playerId, string from, string to)
    {
        var body = new { gameId, playerId, from, to };

        HttpResponseMessage resp = await _client.PostAsJsonAsync("/game/move", body);

        return resp.IsSuccessStatusCode
            ? (await resp.Content.ReadFromJsonAsync<(string, string[])>()).Item2
            : Array.Empty<string>();
    }

    private readonly HttpClient _client;
}
