using GameService.Game.Pieces;
using GameService.Game.Players;

namespace GameService.Game.Responses;

public record GetGameResponse(
    string GameId,
    string WhitePlayerId,
    string? BlackPlayerId,
    Player? Winner,
    Player Turn,
    ICollection<Piece> Pieces
);
