using GameService.Game.Players;
using GameService.Game.Positions;

namespace GameService.Game.Pieces;

public record ChessPieceId(
    Player Player,
    PieceType PieceType,
    Position StartingPosition
);
