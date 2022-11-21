using GameService.Game.Positions;
using GameService.Game.Requests;
using GameService.Game.Responses;
using Microsoft.AspNetCore.Mvc;

namespace GameService.Game;

public static class GameHandlers
{
    public static IResult CreateGame(
        string playerId,
        [FromServices] GameController gameController
    )
    {
        ChessGame game = gameController.CreateGame(playerId);

        return Results.Ok(game);
    }

    public static IResult GetGame(
        string gameId,
        [FromServices] GameController gameController
    )
    {
        ChessGame? game = gameController.GetGame(gameId);

        if (game == null)
        {
            return Results.NotFound();
        }

        GetGameResponse resp = new(
            game.Id,
            game.WhitePlayerId,
            game.BlackPlayerId != null ? (string?)game.BlackPlayerId : null,
            game.GetWinner(),
            game.Turn,
            game.GetPieces()
        );

        return game != null
            ? Results.Ok(resp)
            : Results.NotFound();
    }

    public static IResult JoinGame(
        [FromBody] JoinRequest req,
        [FromServices] GameController gameController
    )
    {
        ChessGame? game = gameController.JoinGame(req.GameId, req.PlayerId);

        return game != null
            ? Results.NoContent()
            : Results.NotFound();
    }

    public static IResult Move(
        [FromBody] MoveRequest req,
        [FromServices] GameController gameController
    )
    {
        Move move = req.ToMove();

        if (!move.From.IsValid || !move.To.IsValid)
        {
            return Results.BadRequest();
        }

        MoveResult? moveResult = gameController.Move(move);

        if (moveResult == null)
        {
            return Results.NotFound();
        }

        if (!moveResult.IsSuccessful)
        {
            return Results.BadRequest(moveResult.Errors);
        }

        MoveResponse resp = new(
            move.To.Stringify(),
            moveResult.CapturedPieces.Select(piece => piece.Position.Stringify()).ToArray()
        );

        return Results.Ok(resp);
    }
}
