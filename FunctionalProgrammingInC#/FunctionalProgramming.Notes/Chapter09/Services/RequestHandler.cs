using FunctionalProgramming.Notes.Chapter09.Models;
using LanguageExt;
using Microsoft.AspNetCore.Http;
using static Microsoft.AspNetCore.Http.Results;

namespace FunctionalProgramming.Notes.Chapter09.Services;

public static class RequestHandler
{
    public static Func<TransferRequest, IResult> HandleTransferRequest(
        Func<TransferRequest, Validation<ICollection<string>, TransferRequest>> validate,
        Func<TransferRequest, Try<int>> save
    )
        => transfer
        => validate(transfer)
            .Map(save)
            .Match(
                Succ: result => result.Match(
                    Succ: _ => Ok(),
                    Fail: _ => StatusCode(500)
                ),
                Fail: err => BadRequest(new { message = err })
            );
}