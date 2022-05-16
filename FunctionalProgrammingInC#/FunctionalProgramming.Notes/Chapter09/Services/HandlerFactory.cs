using FunctionalProgramming.Notes.Chapter09.Examples;
using FunctionalProgramming.Notes.Chapter09.Models;
using LanguageExt;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Configuration;
using SqlTemplate = System.String;

namespace FunctionalProgramming.Notes.Chapter09.Services;

public static class HandlerFactory
{
    public static Func<TransferRequest, IResult> CreateTransferReqestHandler(IConfiguration configuration)
    {
        ConnectionString connectionString = configuration.GetConnectionString("Database");

        var validator = TransferRequestValidators.DateNotPast(() => DateTime.Now);

        Try<int> save(TransferRequest _)
        {
            SqlTemplate sql = "INSERT ...";

            return connectionString.Try(sql);
        }

        return RequestHandler.HandleTransferRequest(validator, save);
    }
}
