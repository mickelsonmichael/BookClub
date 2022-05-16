using FunctionalProgramming.Notes.Chapter09.Models;
using FunctionalProgramming.Notes.Chapter09.Services;
using LanguageExt;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;

namespace FunctionalProgramming.Notes.Chapter09.Controllers;

[ApiController]
[Route("transfers")]
public class MakeTransferController : ControllerBase
{
    public MakeTransferController(ILogger<MakeTransferController> logger) => _logger = logger;

    [HttpPost]
    public IActionResult MakeTransfer(
       [FromBody] TransferRequest request,
       [FromServices] IRepository<TransferRequest> repo,
       [FromServices] IValidator<TransferRequest> validator
    )
    {
        _logger.LogInformation("Transfer requested");

        return validator.Validate(request)
            .Map(repo.Save)
            .Match(
                Succ: result => result.Match<TransferRequest, IActionResult>
                (
                    Succ: transfer => Ok(transfer.GetConfirmation()),
                    Fail: StatusCode(500, new { message = "An unexpected error occurred" })
                ),
                Fail: errors => BadRequest(new { errors }));
    }

    private readonly ILogger<MakeTransferController> _logger;
}
