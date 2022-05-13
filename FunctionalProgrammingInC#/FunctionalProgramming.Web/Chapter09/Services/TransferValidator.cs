using FunctionalProgramming.Web.Chapter09.Models;
using LanguageExt;

namespace FunctionalProgramming.Web.Chapter09.Services;

public class TransferValidator : IValidator<TransferRequest>
{
    public TransferValidator(ILogger<TransferValidator> logger)
    {
        _logger = logger;
    }
    public Validation<ICollection<string>, TransferRequest> Validate(TransferRequest input)
    {
        _logger.LogInformation("Validating transfer request");

        return input.Date.Date < DateTime.Today.Date
                   ? new[] { "Transfer date is in the past!" }
                   : input;
    }

    private readonly ILogger<TransferValidator> _logger;
}