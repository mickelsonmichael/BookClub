using FunctionalProgramming.Web.Chapter09.Models;
using LanguageExt;

namespace FunctionalProgramming.Web.Chapter09.Services;

public static class TransferRequestValidators
{
    public static Func<TransferRequest, Validation<ICollection<string>, TransferRequest>> DateNotPast(Func<DateTime> getCurrentTime)
        => transfer
        => transfer.Date.Date < getCurrentTime().Date
            ? new[] { "Transfer date is in the past!" }
            : transfer;
}
