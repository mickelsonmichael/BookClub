namespace FunctionalProgramming.Web.Chapter09.Models;

public record TransferConfirmation(
    string SourceAccount,
    string DestinationAccount,
    decimal Amount,
    DateTime Date
);
