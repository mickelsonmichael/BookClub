namespace FunctionalProgramming.Notes.Chapter09.Models;

public record TransferConfirmation(
    string SourceAccount,
    string DestinationAccount,
    decimal Amount,
    DateTime Date
);
