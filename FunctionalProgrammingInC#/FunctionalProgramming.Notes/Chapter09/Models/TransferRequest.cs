namespace FunctionalProgramming.Notes.Chapter09.Models;

public record TransferRequest(
    string SourceAccount,
    string DestinationAccount,
    decimal Amount,
    DateTime Date
)
{
    public TransferConfirmation GetConfirmation()
        => new(SourceAccount, DestinationAccount, Amount, Date);
}
