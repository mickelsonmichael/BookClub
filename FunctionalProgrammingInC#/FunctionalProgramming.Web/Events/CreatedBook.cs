namespace FunctionalProgramming.Web.Events;

public record CreatedBook(
    Guid EntityId,
    DateTime Timestamp,
    string Title,
    string Author
) : Event(EntityId, Timestamp);
