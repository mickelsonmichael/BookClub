using FunctionalProgramming.Web.Events;

namespace FunctionalProgramming.Web.Domain;

public readonly record struct Book(
    Guid BookId,
    string Title,
    string Author
)
{
    public static Book Create(CreatedBook createEvent)
        => new(createEvent.EntityId, createEvent.Title, createEvent.Author);

    public Book Apply(Event evt)
        => evt switch
        {
            _ => throw new InvalidOperationException("Invalid event passed to the book")
        };
}
