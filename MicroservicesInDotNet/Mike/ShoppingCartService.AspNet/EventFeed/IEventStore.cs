using LanguageExt;

namespace ShoppingCartService.EventFeed;

public interface IEventStore
{
    Task<Try<IEnumerable<Event>>> GetEvents(
        long? fromSequenceNumber,
        long? toSequenceNumber
    );

    Unit Raise(string eventName, object content);
}
