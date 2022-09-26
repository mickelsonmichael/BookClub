using System.Collections.Concurrent;
using LanguageExt;

namespace ShoppingCartService.EventFeed;

using static LanguageExt.Prelude;

public class InMemoryEventStore : IEventStore
{
    public InMemoryEventStore(ILogger<InMemoryEventStore> logger)
    {
        logger.LogWarning("Using in-memory event feed.");

        _logger = logger;
    }

    public long FirstSequenceNumber => 1;
    private long LastSequenceNumber => _events.Keys.Max();

    public Task<Try<IEnumerable<Event>>> GetEvents(long? fromSequenceNumber, long? toSequenceNumber) =>
        Try(
            _events.SkipWhile(x => x.Key < (fromSequenceNumber ?? FirstSequenceNumber))
                .TakeWhile(x => x.Key <= (toSequenceNumber ?? LastSequenceNumber))
                .OrderBy(x => x.Key)
                .Select(x => x.Value)
        ).AsTask();

    public Unit Raise(string eventName, object content)
    {
        long seqNum = Interlocked.Increment(ref _sequenceNumber);

        _logger.LogDebug("Creating event {SequenceNumber} as {EventName}.\n{EventContent}", seqNum, eventName, content);

        _events[seqNum] = new Event(seqNum, DateTimeOffset.UtcNow, eventName, content);

        return Unit.Default;
    }

    private long _sequenceNumber = 0;
    private readonly ILogger<InMemoryEventStore> _logger;
    private readonly ConcurrentDictionary<long, Event> _events = new();
}
