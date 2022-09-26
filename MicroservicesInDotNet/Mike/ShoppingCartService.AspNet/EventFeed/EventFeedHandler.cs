using System.Net;
using Microsoft.AspNetCore.Mvc;

namespace ShoppingCartService.EventFeed;

public static class EventFeedHandler
{
    public static Task<IResult> HandleGetFeed(
        [FromServices] IEventStore eventStore,
        [FromQuery] long? start,
        [FromQuery] long? end
    ) => eventStore.GetEvents(start, end)
        .Match(
            Succ: events => Results.Ok(events),
            Fail: Results.StatusCode((int)HttpStatusCode.InternalServerError)
        );
}
