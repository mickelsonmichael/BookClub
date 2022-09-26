namespace ShoppingCartService.EventFeed;

public static class EventFeedExtensions
{
    public static WebApplicationBuilder AddEventFeed(this WebApplicationBuilder builder)
    {
        builder.Services.AddSingleton<IEventStore, InMemoryEventStore>();

        return builder;
    }
}
