using FunctionalProgramming.Web.Events;
using MongoDB.Driver;

namespace FunctionalProgramming.Web.Services;

public class MongoDbBookRepository : IBookRepository
{
    public MongoDbBookRepository(ILogger<MongoDbBookRepository> logger, IConfiguration configuration)
    {
        _logger = logger;

        string connectionString = configuration.GetConnectionString("MongoDb");

        MongoClientSettings settings = MongoClientSettings.FromConnectionString(connectionString);

        settings.ServerApi = new ServerApi(ServerApiVersion.V1);

        MongoClient client = new(settings);

        IMongoDatabase database = client.GetDatabase("BookClub");

        _bookEvents = database.GetCollection<Event>("BookEvents");
    }

    public async Task AddAsync(CreatedBook createBookEvent)
    {
        _logger.LogInformation("Adding book creation event.\n{CreateBookEvent}", createBookEvent);

        await _bookEvents.InsertOneAsync(createBookEvent);
    }

    private readonly ILogger<MongoDbBookRepository> _logger;
    private readonly IMongoCollection<Event> _bookEvents;
}
