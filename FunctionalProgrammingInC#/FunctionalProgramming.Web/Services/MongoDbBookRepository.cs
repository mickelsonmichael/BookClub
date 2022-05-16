using FunctionalProgramming.Web.Domain;
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

        _db = client.GetDatabase("books");
    }

    public Task<Book> AddAsync(CreateBook createBookEvent)
    {
        _logger.LogInformation("Adding book.\n{CreateBookEvent}", createBookEvent);

        return Task.FromResult(new Book("bookid", "title", "author"));
    }

    private readonly ILogger<MongoDbBookRepository> _logger;
    private readonly IMongoDatabase _db;
}
