using FunctionalProgramming.Web.Handlers;
using FunctionalProgramming.Web.Services;

WebApplicationBuilder builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddScoped<IBookRepository, MongoDbBookRepository>();

WebApplication app = builder.Build();

app.UseHttpsRedirection();

var HandleCreateBookRequest = Handlers.ConfigureHandleBookRequest(app.Services);
app.MapPost("/books", HandleCreateBookRequest);

await app.RunAsync();
