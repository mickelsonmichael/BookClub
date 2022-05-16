using FunctionalProgramming.Web.Domain;
using FunctionalProgramming.Web.Events;
using FunctionalProgramming.Web.Services;
using Microsoft.AspNetCore.Mvc;

WebApplicationBuilder builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddScoped<IBookRepository, MongoDbBookRepository>();

WebApplication app = builder.Build();

app.UseHttpsRedirection();

app.MapPost("/books", async ([FromServices] IBookRepository repo, CreateBook createBookEvent) =>
{
    Book created = await repo.AddAsync(createBookEvent);

    return Results.Created($"/books/{created.BookId}", created);
});

await app.RunAsync();
