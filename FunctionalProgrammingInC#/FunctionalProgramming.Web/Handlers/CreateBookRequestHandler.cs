using FunctionalProgramming.Web.Domain;
using FunctionalProgramming.Web.Events;
using FunctionalProgramming.Web.Requests;
using FunctionalProgramming.Web.Services;
using static Microsoft.AspNetCore.Http.Results;

namespace FunctionalProgramming.Web.Handlers;

public partial class Handlers
{
    public static Func<CreateBook, Task<IResult>> ConfigureHandleBookRequest(IServiceProvider services)
    {
        var repo = services.GetRequiredService<IBookRepository>();

        return HandleCreateBookRequest(repo);
    }

    public static Func<CreateBook, Task<IResult>> HandleCreateBookRequest(IBookRepository repo)
        => async req
        =>
        {
            DateTime timestamp = DateTime.Now;

            Guid bookId = Guid.NewGuid();

            CreatedBook evt = new(
                bookId,
                timestamp,
                req.Title,
                req.Author
            );

            await repo.AddAsync(evt);

            Book book = Book.Create(evt);

            return Created($"/books/{bookId}", book);
        };
}