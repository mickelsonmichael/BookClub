using FunctionalProgramming.Web.Domain;
using FunctionalProgramming.Web.Events;

namespace FunctionalProgramming.Web.Services;

public interface IBookRepository
{
    public Task<Book> AddAsync(CreateBook createBookEvent);
}
