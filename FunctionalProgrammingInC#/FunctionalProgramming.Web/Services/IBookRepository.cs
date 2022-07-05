using FunctionalProgramming.Web.Events;

namespace FunctionalProgramming.Web.Services;

public interface IBookRepository
{
    public Task AddAsync(CreatedBook createBookEvent);
}
