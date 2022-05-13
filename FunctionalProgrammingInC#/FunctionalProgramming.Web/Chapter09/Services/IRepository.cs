using LanguageExt;

namespace FunctionalProgramming.Web.Chapter09.Services;

public interface IRepository<T>
{
    Option<T> Lookup(Guid id);
    Try<T> Save(T entity);
}
