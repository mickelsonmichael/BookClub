
using System.Net;

namespace FunctionalProgramming.Console.Chapter04;

public enum ValidationType
{
    NotFound,
    UserError
}

public record ValidationError(
    ValidationType type,
    string message
);

public class Service
{
    /// <summary>
    /// This is a message
    /// </summary>
    /// <exception cref="ArgumentException">This is what happends when not found</exception>
    public (int? item, IReadOnlyList<ValidationError> errors) Get(Func<int, bool> predicate)
    {
        var item = Enumerable.Range(0, 100)
            .FirstOrDefault(predicate, -1);

        if (item == -1)
        {
            return (null,
                new List<ValidationError>
                {
                    new ValidationError(ValidationType.NotFound, "not found")
                });
        }

        return (item, new List<ValidationError>());
    }
}

public static class ServiceController
{
    public static HttpStatusCode GetServices(int i)
    {
        var (item, errors) = new Service().Get(n => n == i);

        if (errors.Any(e => e.type == ValidationType.NotFound))
        {
            return HttpStatusCode.NotFound;
        }

        return HttpStatusCode.OK;
    }
}
