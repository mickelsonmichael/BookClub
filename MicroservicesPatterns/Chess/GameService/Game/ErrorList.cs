using System.Collections;

namespace GameService.Game;

public class ErrorList : IEnumerable<string>
{
    public bool IsEmpty() => _errors.Count == 0;

    public ErrorList Add(string errorMessage)
    {
        _errors.Add(errorMessage);

        return this;
    }

    public IEnumerator<string> GetEnumerator() => _errors.GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => _errors.GetEnumerator();

    public static ErrorList Empty() => new();

    private readonly HashSet<string> _errors = new();
}
