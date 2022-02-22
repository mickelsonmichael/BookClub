
namespace Chapter05;

public static class MatchExtension
{
    public static TResult Match<T, TResult>(this Option<T> opt, Func<TResult> None, Func<T, TResult> Some)
        => opt switch
        {
            None<T> => None(),
            Some<T>(T val) => Some(val),
            _ => throw new ArgumentException($"{nameof(Option<T>)} must be {nameof(None)} or {nameof(Some)}.")
        };
}
