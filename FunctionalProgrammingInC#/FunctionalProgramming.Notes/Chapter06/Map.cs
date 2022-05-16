using LaYumba.Functional;

namespace FunctionalProgramming.Notes.Chapter06;

public static class MapExtension
{
    public static IEnumerable<TResult> Map<T, TResult>(
        this IEnumerable<T> list,
        Func<T, TResult> predicate)
    {
        foreach (T value in list)
        {
            yield return predicate(value);
        }
    }

    /// <summary>
    /// Map can be implemented by simply forwarding to <see cref="Enumerable.Select" />.
    /// This is potentially a savings because Linq may perform some optimizations depending on the
    /// underlying implementation of the IEnumerable (e.g. <see cref="List{T}"/>, <see cref="LinkedList{T}"/>).
    /// </summary>
    public static IEnumerable<TResult> Map2<T, TResult>(this IEnumerable<T> list, Func<T, TResult> predicate)
        => list.Select(predicate);

    interface Functor<Ft, T>
    {
        // F<R> Map<R>(Func<Ft, R> f);
    }

    // public struct Option<T> : Functor<Option, T>
    // {
    //     public Option<R> Map<R>(Func<T, R> f) => null;
    // }
}
