namespace FunctionalProgramming.Notes.Chapter09.Examples;

public static class ApplyExtension
{
    private static Func<T2, TResult> Apply<T1, T2, TResult>(
        this Func<T1, T2, TResult> f,
        T1 t1
    ) => t2 => f(t1, t2);

    private static Func<T2, T3, R> Apply<T1, T2, T3, R>(
        this Func<T1, T2, T3, R> f,
        T1 t1
    ) => (t2, t3) => f(t1, t2, t3);
}
