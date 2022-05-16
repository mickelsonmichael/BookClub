using static System.Console;

// https://louthy.github.io/language-ext/LanguageExt.Core/Prelude/index.html
using static LanguageExt.Prelude;

namespace FunctionalProgramming.Notes.Chapter09.Examples;

public static class Currying
{
    // 9.3 Curried functions: Optimized for partial application

    // <definition>
    // Currying
    // - Transforming an n-ary function with arguments t1,t2,...,tn and turning it into a
    //   unary function that takes t1 and returns a new function that takes t2, and so on.
    //
    // - Before: (T1, T2, ..., Tn) -> R
    // - After: T1 -> T2 -> ... -> Tn -> R
    // </definition>

    // (T1, T2) -> TResult
    //
    // CHANGES TO
    // 
    // T1 -> T2 -> TResult
    public static Func<T1, Func<T2, TResult>> Curry<T1, T2, TResult>(
        this Func<T1, T2, TResult> f
    ) => t1 => t2 => f(t1, t2);

    // (T1, T2, T3) -> TResult
    //
    // CHANGES TO
    //
    // T1 -> T2 -> T3 -> TResult
    public static Func<T1, Func<T2, Func<T3, TResult>>> Curry<T1, T2, T3, TResult>(
        this Func<T1, T2, T3, TResult> f
    ) => t1 => t2 => t3 => f(t1, t2, t3);

    // Partial Application vs currying
    // 
    // Partial application = giving arguments
    //
    // Currying = transforming without providing arguments, optimizes to make partial application easier

    public static void Example()
    {
        var greet = (string greeting, string name) => $"{greeting}, {name}";

        var greetWith = curry(greet); // LanguageExt uses LanguageExt.Prelude.curry as a static function

        var greetFrench = greetWith("Hallo");

        new[] { "Mike", "Lyn" }
            .Map(greetFrench)
            .Iter(WriteLine);
    }
}