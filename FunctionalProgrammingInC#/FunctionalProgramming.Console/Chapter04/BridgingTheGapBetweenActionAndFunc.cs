using static System.Console;

using Unit = System.ValueTuple;

namespace FunctionalProgramming.Console.Chapter04;

public static class ActionExtensions
{
    public static Func<Unit> ToFunc(this Action action)
        => () =>
        {
            action();
            return default;
        };

    public static Func<T, Unit> ToFunc<T>(this Action<T> action)
        => (t) =>
        {
            action(t);
            return default;
        };
}

public static class BridgingTheGapBetweenActionAndFunc
{
    public static void Run()
    {
        Action act = () => WriteLine("Hello!");

        Func<Unit> f = act.ToFunc();

        f();
    }
}
