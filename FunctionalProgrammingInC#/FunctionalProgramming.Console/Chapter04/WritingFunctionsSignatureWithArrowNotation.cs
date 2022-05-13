using static System.Console;

namespace FunctionalProgramming.Console.Chapter04;

public static class WritingFunctionsSignaturesWithArrowNotation
{
    public static void Run()
    {
        // int -> string
        Func<int, string> oneInputOneOutput = (int i) => i.ToString();

        // () -> string
        Func<string> noInputOneOutput = () => "hello";

        // int -> ()
        Action<int> oneInputNoOutput = (int i) => WriteLine($"Give me {i}");

        // () -> ()
        Action noInputNoOutput = () => WriteLine("Hello, World!");

        // (int, int) -> int
        Func<int, int, int> twoInputsOneOutput = (int a, int b) => a + b;

        // (string, (bool -> bool)) -> bool
        Func<string, Func<bool, bool>, bool> higherOrder = (string message, Func<bool, bool> swapper)
            =>
        {
            WriteLine(message);
            return swapper(true);
        };
    }
}
