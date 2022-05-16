using LaYumba.Functional;
using static LaYumba.Functional.F;

namespace FunctionalProgramming.Notes.Chapter10;

public class Testing
{
    public void Do()
    {
        // T -> R
        var multiply = (int x, int y) => x * y;

        // F<T>
        Option<int> functor = Some(2);

        // F<T> -> (T -> R) -> F<R>
        Option<Func<int, int>> result = functor.Map(multiply);

        var final = result.Apply(4);

        var final2 = Some(multiply)
            .Apply(2)
            .Apply(4);
    }
}
