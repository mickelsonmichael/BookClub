
namespace Chapter05;

// important for shortand Some and None
using static Chapter05.Option<int>;

public static class Int
{
    public static Option<int> Parse(string str)
        => int.TryParse(str, out int result)
            ? Some(result)
            : None;
}