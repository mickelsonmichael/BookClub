
namespace Chapter05;

// allows convenient use of Some and None
using static Chapter05.Option<Age>;

public struct Age
{
    public int InYears() => _value;
    public int InMonths() => _value * 12;

    public static Option<Age> Create(int age)
       => IsValid(age) ? Some(new Age(age)) : None;

    private Age(int value)
       => _value = value;

    private static bool IsValid(int age)
       => 0 <= age && age < 120;

    private readonly int _value;
}
