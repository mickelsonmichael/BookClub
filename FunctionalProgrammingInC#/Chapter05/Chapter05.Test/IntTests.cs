using Xunit;

namespace Chapter05.Test;

public class IntTests
{
    [Fact]
    public void Parse_Valid()
    {
        const string n = "1234";

        Option<int> result = Int.Parse(n);

        Assert.Equal(Option<int>.Some(1234), result);
    }

    [Fact]
    public void Parse_Invalid_Alpha()
    {
        const string str = "hello moto";

        Option<int> result = Int.Parse(str);

        Assert.Equal(Option<int>.None, result);
    }

    [Fact]
    public void Parse_Invalid_TooBig()
    {
        string huge = long.MaxValue.ToString();

        Option<int> result = Int.Parse(huge);

        Assert.Equal(Option<int>.None, result);
    }
}
