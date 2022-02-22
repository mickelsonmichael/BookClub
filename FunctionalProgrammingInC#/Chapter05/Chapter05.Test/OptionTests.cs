using System.Collections.Specialized;
using Xunit;

namespace Chapter05.Test;

public class OptionTests
{
    [Fact]
    public void ImplicitConversion()
    {
        NameValueCollection empty = new();

#pragma warning disable CS8604
        Option<string> green = empty["green"];
#pragma warning restore CS8604

        Assert.Equal(Option<string>.None, green);
    }
}
