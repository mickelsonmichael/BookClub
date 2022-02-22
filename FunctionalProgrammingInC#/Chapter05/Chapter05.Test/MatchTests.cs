using Xunit;

namespace Chapter05.Test;

public class MatchTests
{
    [Fact]
    public void Match_ReturnsNone()
    {
        Option<bool> option = Option<bool>.None;

        bool result = option.Match(
            () => true,
            (bool _) => false
        );

        Assert.True(result);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Match_ReturnsSome(bool input)
    {
        Option<bool> option = Option<bool>.Some(input);

        bool result = option.Match(
            () => false,
            (bool _) => true
        );

        Assert.True(result);
    }
}