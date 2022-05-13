using System;
using Xunit;
using FunctionalProgramming.Console.Chapter04;

namespace FunctionalProgramming.Console.Test.Chapter04;

public class BridgingTheGapBetweenActionAndFuncTests
{
    [Fact]
    public void ToFunc_WhenCalled_PerformsAction()
    {
        bool isRan = false;
        Action testAction = () => isRan = true;

        var f = testAction.ToFunc();

        f();

        Assert.True(isRan);
    }

    [Fact]
    public void ToFunc_WhenCalled_WithARgument_PerformsAction()
    {
        bool isRan = false;
        Action<string> testAction = (string _) => isRan = true;

        var f = testAction.ToFunc();

        f("something");

        Assert.True(isRan);
    }
}
