using System;
using Xunit;
using FunctionalProgramming.Console.Chapter04;

namespace FunctionalProgramming.Console.Test.Chapter04;

public class ConstrainingInputsWithCustomTypesTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(100)]
    [InlineData(42)]
    public void Age_AcceptsValidValue(int expected)
    {
        Age age = new(expected);

        Assert.Equal(expected, age.Value);
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(int.MaxValue)]
    [InlineData(121)]
    [InlineData(int.MinValue)]
    public void Age_RejectsInvalidValue(int expected)
    {
        Assert.Throws<ArgumentException>(() => new Age(expected));
    }

    [Fact]
    public void Age_GreaterThanOperator()
    {
        Age bigger = new(20);
        Age smaller = new(10);

        Assert.True(bigger > smaller);
    }

    [Fact]
    public void Age_LessThanOperator()
    {
        Age bigger = new(20);
        Age smaller = new(10);

        Assert.True(smaller < bigger);
    }

    [Fact]
    public void Age_GreaterThanIntOperator()
    {
        Age bigger = new(20);
        const int smaller = 10;

        Assert.True(bigger > smaller);
    }

    [Fact]
    public void Age_LessThanIntOperator()
    {
        const int bigger = 20;
        Age smaller = new(10);

        Assert.True(smaller < bigger);
    }

    // [Fact]
    // public void PhoneNumber_Invalid()
    // {
    //     PhoneNumber invalid = new(545_222_444);

    //     Assert.Equal(PhoneNumber.Invalid(), invalid);
    // }

    // [Fact]
    // public void PhoneNumber_Valid()
    // {
    //     PhoneNumber valid = new(555_123_4567L);

    //     Assert.NotEqual(PhoneNumber.Invalid(), valid);
    // }
}
