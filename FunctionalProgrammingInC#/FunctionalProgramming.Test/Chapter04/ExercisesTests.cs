using FunctionalProgramming.Notes.Chapter04;
using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Configuration;
using Xunit;

namespace FunctionalProgramming.Console.Test.Chapter04;

public class ExercisesTests
{
    [Theory]
    [InlineData("monday", Exercises.DayOfWeek.Monday)]
    [InlineData("MONDAY", Exercises.DayOfWeek.Monday)]
    [InlineData("tuesday", Exercises.DayOfWeek.Tuesday)]
    [InlineData("wednesday", Exercises.DayOfWeek.Wednesday)]
    [InlineData("thursday", Exercises.DayOfWeek.Thursday)]
    [InlineData("friday", Exercises.DayOfWeek.Friday)]
    [InlineData("saturday", Exercises.DayOfWeek.Saturday)]
    [InlineData("sunday", Exercises.DayOfWeek.Sunday)]
    public void Question1_ValidDayOfWeek(string str, Exercises.DayOfWeek expected)
    {
        Assert.Equal(expected, Exercises.ParseDay(str));
    }

    [Fact]
    public void Question1_InvalidReturnsNull()
    {
        Assert.Null(Exercises.ParseDay("invalid"));
    }

    [Fact]
    public void Question2_NoMatch()
    {
        int? result = Exercises.Lookup(new List<int>(), (int _) => true);

        Assert.Null(result);
    }

    [Fact]
    public void Question2_Match()
    {
        int? result = Exercises.Lookup(new List<int> { 1 }, (int i) => i % 2 == 1);

        Assert.Equal(1, result);
    }

    [Fact]
    public void Question3_SmartConstructor()
    {
        const string invalidEmail = "mike@mike";

        Assert.Throws<ArgumentException>(() => new Exercises.Email(invalidEmail));
    }

    [Fact]
    public void Question3_ImplicitCast()
    {
        Exercises.Email email = new("test@example.com");

        var isEmptyString = (string a) => string.IsNullOrWhiteSpace(a);

        Assert.True(isEmptyString(email)); // implicit conversion to string
    }

    [Fact]
    public void Question5_GetString()
    {
        NameValueCollection values = new();

        values.Add("key", "value");

        var config = new Exercises.AppConfig(values);

        Assert.Equal("value", config.Get<string>("key"));
    }
}