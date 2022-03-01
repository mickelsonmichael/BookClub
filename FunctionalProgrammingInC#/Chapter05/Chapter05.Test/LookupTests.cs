
using System.Collections.Generic;
using System.Collections.Specialized;
using Xunit;

namespace Chapter05.Test;

public class LookupTests
{
    [Fact]
    public void NameValueCollection_None()
    {
        NameValueCollection empty = new();

        Option<string> value = empty.Lookup("invalid");

        Assert.Equal(Option<string>.None, value);
    }

    [Fact]
    public void NameValueCollection_Some()
    {
        NameValueCollection collection = new();

        collection.Add("valid", "value");

        Option<string> value = collection.Lookup("valid");

        Assert.Equal(Option<string>.Some("value"), value);
    }

    [Fact]
    public void Dictionary_None()
    {
        Dictionary<string, string> empty = new();

        Option<string> value = empty.Lookup("invalid");

        Assert.Equal(Option<string>.None, value);
    }

    [Fact]
    public void Dictionary_Some()
    {
        Dictionary<string, string> dict = new();

        dict.Add("valid", "value");

        Option<string> value = dict.Lookup("valid");

        Assert.Equal(Option<string>.Some("value"), value);
    }
}
