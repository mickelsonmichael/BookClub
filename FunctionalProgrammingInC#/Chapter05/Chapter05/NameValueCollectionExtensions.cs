
using System.Collections.Specialized;

namespace Chapter05;

public static class NameValueCollectionExtensions
{
#pragma warning disable CS8604
    public static Option<string> Lookup(this NameValueCollection nvc, string value)
        => nvc[value];
#pragma warning restore CS8604
}
