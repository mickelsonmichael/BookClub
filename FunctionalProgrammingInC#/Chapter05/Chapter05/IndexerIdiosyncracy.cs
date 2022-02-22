using System.Collections.Specialized;

using static System.Console;

namespace Chapter05;

public static class IndexerIdiosyncracy
{
    public static void Go()
    {
        try
        {
            var empty = new NameValueCollection();

            var green = empty["green"]; // returns null

            WriteLine("green!");

            var alsoEmpty = new Dictionary<string, string>();

            var blue = alsoEmpty["blue"]; // throws exception

            WriteLine("blue!");
        }
        catch (Exception ex)
        {
            WriteLine(ex.GetType().Name);
        }
    }
}
