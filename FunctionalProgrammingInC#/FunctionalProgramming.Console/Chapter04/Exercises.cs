using System.Collections.Specialized;
using System.Configuration;

using static System.Text.RegularExpressions.Regex;

namespace FunctionalProgramming.Console.Chapter04;

/// <summary>
/// These are the exercise questions from the previous edition, that were removed or moved in the latest edition.
/// </summary>
public static class Exercises
{
    // ==========
    // Question 1
    // ==========
    public enum DayOfWeek { Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday };

    public static Func<string, DayOfWeek?> ParseDay = (string str) => Enum.TryParse(str, true, out DayOfWeek result) ? result : null;

    // ==========
    // Question 2
    // ==========
    public class Optional<T>
    {

    }
    public static T? Lookup<T>(IEnumerable<T> ls, Func<T, bool> predicate)
    {
        try
        {
            // can't use FirstOrDefault because it will return a default value
            // and that default cannot be null for non-nullable types
            return ls.First(predicate);
        }
        catch (InvalidOperationException)
        {
            // have to check if T is nullable already
            if (Nullable.GetUnderlyingType(typeof(T)) != null)
            {
                return (T)Convert.ChangeType(null, typeof(T));
            }
            else
            {
                return (T?)Convert.ChangeType(null, typeof(T?));
            }
        }
    }

    // ==========
    // Question 3
    // ==========
    public record Email
    {
        public string Value { get; }

        public Email(string value)
        {
            if (!IsMatch(value, "^[A-Za-z]+@[A-Za-z]+\\.[A-Za-z]{2,3}$"))
            {
                throw new ArgumentException("Invalid email");
            }

            Value = value;
        }

        public static implicit operator string(Email e) => e.Value;
    }

    // ==========
    // Question 4
    // ==========

    // The `First<TSource>(IEnumerable<TSource>, Func<TSource,Boolean>)` function throws an `InvalidOperationException` if
    // no matching element is found instead of returning null.
    //
    // There's obviously a good argument for returning `null` instead of an exception, since
    // the `FirstOrDefault` methods exist.

    // ==========
    // Question 5
    // ==========

    public class AppConfig
    {
        readonly NameValueCollection source;

        public AppConfig() : this(ConfigurationManager.AppSettings) { }

        public AppConfig(NameValueCollection source)
        {
            this.source = source;
        }

        public T? Get<T>(string name)
        {
            if (!source.HasKeys() && !source.AllKeys.Any(k => k?.Equals(name, StringComparison.CurrentCultureIgnoreCase) ?? false))
            {
                return default;
            }

            return(T?)Convert.ChangeType(source.Get(name), typeof(T));
        }

        public T Get<T>(string name, T defaultValue)
        {
            if (!source.HasKeys() && !source.AllKeys.Any(k => k?.Equals(name, StringComparison.CurrentCultureIgnoreCase) ?? false))
            {
                return defaultValue;
            }

            return(T?)Convert.ChangeType(source.Get(name), typeof(T)) ?? defaultValue;
        }
    }
}