
namespace Chapter05;

public static class DictionaryExtensions
{
    public static Option<T> Lookup<K, T>(this IDictionary<K, T> dict, K key)
       => dict.TryGetValue(key, out T? value) ? Option<T>.Some(value) : Option<T>.None;
}
