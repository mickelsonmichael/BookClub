
using BenchmarkDotNet.Attributes;

namespace FunctionalProgramming.Notes.Chapter06;

[MarkdownExporter]
public class MapBenchmarks
{
    [Benchmark]
    [ArgumentsSource(nameof(EnumerableImplementations))]
    public int[] Map(IEnumerable<int> list)
    {
        return list.Map(v => v * 3).ToArray();
    }

    [Benchmark]
    [ArgumentsSource(nameof(EnumerableImplementations))]
    public int[] Map2(IEnumerable<int> list)
        => list.Map2(value => value * 3).ToArray();

    public IEnumerable<IEnumerable<int>> EnumerableImplementations()
    {
        IEnumerable<int> baseEnumerable = Enumerable.Repeat(1, 1_000_000);

        yield return baseEnumerable.ToList();
        yield return baseEnumerable.ToArray();
        yield return baseEnumerable.ToHashSet();
        yield return new LinkedList<int>(baseEnumerable);
        yield return new Queue<int>(baseEnumerable);
        yield return new Stack<int>(baseEnumerable);
    }
}
