
namespace FunctionalProgramming.Core;

/// <summary>
/// <para>
/// The author points out that the TypeScript unions feature would be useful for this case,
/// but currently a similar feature is not available in C# 10.
/// However, there are ongoing discussions about it being added to C# 11 in November,
/// but nothing is set in stone quite yet.
/// </para>
/// <para>
/// https://www.typescriptlang.org/docs/handbook/2/everyday-types.html#union-types
/// https://github.com/dotnet/csharplang/discussions/399
/// https://github.com/dotnet/csharplang/blob/main/meetings/2021/LDM-2021-08-30.md#discriminated-unions
/// </para>
public abstract record Option<T>
{
    public static readonly NoneType None;

    public static Option<T> Some(T val) => new Some<T>(val);

    public static implicit operator Option<T>(NoneType _)
        => new None<T>();

    public static implicit operator Option<T>(T? value)
        => value is null ? None : Some(value);
}

public record None<T> : Option<T>;

public record Some<T> : Option<T>
{
    private T Value { get; }

    public Some(T value) => Value = value ?? throw new ArgumentNullException(nameof(value));

    public void Deconstruct(out T value) => value = Value;
}

public struct NoneType {}
