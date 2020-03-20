# Chapter 2 - C# 2

## 2.1 Generics

- "I view array covariance as an early design mistake that's beyond the scope of this book. Eric Lippert wrote about this at <http://mng.bz/gYPv> part of his series of blog posts on covariance and contravariance."
  - pg. 22, para. 6, *NOTE*
- "A method declares its inputs as parameters, and they're provided by calling code in the form of arguments."
  - pg. 26, para. 2
- "The generic *arity* of a declaration is the number of type parameters it has."
  - pg. 28, para. 3
- "I've allready referred to the `IEnumerable<T>` interface introduced in >NET 2.0, but that's a distinct type from the nongeneric `IEnumerable` interface..."
  - pg. 28, para. 4
- "...you can write methods with the same name but a different generic arity, even if their signatures are otherwise the same..."
  - pg. 28, para. 4
- "Not all types or type members can be generic."
  - pg. 29, para. 6
- "[`List<int> firstTwo = new CopyAtMost<int>(numbers, 2)` and `List<int> firstTwo = CopyAtMost(numbers, 2)` are] exactly the same method call in terms of the IL the compiler will generate."
  - pg. 30, para. 8
- "...you either explicitly specify all the type arguments or none of them."
  - pg. 31, para. 1
- "(Don't be fooled by the use of the `class` keyword [when setting type constraints]; it can be any reference type, including interfaces and delegates.)"
  - pg. 33, para. 4

```c#
TResult Method<TArg, TResult>(TArg input)
    where TArg : IComparable<TArg>
    where TResult : class, new()
```

- *Code snippet above displays how each type parameter can have a unique set of constraints*
  - pg. 34, para. 4
- "The `List'1` indicates that this is a generic type called `List` with generic arity 1 (one type parameter)..."
  - pg. 36, para. 5

## 2.2 Nullable value types

- "`Nullable<ValueType>` [is invalid] (`ValueType` itself isn't a value type)"
  - pg. 41, para. 2
  - To be clear, `ValueType` is a class in C#
- "...progress doesn't come just from making it easier to write correct code; it also comes from making it harder to write broken code or making the consequences less severe."
  - pg. 41, para. 6, *NOTE*
- "The parameterized `GetValueOrDefault(T defaultValue)` method will return the value in the struct or the specified default value if `HasValue` is `false`."
  - pg. 41, para. 7
- "The `GetType()` method declared on `System.Object` is nonvirtual, and the somewhat complex rules around when boing occurs mean that if you call `GetType()` on a value type, it always needs to be boxed first."
  - pg. 43, para. 1
- "You can mix and match [the `?` suffix and `Nullable<T>`] however you like. The generated IL won't change at all."
  - pg. 43, para. 5
- "...these two lines are equivalent: `if (x != null)` [and] `if (x.HasValue)`"
  - pg. 44, para. 2
- "VB treats lifted operators far more like SQL, so the result of `x < y` is `Nothing` if `x` or `y` is `Nothing`."
  - pg. 48, para. 1, **(continued)**

## 2.3 Simplified delegate creation

## 2.4 Iterators

## 2.5 Minor features

- "The various parts [of partial types] are combined by the compiler as if they were all declared together."
  - pg. 67, para. 2
- "...the `partial` modifier... must be present in every part [of the `partial` class]."
  - pg. 67, para. 4
- "Different parts [of a partial class] can contribute different interfaces... implementation doesn't need to be in the part that specifies the interface."
  - pg. 67, para. 5
- "...if a partial method hasn't been implemented all calls to it are removed... [this] allows generated code to provide optional hooks for manually written code to add extra behavior."
  - pg. 68, para. 1

```c#
partial void OnConstruction();
partial void CustomizeToString(ref string text);
```

- *The above code snippet shows and example of "Partial method declarations"
  - pg. 68, Listing 2.23
- "Because [the partial method] is never implemented, it's completely removed by the compiler."
  - pg. 68, para. 3
- "[Static classes are] utility classes composed entirely of static methods."
  - pg. 69, para. 2
- "...extension methods... can be declared only in non-nested, nongeneric, static classes."
  - pg. 69, para. 4
- "... using a property feels more idiomatic in C#."
  - pg. 70, para. 2
- "...a *namespace alias qualifier*, ... is just a pair of colons... `WinForms::Button`..."
  - pg. 71, para. 5
- "I suggest using `::` anywhere you use a namespace alias."
  - pg. 71, para. 2
- "...[Namespace aliases help clarify] multiple types with the same name but in a different namespaces."
  - pg. 71, para. 5
- "...[extern aliases] could be specified in project options or on the compiler command line."
  - pg. 72, para. 1
- "... [the pragma prefix] `CS` [is] for the C# compiler."
  - pg. 72, para. 5
- "If you omit a specific warning identifier, *all* warnings will be disabled... I recommend against it in general."
  - pg. 72, para. 6
- "Fixed-size buffers can be used only in unsafe code and only within structs."
  - pg. 73, para. 2
- "...when internal code is exposed to another library that's version idependently, it takes on the same versioning characteristics as public code."
  - pg. 74, para. 4
