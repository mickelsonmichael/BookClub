# Chapter 3 - C# 3: LINQ and everything that comes with it

## 3.1 Automatically implemented properties

- "... the compiler provides the implementation [for automatically implemented properties]."
  - pg. 76, para. 3
- "There's still a field, but it's created for you automatically by the compiler and given a name that can't be referred to anywhere in the C# code."
  - pg. 76, para. 4

## 3.2 Implicit typing

- "Languages that are *dynamically typed* leave all or most of the binding to execution time."
  - pg. 77, para. 3
- "..the type is inferred by the compiler from the compile-time type of the value assigned to it."
  - pg. 78, para. 2
- "...`var` can be used for only local variables."
  - pg. 78, para. 7
- "I suggest you discuss [using explicitly typed variables vs implicitly typed variables] with other people who'll work with your code the most... , get a sense of everyone's comfort level, and try to abide by that."
  - pg. 79, para. 6

```c#
int[] array;
array = { 1, 2, 3, 4, 5 };
```

- *The above code shows an invalid use of implicitly typed arrays*
  - pg. 80, para. 1
- `array = new[] { 1, 2, 3, 4, 5 };`
  - pg. 80, para. 3
  - *Shows how you can use `new[]` instead of `new int[]`*
- `var array = new[,] { { 1, 2, 3 }, { 4, 5, 6 } };`
  - pg. 80, para. 4
  - *Shows how you can create a multidementional array shorthand*
- `new[] { null, null }` [results in an error since] no elements have types."
  - pg. 80, Table 3.1
- `new[] { 10, null }` [results in an] Error [since] Only candidate type is `int`, but there's no conversion from null to `int`.
  - pg. 81, Table 3.1
- "Implicitly typed arays are mostly a convenience... except for anonymous types, where the array type can't be stated explicitly even if you want to."
  - pg. 81, para. 1

## 3.1 Object and collection initializers

- "[Object and collection initializers] require types to be mutable"
  - pg. 81, para. 3
- "Syntactically, an object initializer is a sequence of *member initializers* within braces."
  - pg. 83, para. 2
- "Object initializers can be used only as part of a constructor call or another object initializer."
  - pg. 83, para. 4
- "Each element initializer is either a signel expression or a comma seperated list of expression also in curly braces."
  - pg. 84, para. 3
- "The compiler compilers [collection initializers] into a constructor call followed by a sequence of calls to an `Add` method..."
  - pg. 84, para. 5
- "The compiler treats each element initializer as a separate `Add` call."
  - pg. 85, para. 2
- "Collection initializers are valid only for types that implement `IEnumerable`..."
  - pg. 85, para. 5
- "The compiler never uses the implementation of `IEnumerable` when compiling a collection initializer."
  - pg. 85, para. 6
- "Of course, the larger the initialization expression becomes, the more you may want to consider separating it out."
  - pg. 86, para. 2

## 3.4 Anonymous types

- "...[the syntax is called] an *anonymous object creation expression*."
  - pg. 87, para. 1
- "The [onoymous object creation expression] would work if you used `object` instead [of `var`]..."
  - pg. 87, para. 1
- "Anonymous types allow us to express those one-off cases concisely without losing the benefits of static typing."
  - pg. 87, para. 3
- "...[when making an anonymous object and] copying a property or field from somewhere else and you're happy to use the same name. This syntax is called a *projection initializer*."
  - pg. 87, para. 4
- "...[with projection initializers,] instead of writing this `SomeProperty = variable.SomeProperty` you can just write this: `variable.SomeProperty`."
  - pg. 88, para. 3
- "The properties [of an anonymous object] are all read-only."
  - pg. 89, para. 2
- "[The compiler] overrides `GetHashCode()` and `Equals()` so that two instances [of an anonymous type] are equal if all their properties are equal."
  - pg. 89, para. 2
- "[The compiler] overrides `ToString()`... lists the property names and their values."
  - pg. 89, para. 2
- "If two anonymous object creation expressions use the same property names in the same order with the same property types in the same assebmbly [they are the] same type."
  - pg. 89, para. 2
- "...properties must have the same names and types and be in the same order for two anonymous object creation expressions to use the same type."
  - pg. 90, para. 3
- "In VB, only properties declared with the `Key` modifier [are used in determining equality and hash codes]. Nonkey properties are read/write and don't affect equality or hash codes."

## Lambda expressions

- "Lambda expressions shortened like `[`var = (x) => x.Console.WriteLine(x);`] are said to have *epxression bodies*, whereas lambda expressions using braces are said to have *statement bodies*."
  - pg. 92, para. 7
- "Lambda expressions don't have a type but are convertible to compatible delegate types..."
  - pg. 92, para. 8
- "All of these are captured variables, because they're variables declared outside the immediate context of the lambda expression."
  - pg. 94, para. 4
- "...lambda [expressions capture] the variables themselves, *not* the values of the variables at the point when the delegate is created."
  - pg. 95, para. 3
- "...if the delegate modified any of the captured variables, each invocation would see the results of the previous invocations."
  - pg. 97, para. 1
- "If you use a lambda expression in a performance-critical piece of code, you should be aware of how many objects will be created to support the variables it captures."
  - pg. 100, para. 3
