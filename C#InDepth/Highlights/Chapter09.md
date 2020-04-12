# Chapter 9 - Stringy Features

## Intro

- it’s worth remembering that the strings themselves haven’t changed at all. Both features provide new ways of obtaining strings, but that’s all.
  - pg 252, para.1

## 9.1 - A recap on string formatting in .NET

- An *alignment*, which specifies a minimum width
  - pg 254, para. 2
- "The alignment and the format string are independently optional; you can specify either; both, or neither."
  - pg. 254, para. 4
- A comma in the format item indicates an alignment, and a colon indicates a format string
  - pg 254, para.4
- "If you need a comma in the format string, that's fine; there's no concept of a second alignment value."
  - pg. 254, para. 4
- "...alignment... specifies only a minimum width. If you wanted to write code that right-aligned every possible set of values, you'd have to work out how wide the biggest one would be first."
  - pg. 254, para. 7
- …*localization* is the task of making sure your code does the right thing for all your users, no matter where they are in the world
  - pg 255, para.3
- "Considering the world is basically round, it certainly seems to have a lot of nasty corner cases to handle."
  - pg. 255, para. 3
- "In .NET, the most important type to know about for localization purposes is `Culture-Info`."
  - pg. 255, para. 4
- "Often, you won’t see `CultureInfo` in a method signature, but instead the `IFormatProvider` interface, which `CultureInfo` implements"
  - pg. 255, para. 5
- "Even though the parameter is of type `IFormatProvider`, the value you pass as an argument is almost always `CultureInfo`."
  - pg. 256, para. 2
- "If you don’t specify a format provider, or if you pass `null` as the argument corresponding to an `IFormatProvider` parameter, `CultureInfo.CurrentCulture` will be used as a default."
  - pg. 257, para. 2
- "For machine-to-machine communication… you should use the _invariant culture_.. `CultureInfo.InvariantCulture`"
  - pg. 257, para. 4
  
## 9.2 - Introducing interpolated string literals

- with interpolated string literals, you specify the values and their formatting information inline
  - pg 258, para.2
- "The [interpolated string] expression can be any expression that computes a value."
  - pg. 259, para. 5
- "If the value implements the `IFormattable` interface, its `ToString(string, IFormatProvider)` method will be called; otherwise, `System.Object.ToString()` is used."
  - pg. 259, para. 4
- Verbatim string literals are typically used for the following: Strings breaking over multiple lines, Regular expressions (which use backslashes for escaping, quite separate from the escaping the C# compiler uses in regular string literals), Hardcoded Windows filenames
  - pg 260, para. 1
- "Although the difference between "carriagea-return" and "carraige-return line-feed separators" is irrelevant in most code, it's significant in verbatim string literals."
  - pg. 260, para. 2, __NOTE__
- "The order of the symbols matters. `$@"Text"` is a valid interpolated verbatim string literal, but `@$"Text" isn't.`
  - pg. 260, para. 5, __TIP__
- "The compiler... converts the interpolated string literal into a call to `string.Format`..."
  - pg. 261, para. 1
- "...unlike regular or verbatim string literals, interpolated string literals don't count as constant expressions."
  - pg. 261, para. 5

## 9.3 - Localization using FormattableString

- **FormattableString** a class in the System namespace introduced in .NET 4.6 (and .NET Standard 1.3 in the .NET Core world). It holds the composite format string and the values so they can be formatted in whatever culture you want later.
  - pg 262, para. 2
- there are conversions from interpolated string literal expressions to both FormattableString and IFormattable
  - pg 262, para. 4
- But instead of string.Format, it calls the static Create method on the System.Runtime.CompilerServices.FormattableStringFactory class.    - pg 263, para. 1
- `FormattableString formattableString = $"Jon was born on {dateofBirth:d}"; var result = formattableString.ToString(culture);`
  - pg. 262, para. 2
- "When the compiler needs to convert an interpolated string literal into a `FormattableString`... instead of `string.Format`, it calls the static `Create` method on the `System.Runtime.CompilerServices.FormattableStringFactory` class."
  - pg. 263, para. 1
- FormattableString is an abstract class…
  - pg 263, para. 3
- You want to use the invariant culture to avoid any unexpected results from using the default culture
  - pg 264, para. 2
- `string parameter3 = FormattableString.Invariant($"x={date:yyy-MM-dd}"); string parameter4 = Invariant($"x={date:yyyy-MM-dd"}");`
  - pg. 264, __Listing 9.6__
- If you want to format a FormattableString in any culture other than the invariant one, you need to use one of the ToString methods.
  - pg 265, para. 2
- …so any method accepting an IFormattable will accept a FormattableString
  - pg 265, para. 3
- "Just because you're using a C# 6 compiler doesn't mean you're necessarily targeting a modern version of the framework."
  - pg. 268, para. 4
- "If you want to take advantage of the benefits of `FormattableString` but you're stuck targeting an earlier version of the framework, you can implement both types yourself."
  - pg. 268, para. 5

## 9.4 - Uses, guidelines, and limitations

- almost anywhere you’re already using string formatting with hardcoded composite format strings or anywhere you’re using plain string concatenation you can use interpolated strings.
  - pg 270, para. 3
- Interpolated string literals aren’t dynamic.
  - pg 270, para. 4
- You need to keep track of what level of nesting each part of your code is working at.
  - pg 271, para. 1
- If you know that all the developers reading these strings are going to be in the same non-English culture, it’s entirely reasonable to write all those messages in that culture instead.
  - pg 271, para. 5
- you can’t use interpolated string literals for resource files.
  - pg 272, para. 2
- Hard limitations of interpolated string literals: **NO DYNAMIC FORMATTING, NO EXPRESSION REEVALUATION, NO BARE COLONS**
  - pp 272, 273 section 9.4.2
- …two primary reasons not to use them even where you can. **[DEFER FORMATTING FOR STRINGS THAT MAY NOT BE USED]** **[FORMAT FOR READABILITY]**
  - pg 273, para. 6
- ...it forces the string to be formatted even if it’s just going to be thrown away, because the formatting will unconditionally be performed before the method is called rather than within the method only if it’s needed.
  - pg 274, para. 4
- …they can make the code harder to read
  - pg 274, para. 6

## 9.5 - Accessing identifiers with nameof

- The nameof operator … it takes an expression referring to a member or local variable, and the result is a compile-time constant string with the simple name for that member or variable.
  - pg 275, para. 5
- In terms of syntax, the nameof operator is like the typeof operator, except that the identifier in the parentheses doesn’t have to be a type.
  - pg 275, para. 6
- So why is it better to use nameof? In one word, robustness.
  - pg 276, para. 1
- If you rename it in a refactoring-aware way, your nameof operand will change, too.
  - pg 276, para. 2
- nameof makes it easier than ever to perform robust validation with informational messages.
  - pg 277, para. 3
- It’s useful to be able to raise the event and specify the name of the property in a safe way
  - pg 278, para. 1
- **Attributes** The nameof operator allows you to refer to that member in a safe way.
  - pg 279, para. 1
- [use in] object-relational mapping technologies such as Entity Framework
  - pg 279, para. 4
- **Referring to members of other types** qualify it with the type as well
  - pg 280, para. 2
- "I prefer to always use the type name where possible; if you use a variable instead, it _looks_ like the value of the variable may matter, but really it's used only at compile time to determine the type."
  - pg. 280, para. 4
- ...anonymous type, there’s no type name you could use, so you have to use the variable.
  - pg 280, para. 4
- "With `nameof`, the type argument must be specified but isn't included in the result. Additionally, there's no indication of the number of type parameters in the result: `nameof(Action<string)` and `nameof(Action<string, string>)` both have a value of just '`Action`'."
  - pg. 281, para. 2
- "It doesn't matter how you call the method: `Method<Guid>()` or `Method<Button>()` will both return '`T`'."
  - pg. 281, para. 5
- You can use a type parameter with a nameof operator, but unlike typeof(T), it’ll always return the name of the type parameter rather than the name of the type argument used for that type parameter at execution time
  - pg 281, para. 4
- "The `nameof` operator can't be used with any of the predefined aliases (`int`, `char`, `long`, and so on) or the `?` suffix to indicate a nullable value type or array types."
  - pg. 282, para. 1
- you have to use the CLR type name for the predefined aliases and the `Nullable<T>` syntax for nullable value types
  - pg 282, para. 2
- [infoof](http://mng.bz/6GVe)
  - pg 282, para. 4
- One point to note about what’s returned—the _simple name_ or “bit at the end” in less specification-like terminology
  - pg 282, para. 4
- "`nameof(Guid)`.. `nameof(System.Guid)`... result will still be only "`Guid`".
  - pg. 282, para. 5
- ...you can take the name of a namespace.
  - pg 282, para 5
- "If you use `nameof(System.Collections.Generic)`...it's just `Generic`."
  - pg. 283, para. 1
