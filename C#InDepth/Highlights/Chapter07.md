# Chapter 7 - C# 5 Bonus Features

## 7.1 Capturing variables in foreach loops

…[our earlier expansion of the foreach loop] If that iteration variable is captured in an anonymous function
for a delegate, then whenever the delegate is invoked, the delegate will use the current value of that single variable - pg 221 para. 3
- "As of C# 5, the specification for the `foreach` loop has been changed so that a new variable is introduced in each iteration of the loop... Note that this change affects only `foreach` loops."
  - pg. 222, para. 2
- "...a `for` loop initializer declares a local variable, it does so once for the whole duration of the loop. The syntax of the loop makes that model easy to see..."
  - pg. 222, para. 5

## 7.2 Caller information attributes

- "All three attributes[, `CallerFilePathAttribute`, `CallerLineNumberAttribute`, and `CallerMemberNameAttribute`], can be applied only to parameters, and they're useful only when they're applied to optional parameters with appropriate types."
  - pg. 223, para. 2
- "...if the call site can't provide the argument, the compiler will use the current file, line number, or member name to fill in the argument instead of taking the normal default value."
  - pg. 223, para. 2
- "The parameter types are almost always `int` or `string` in normal usage. They can be other types where appropriate conversions are available."
  - pg. 223, para. 3,, **NOTE**
- "The most obvious case in which caller information is useful is when writing to a log file."
  - pg. 224, para. 1
- But it’d be entirely reasonable to write your own extension methods for ILogger that use these attributes and create an appropriate state object to be logged. 
  - pg 224 para. 3
- The less obvious use of just one of these attributes, [CallerMemberName], may be obvious to you if you happen to implement INotifyPropertyChanged frequently 
  - pg 224, para. 6
- "NLog was the only logging framework I found with direct support and then only conditionally based on the target framework."
  - pg. 224, Footnote 1
- [Dynamically invoked members] If the member being invoked includes an optional parameter with a caller information attribute but the invocation doesn’t include a corresponding argument, the default value specified in the parameter is used as if the attribute weren’t present.  
  - pg 226, para. 5
- "...the compiler would have to embed all the line-number information for every dynamically invoked member just in case it was required, thereby increasing the resulting assembly size for no benefit in 99.9% of cases."
  - pg. 226, para. 6, **Dynamically Invoked Members**
- "...(indexers) is specified to use the name `Item` unless `IndexerNameAttribute` has been applied to the indexer."
  - pg. 228, para. 2, **Non-obvious Member Names**
- "The Roslyn compiler uses the names that are present in the IL for [calls from an instance constructor, static constructor, finalizer, or operator]: `.ctor`, `.cctor`, `Finalize`, and operator names such as `op_Addition`."
  - pg. 228, para. 3, **Non-obvious Member Names**
- [Implicit constructor implications] The language specification calls out constructors as an example in which caller member information isn’t provided by the compiler unless the call is explicit 
  - pg 228, - para 5
- [Query expression implications] the language specification calls out query expressions as one place where caller information is provided by the compiler even though the call is implicit. 
  - pg 229, para. 3
- The Main method uses reflection to fetch the attribute from all the places you’ve applied it. 
  - pg 231, para. 3
- [in old versions of .NET]  …use the Microsoft.Bcl NuGet package 
  - pg 232, para. 3
