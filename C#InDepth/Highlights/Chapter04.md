# Chapter 4 - C# 4: Improving interoperability

- "[A language that is both statically and dynamically typed] is rare within programming languages."
  - pg. 113, para 1

## 4.1 Dynamic typing

- "[dynamic] lookups are performed at execution time."
  - pg. 114, para. 6
- "...the CLR doesn't know about [the dynamic type] at all... the IL uses `object` decorated with `[Dynamic]` if necessary."
  - pg. 115, para. 3
- "...the binder [can] use the user-defined implicit conversion from `string` to `XNamespace`."
  - pg. 116, para. 1
- "...*almost* all operations with dynamic values have a result that is also dynamic."
  - pg. 118, para. 5
- "[`ExpandoObject`] operates in two modes depending on whether you're using it as a dynamic value."
  - pg. 120, para. 3
- "When `ExpandoObject` is used in a statically typed context, it's a dictionary of name/value pairs, and it implements `IDictionary<string, object>` as you'd expect..."
  - pg. 120, para. 4
- "...[`ExpandoObject`] also implements `IDynamicMetaObjectProvider`."
  - pg. 120, para. 5
- "The `DynamicObject` class acts as a base class for types that want to implement dynamic behavior as simply as possible."
  - pg. 122, para. 7
- "If you return `false` [from the `DynamicObject.TryInvokeMember` or `DynamicObject.TryGetMember` methods], a `RuntimeBinderException` will be thrown
  - pg. 123, para. 4
- "The execution-time binder doesn't resolve extension methods."
  - pg. 128, para. 3, **Extension Methods**
- "...anonymous methods can't be assigned to a variable of type `dynamic`..."
  - pg. 129, para. 5, **Anonymous Functions**
- "...lambda expressions can't appear within dynamically bound operations."
  - pg. 129, para. 7, **Anonymous Functions**
- "...lambda expressions that are converted to expression trees must not contain any dynamic operations."
  - pg. 129, para. 9, **Anonymous Functions**
- "...anonymous types are generated as regular classes in IL by the C# compiler."
  - pg. 130, para. 4, **Anonymous Types**

## 4.2 Optional parameters and named arguments

- "Parameters with `ref` or `out` modifiers aren't permitted to have default values."
  - pg. 134, para. 4, **4.2.1 Parameters with default values and arguments with names**
- "An argument without a name is called a *positional argument*."
  - pg. 134, para. 4, **4.2.1 Parameters with default values and arguments with names**
- "A default expression, such as `default` [can be a default value for a parameter]."
  - pg. 135, para. 1, **4.2.1 Parameters with default values and arguments with names**
- "Named arguments can be specified in any order."
  - pg. 135, para. 5, **4.2.2 Determining the meaning of a method call**
- "...arguments are evaluated in the order they appear in the source code for the method call, left to right. In most cases, this wouldn't matter, but if argument evaluation has side effects, it can."
  - pg. 136, para. 3, **4.2.2 Determining the meaning of a method call**
- "...if the compiler has to specify any default values for parameters, those values are embedded in the IL for the calling code."
  - pg. 137, para. 2, **4.2.2 Determining the meaning of a method call**
- "A method that has no optional parameters without corresponding arguments is "better" than a method with at least one optional parameter without a corresponding argument. But a method with one unfilled parameter is no better than a method with two such parameters."
  - pg. 138, para. 3, **Adding Overloads is Fiddly**

## 4.3 COM interoperability improvements

## 4.4 Generic variance

- "*Covariance* occurs when values are returned only as output. *Contravariance* occurs when values are accepted only as input. *Invariance* occurs when values are used as input and output."
  - pg. 144, para. 4, **4.4.1 Simple examples of variance in action**
- "...the modifiers `in` and `out` are used to specify the variance of a type parameter..."
  - pg. 144, para. 7, **4.4.2 Syntax for variance in interface and delegate declarations**
- `public TResult Fnc<in T, out TResult>(T arg)`
  - pg. 145, para. 3, **4.4.2 Syntax for variance in interface and delegate declarations**
- "...variance isn't inherited by classes or structs implementing interfaces; classes and structs are always invariant."
  - pg. 145, para. 5, **4.4.3 Restrictions on using variance**
