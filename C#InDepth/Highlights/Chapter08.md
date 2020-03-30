# Chapter 08 - Super-sleek Properties and Expression-Bodied Members

## 8.1 - A brief history of properties

- "[Jon prefers] fields to be private in almost all cases."
  - pg 237, para 1

## 8.2

- "[`public double X { get; }]` Declares read-only automatically implemented properties."
  - pg 238, Listing 8.6
- "...the field declared by the automatically implemented property is read-only, and any assignments to the property are translated by the compiler into direct field assignments."
  - pg 239, Para 1
- "Any attempt to set the property in code other than the constructor results in a compile-time error."
  - pg 239, para 1
- "One common pattern is to have a read-only property exposing a mutable collection, so a caller can add or remove items from the collection but can never change the property to refer to a different collection (or set it to be a null reference)."
  - pg 240, para 2
- "You can't use any properties, methods, indexers, or events in a struct until the compiler considers that all the fields have been definitely assigned."
  - pg 241, para 4
- "Every struct constructor must assign values to all fields before it returns control to the caller."
  - pg 241, para 4
- "Setting the `X` and `Y` properties still counts as using the value of the struct..."
  - pg 241, para 5, 8.3.2 Automatically implemented properties in structs
- "Chaining to the default constructor is a workaround because that assigns all fields before your constructor body executes."
  - pg 241, para 5, 8.2.3 Automatically implemented properties in structs
- "You're allowed to set an automatically implemented property before all the fields are initialized [in C# 6.]"
  - pg 241, para 6, 8.2.3 Automatically implemented properties in structs
- "Setting an automatically implemented property counts as intializing the field [in C# 6.]"
  - pg 241, para 6, 8.2.3 Automatically implemented properties in structs
- "You're allowed to read an automatically implemented property before other fields are initialized, so long as you've set it beforehand

## 8.3

- "Expression-bodied members use the `=>` syntax but aren't lambda expressions."
  - pg 244, **No, this isn't a lambda expression**
- "When talking about the syntax out loud, [Jon] usually [describes] `=>` as a *fat arrow*."
  - pg 244, **No, this isn't a lambda expression**
- "Expression-bodied properties have one downside: there's only a single-character difference between a read-only property and a public read/write field [(`=>)]."
  - pg 245, **Important caveat**
- "In general, I put the `=>` at the end of the declaration part and indent the body [of the expression-bodied member] as usual."
  - pg 246, para 2, 8.3.2 Expression-bodied methods, indexers, and operators
- "...there's no such thing as an expression-bodied constructor in C#6... You can't have expression-bodied [static constructors, finalizers, instance constructors, read/write or write-only properties, read/write or write-only indexers, and events]."
  - pg 247, para 4, 8.3.3 Restrictions on expression-bodied members in C# 6
- "One nice aspect of [C# 7] is that the `get` accessor can be expression-bodied even if the `set` accessor isn't, or vice versa."
  - pg 248, para 1, 8.3.3 Restrictions on expression-bodied members in C# 6
- "...one validation check puts a method on the borderline for expression-bodied members; with two of them, it's just too painful."
  - pg 250, para 2, 8.3.4 Guidelines for using expression-bodied members

