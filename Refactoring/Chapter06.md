# Chapter 06 - A First Set of Refactorings

## Encapsulate Variables

This refactoring appears to be when you encapsulate an object, or data, in a method instead of using the object directly. The example given is a default value `const defaultOwner = { firstName: "martin", lastName: "fowler" };`, which can then be refactored to a _method_ instead, like `function defaultOwner() { return defaultOwner; }`. This will also allow for the default to be modified using methods like `function setDefaultOwner(arg) { defaultOwner = arg; }`.

This is particularly useful for mutable data, since validations can be wrapped around the data. By calling `setDefaultOwner` you can verify that the data that is passed in is valid.

For C#, this isn't a particularly useful refactoring, since it is essentially built-in to most public variables. C# provides the automatic getters and setters, as well as the ability to expand those getters and setters to include validation logic. Therefore, it is much more idiomatic in C# to keep even mutable data as a public property and have it wrapping around a `private` field, like below.

```c#
public class MyClass
{
	private int _myField;
	public int MyField
	{
		get;
		set
		{
			if (value < 1) throw new InvalidOperationException("that's not valid");
			_myField = value;
		}
	}
}
```

However, Fowler does expand that this could be useful even for `private` fields, although personally I can't see that it would be necessary in most cases.

## Rename Variable

This one is very straightforward, it really doesn't necessitate much discussion. It simply is renaming a variable to a different, ideally more descriptive name.

## Introduce Parameter Object

Often you will see methods with dozens of parameters, which can be cumbersome to read and lead to multiple confusions when working with the method call, even with an IDE. This one is simply just moving those parameters into a single object.

The best example I can think of is probably more of a fault of the original developer than anything. It was the update and insert methods, which instead of taking in the orginal object (like a patient object), it took in all the unique properties of that object. Add on to that that the program was written in VB.NET, and it was a gruesome bit of code.

## Combine Functions into Class

Of all the refactorings, this is likely the hardest to detect the potential for. While it may be obvious in some cases, it's not always evident, and it is also a particularly intense operation to perform. You'll need to replace all the calls to the methods to the new class.

That being said, I do think this is one of my more favorite refactorings that can have the largest impact. While the others are certainly important, they won't be as noticably helpful towards the interpretation of the program. The more you can condense down a class into just the core functionality, following the common SRP (Single-Responsibility Principle), the easier it is to read, understand, change, and use.

## Combine Functions into Transform

I'm uncertain we have a ton of uses for this one, since we don't often work with immutable data. The basic premise is to make a copy of the object, then return the modified object, leaving the original intact, generally in instances where there are multiple modifications done to this object.

## Split Phase

This one is a particulary difficult one to detect, in my opinion. It would definitely take a trained eye to see the potential for a _Split Phase_ refactoring quicly. Essentially, in a series of calculations, you are detecting several actions that can be grouped together into a single action. The example given in the book includes the act of calculating the shipping costs, and extracting that information out into a separate function.

Probably the most useful feature of this refactoring is the fact that the calculations can be made more *sequential*, and can be more easily tested at a granular level. 

