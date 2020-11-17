# Encapsulation

## Encapsulate Record

Essentially a conversion from `struct` to `class`, the _Encapsulate Record_ refactoring is storing the values of a record field into a `class` with getter and setter methods instead, allowing for increased security in the form of validation, and an increased ability to perform further refactorings.

This isn't a particularly huge concern for C#, in my opinion, since until recently we haven't had access to record types (they were released a few days ago, November 2020, as part of C# 9), and those record types are immutable, and therefore wont' benefit from getters and setters. This is mostly important for `struct` implementations, but personally I don't often create structs. I find that I don't understand the implications of a struct when it comes to the heap and the stack well enough to properly utilize them, so I reach for a class instead.

## Encapsulate Collection

I've seen this particular refactoring (or the end result of it) referenced a lot this year. It has been brought up several times in the Nasdaq Book Club when speaking with Graham about Domain-Driven Design, and, if I recall correctly, it was also mentioned in the _Entity Framework Core in Action_ book the LMCU Book Club recently read.

The jist is that it is removing the direct modification of a list within a class, and favoring `add` and `remove` ethods instead, like below.

This encapsulation obviously allows for a better management of what is added to the list and _how_ it is added. Which may not immediately sound beneficial, but as Fowler alludes to, it is much easier to make future changes to the list validation in the future using this method. It also allows you to return a copy of this list instead of the original, to prevent accidental mutation of the underlying list.

```c#

public class BookClub
{
	private readonly List<string> _books = new List<string>();

	public IList<string> Books => new List<string>(_books);
	
	public void AddBook(string book)
	{
		_books.Add(book);
	}

	public void RemoveBook(string book)
	{
		_books.Remove(book);
	}
}
```

## Replace Primitive with Object

Martin argues that this is one of the most popular refactorings in his toolkit. Changing an `int` or a `string` representation of something, like a telephone number, into an object representation. This has numberous benefits, probably most importantly, the ability to add validations.

This is one I look forward to using more often. My latest exposure to C++, where classes can often be treated more like primitives with operator overloading has given me a greater appreciation for working with data using encapsulated ways. Particularly interesting is the phone number example, since I've battled with phone numbers in the past, and having a class would give a sense of organization to the concept that I hadn't used before.

## Replace Temp with Query

This refactoring is more of a transitional method than anything, it doesn't really have much use besides working as a stepping stone towards further refactoring. It's straightforward enough, replace a calculation stored in a temporary variable with a calculation done in a method instead. This allows for more logic to be extracted out into the new method at a later date.

## Extract Class

This one is very similar to _Extract Functions to Class_ from Chapter 6. Effectively you are removing several related fields in a class into a new, separate class. The example given, is removing several telephone-related properties (e.g. number, area code) and extracting them into a new class with those given proprties. Then the old class can simply reference the new class as a single property instead of as a series of properties.

This helps further simplify classes and clean things up. The phone number class can be concerned with phone numbers and the person class doesn't have to worry about all the requirements of a phone number, only of the person. This also allows the class to be reused in more areas and for the reduction of duplicated code.

## Inline Class

The inverse of _Extract Class_, _Inline Class_ is often a step performed after a different series of refactorings, once a class has had most of its useful functionality removed, and it only acts as a shell for the properties now.

The other case where Fowler recommends using it is when you want to refactor two classes and divvy up their responsibilities in a new way.

## Hide Delegate

An example is easier than a description for this one. In the case of an employee in a particular department which has a particular manager, you may be able to access that manager via `const manager = employee.department.manager;`, but instead, there could be a method for this logic like `function getManager() { return this.department.manager; }`, which could then be accessed with `const manager = employee.getManager();`.

This has the benefit of allowing the code to retrieve the manager be modified in the future without having to change it in a lot of places; there's just one spot where the logic occurs so it is easy to change.

## Remove Middle Man

Another inverse method, _Remove Middle Man_ undoes _Hide Delegate_ (see _Hide Delegate_ for the code examples. Fowler says that this is mostly useful when people take the "Law of Demeter" too far; when too many things have middle-men in the way, they get cumbersome and difficult to deal with. As he puts it, "good encapsulation six months ago may be awkward now."

I've heard the Law of Demeter several times, so I used [Wikipedia](https://en.wikipedia.org/wiki/Law_of_Demeter) to shake the cobwebs off of my memory banks. The Law of Demeter is essentially the rule of loose coupling; your objects should assume as little as possible about the other objects they interact with.

## Substitute Algorithm

While a little vauge, this one is at least somewhat straightforward. This one is focused on removing inferior or outmoded algorithms in favor of new ones; Folwer gives the example of finding a more efficient algorithm, or replacing an algorithm with a library method call instead.

There isn't really a simple guide to doing this one, the instructions are basically "delete the old algorithm and paste in the new." Which means it is particularly important that your algorithm be well tested, and those tests be easy to run, before you go replacing algorithms.

