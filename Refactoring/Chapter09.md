# Chapter 9 - Organizing Data

## Split Variable

Possibly the least useful "refactoring" in my opinion; essentially it is the "splitting up" of a variable doing two functions into two separate variables. In my opinion, this should almost _never_ be necessary. If you've got to use this refactoring, then something has gone horribly, horribly wrong. I almost want a new word for it instead of refactoring, perhaps "fixing" or "repairing" because this practice is horrid. Best way to really demonstrate it is to show an example of the root issue.

```c#
var myPerson = db.GetPerson(1);

// do something with myPerson

myPerson = myPerson.Manager; // assignment of myPerson again

// do something with the "new" myPerson
```

## Rename Field

Super straightforward, I almost didn't read the "motivation" behind this one. Just renaming a field or property of a class, record, or struct.

## Replace Derived Variable with Query

Compared to the previous refactoring, _Replace Derived Variable with Query_ is substantially more difficult to get your head around (although that isn't saying much). But, more or less it is replacing a property of a class that is calculated when another property is set, with a query instead. In the example below, every time `Discount` is set, the `DiscountedTotal` is also updated, which can lead to issues and unexpected side-effects if things are changed. It is better to change `DiscountedTotal` to be an equation that is calculated each time the property is requested.

```c#

public double DiscountedTotal { get; private set; }

private double discount;
public double Discount
{
	set
	{
		var old = DiscountedTotal;
		discount = value;
		DiscountedTotal += old - value;
	}
}

// REFACTORED TO

private double baseTotal;
private double discount;
public double DiscountedTotal => baseTotal - discount;

public double Discount
{
	set
	{
		discount = value;
	}
}
```

## Change Reference to Value

_ALERT: There has been yet another Domain-Driven Design reference in this chapter._ This time Fowler references _value types_, in that if he has a property in a class that would be better suited as a value type instead of a reference type, he switches it. This is primarily done so that the data-sharing of a class is reduced; if the property were to be a reference type and that type were to be shared with other instances, then a change in one instance could potentially affect the other instances. By changing to the immutable value objects or value types, instances won't share the same data, and you can safely modify one without worrying about changes occuring in the other.

## Change Value to Reference

The inverse of [Change Reference to Value](#change-reference-to-value), this refacotring involves changing a value type into a reference type to encourage sharing of data. The example in the book is of a series of orders with a single customer; an update to `order1.customer.lastname` should be reasonably reflected in the `order2.customer.lastname` instance (assuming they are from the same customer). In the actual code representation of the example, Fowler simply changes `let customer = new Customer(customerData);` into `let customer = customerRepository.get(customerData.id);`, which centralizes the reference to the customer in one location (the repository).

## Replace Magic Literal

This section wasn't in the physical copy of the book, but seeing as it was on the website, I decided to add the notes for it here. Part of the reason for this decision is because I like this refactoring quite a bit; in my opinion, using a static number anywhere in your code has the chance to be a _huge_ issue. For example, in the Nasdaq project I've been working on, we create buffers with the default size of 1024, which I know has a semantic meaning, but simply 1024 doesn't help me understand the meaning behind it. In addition, in the merge request for the code involving that 1024 constant, I accidentally put 1025 in place of a 1024 in one spot. Luckily, my co-worker caught it during the review, but if he hadn't , I have no idea what the ramifications of that mistake may have been (possibly just a slightly less efficient code, but I can't say), but what I do know is that if that code was wrapped in a constant like `public const int Kilobyte = 1024` then it would have been nearly impossible to make a typo like 1025. Between the IDE and the compiler, the issue would have been caught before the review had even began, and less time would have been wasted by fewer people (sorry Graham).

So yeah, use your constants.
	
