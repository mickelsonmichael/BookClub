# Chapter 10 - Simplifying Conditional Logic

## Decompose Conditional

As Folwer mentions in the _Motivation_ section of this refactoring, this is a version of _Extract Function_ of particular significance. A lot of cognitive complexity in code comes from conditionals, and by extracting out the conditional logic into separate functions, you can make the code clearer and more concise.

I would like to see a few demonstrations of this with complex logic, because without local functions, I could see a lot of parameters needing to be passed in many scenarios.

This also reminds me of my favorite use-case for local functions, which is to abstract out information in LINQ statements. See below for an example.

### Example of my Local Method Strategy

```c#
var activeNotifications = context.Notifications
	.Where(x => x.ExpirationDate > DateTime.Today && x.IsRead);
```

Can be converted to

```c#
var activeNotifications = context.Notifications
	.Where(IsActive);

bool IsActive(Notification x)
{
	return x.ExpirationDate > DateTime.Today
		&& x.IsRead;
}
```

Obviously there are other ways you could handle this, like creating a `GetActiveNotifications` method instead, which is a great way to do it, but there are some, less contrived, cases where I prefer this local method strategy.

## Consolidate Conditional Expression

Back to simple refactorings, [Consolidate Conditional Expression](#consolidate-conditional-expression) is very straightforward. While all his refactoring examples are contrived to some extent, this is potentially the most head-scratching as to why the original code existed in the first place.

```javascript
if (anEmployee.seniority < 2) return 0;
if (anEmployee.monthsDisabled > 12) return 0;
if (anEmployee.isPartTime) return 0;
```

Why these three conditions would be three separate statements in the first place is unrealistic, and Fowler does the additional step of [Decompose Conditional](#decompose-conditional) which is just making this refactoring feel even more oddly redundant, even if it is completely necessary when this scenario occurs.

It is worth noting that Fowler mentions it may not always be in your best interest to extract the three conditions into a single statement; if those three statements are completely unrelated and may change in the future, then it's not always beneficial to combine them. Sometimes duplicate code is kept because it represents unique statements, even if they end up with the same result. This is one of the more difficult concepts for me to stomach; I get what it means in spirit, but at the same time I've been so ingrained with the concept of DRY that it is hard to let it slide. And to counter the argument with Refactoring itself, if you really ever needed to split the statements up again, doing so it relatively trivial, so why not combine them for now and save yourself the duplicated code?

## Replace Nested Conditional with Guard Clauses

I actually really like this refactoring, and I use the concept of guard clauses often (even if I didn't know the nomenclature before). General summary is that when there is a unique condition in a series of `if...else` statements, then it may be best to use a guard clause and return immediately from the function instead of returning at the end. I will often perform this refactoring myself already; if I find a scenario where code execution reaches the end of the method when it could have potentially returned earlier in some cases, I will add in those early return statements.
As Fowler argues, this makes the code clearer, more concise, and easier to reason about.

The way he puts it, this is a good use-case when the `if` or the `else` statements aren't part of the normal method flow; when they are part of something unusual occurring that could stop or divert execution early.

## Replace Conditional with Polymorphism

Some refactorings are a simple as renaming a variable, but this one feels much more involved than those others. For starters, if you don't already have an inheritance hierarchy set up for your solution, then you'll have to implement it, but even when you do then you'll have to update or create multiple instances of a method in multiple places. This could impact dozens of files potentially. I suppose, upon reflection, renaming a variable could also have far reaching impacts as well, so my reaction here may be blown out of proportion.

That overreaction probably is based on how infrequently I utilize polymorphism in my own code, and while that may be a case of me just having never worked with a system that lent itself to polymorphism well, it could also be my general inexperience with implementing the concepts that lead to my trepidation. I know how to implement the changes, but I'm hesitant to, since at this point it is very non-idiomatic to the way I program. But, almost certainly, its an area I need to explore and practice more.

## Introduce Special Case

This is similar to the constant refacatoring from the online version of the previous chapter. Replacing a "null" value with a more "static" and centralized version of the value. The example in the book is of a utility company with customers that my or may not be "unknown" and how the system can put in place an `UnknownCustomer` class (that in a statically typed language would inherit from `Customer`, but in Javascript this isn't necessary); this new class can then be returned instead of setting the properties of a `Customer` to some default values and forcing the system to check them in every instance.

## Introduce Assertions

As far as I am aware, assertions are not very common practice in C#, so this one is a little bit unusual for us, but I think it may be something we as C# developers should start adapting more. I did some reading on assertions in C# over at [this StackOverflow question](https://stackoverflow.com/questions/163538/c-sharp-what-does-the-assert-method-do-is-it-still-useful), so I may start sprinkling in more `Debug.Assert` calls here and there. One of the answers proposes that these calls only be done in private methods, and that everywhere else you'd generally want to do argument checking instead, since you always want to treat (potentially) foreign inputs as incorrect and throw an `ArgumentException` instead.

